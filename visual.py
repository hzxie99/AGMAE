
import cv2
import torch
# 1. 定义视频路径和图像保存路径
video_path = "path/to/video.mp4"
output_image_dir = "path/to/output_images/"
output_mask_dir = "path/to/output_masks/"
@torch.no_grad
def predict(self, x):
    self.eval()

    device = x.device
    b, c, h, w = x.shape

    '''i. Patch partition'''

    num_patches = (h // self.patch_h) * (w // self.patch_w)
    # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
    patches = x.view(
        b, c,
        h // self.patch_h, self.patch_h,
        w // self.patch_w, self.patch_w
    ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

    '''ii. Divide into masked & un-masked groups'''

    num_masked = int(self.mask_ratio * num_patches)

    # Shuffle
    # (b, n_patches)
    shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

    # (b, 1)
    batch_ind = torch.arange(b, device=device).unsqueeze(-1)
    mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

    '''iii. Encode'''

    unmask_tokens = self.encoder.patch_embed(unmask_patches)
    # Add position embeddings
    unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
    encoded_tokens = self.encoder.transformer(unmask_tokens)

    '''iv. Decode'''

    enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

    # (decoder_dim)->(b, n_masked, decoder_dim)
    mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
    # Add position embeddings
    mask_tokens += self.decoder_pos_embed(mask_ind)

    # (b, n_patches, decoder_dim)
    concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
    # dec_input_tokens = concat_tokens
    dec_input_tokens = torch.empty_like(concat_tokens, device=device)
    # Un-shuffle
    dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
    decoded_tokens = self.decoder(dec_input_tokens)

    '''v. Mask pixel Prediction'''

    dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
    # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
    pred_mask_pixel_values = self.head(dec_mask_tokens)

    # 比较下预测值和真实值
    mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
    mse_all_patches = mse_per_patch.mean()

    print(
        f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
    print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')

    '''vi. Reconstruction'''

    recons_patches = patches.detach()
    # Un-shuffle (b, n_patches, patch_size**2 * c)
    recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
    # 模型重建的效果图
    # Reshape back to image
    # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
    recons_img = recons_patches.view(
        b, h // self.patch_h, w // self.patch_w,
        self.patch_h, self.patch_w, c
    ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
    # mask 效果图
    patches[batch_ind, mask_ind] = mask_patches
    patches_to_img = patches.view(
        b, h // self.patch_h, w // self.patch_w,
        self.patch_h, self.patch_w, c
    ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    return recons_img, patches_to_img
# 2. 加载视频
video = cv2.VideoCapture(video_path)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. 循环读取视频的每一帧，并对其进行处理
for frame_index in range(frame_count):
    ret, frame = video.read()
    if not ret:
        break

    # 4. 调整帧的尺寸为224x224
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

    # 5. 在此处添加您的模型预测代码，将预测的图像和掩码保存为NumPy数组
    predicted_image = predict_image(resized_frame)  # 使用您的模型进行预测
    mask = predict_mask(resized_frame)  # 使用您的模型进行掩码预测

    # 6. 将图像和掩码调整回原始尺寸
    predicted_image = cv2.resize(predicted_image, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    # 7. 保存图像和掩码为PNG文件
    cv2.imwrite(output_image_dir + f"frame_{frame_index}.png", predicted_image)
    cv2.imwrite(output_mask_dir + f"frame_{frame_index}.png", mask)

# 8. 释放资源
video.release()