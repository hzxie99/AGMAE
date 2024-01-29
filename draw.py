import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 输入数据
x = [25.0,50.0,75.0,90.0]
y1 = [71.88 , 71.32, 71.79,71.02]
y2 = [71.30, 70.97, 69.46, np.NAN]
# 设置颜色代码
color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)

# 绘图
sns.set_style("whitegrid") # 设置背景样式
sns.lineplot(x=x, y=y1, color=color1, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='AGMAE')
sns.lineplot(x=x, y=y2, color=color2, linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='MAR')

# 添加标题和标签

plt.xlabel("Mask Ratio(%)", fontsize=12)
plt.ylabel("Accuracy(%)", fontsize=12)

# 添加图例
plt.legend(loc='upper left', frameon=True, fontsize=10)

# 设置刻度字体和范围
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(0, 100)
plt.ylim(69.0, 72)

# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)

plt.savefig('lineplot.png', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()