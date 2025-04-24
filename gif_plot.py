import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio

# 加载 JSON 文件
with open(r"D:\workspace\EMAP\exp\ABC\00000325\emap\results\parametric_edges.json", "r") as file:
    data = json.load(file)

# 提取曲线控制点和直线端点
curves_ctl_pts = data["curves_ctl_pts"]
lines_end_pts = data["lines_end_pts"]

# 设置 GIF 保存路径
output_gif = "3d_edges_animation.gif"

# 创建帧列表
frames = []

# 生成旋转角度的帧
num_frames = 60  # 总帧数
for angle in np.linspace(0, 360, num_frames):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")  # 背景设置为白色

    # 绘制曲线
    for curve in curves_ctl_pts:
        x, y, z = zip(*curve)  # 提取 x, y, z 坐标
        ax.plot(x, y, z, color="blue", linewidth=2)

    # 绘制直线
    for line in lines_end_pts:
        x = [line[0], line[3]]
        y = [line[1], line[4]]
        z = [line[2], line[5]]
        ax.plot(x, y, z, color="red", linewidth=2)

    # 设置视角
    ax.view_init(elev=30, azim=angle)

    # 设置坐标轴样式
    ax.set_box_aspect([1, 1, 1])  # 等比例缩放
    ax.grid(False)  # 不显示网格
    ax.set_xticks([])  # 去掉坐标轴刻度
    ax.set_yticks([])
    ax.set_zticks([])

    # 保存当前帧
    frame_path = f"frame_{int(angle)}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches="tight")
    frames.append(imageio.imread(frame_path))
    plt.close()

# 生成 GIF
imageio.mimsave(output_gif, frames, duration=0.05)  # 每帧持续时间 0.05 秒

# 清理中间帧文件
import os
for frame in frames:
    os.remove(frame)

print(f"GIF 动图已保存为 {output_gif}")
