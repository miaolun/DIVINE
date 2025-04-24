import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 计算两个向量的点积
def dot_product(v1, v2):
    return np.dot(v1, v2)


# 计算向量的模
def magnitude(v):
    return np.linalg.norm(v)


# 检查视角是否对准x轴
def check_visibility(camera_position, camera_direction, line_point, line_direction):
    # 计算摄像机方向与直线方向的点积
    cos_angle = dot_product(camera_direction, line_direction) / (
                magnitude(camera_direction) * magnitude(line_direction))

    # 视角仅在x轴方向上有效
    # 如果摄像机视角接近x轴方向 (cos_angle接近1)，则显示直线
    if np.abs(cos_angle) > 0.1:  # 设置阈值
        return True
    return False


# 绘制直线
def plot_line(ax, line_point, line_direction, length=10):
    # 绘制一条沿着给定方向的直线
    t = np.linspace(-length, length, 100)
    line_points = line_point + t[:, np.newaxis] * line_direction
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], label="Line")


# 设置3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义直线的起点和方向
line_point = np.array([0, 0, 0])  # 直线起点
line_direction = np.array([1, 0, 0])  # 直线方向（沿x轴）

# 定义摄像机的位置和方向
camera_position = np.array([0, 0, 5])  # 摄像机位置
camera_direction = np.array([0, 0, -1])  # 摄像机方向（指向z轴负方向）

# 检查视角条件并渲染直线
if check_visibility(camera_position, camera_direction, line_point, line_direction):
    plot_line(ax, line_point, line_direction)
else:
    print("The line is not visible in the current view.")

# 设置图形的范围
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# 显示图形
plt.show()
