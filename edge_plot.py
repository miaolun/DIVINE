import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
#
# # 加载 JSON 文件
with open(r"D:\workspace\EMAP\exp\ABC\00000843\emap\results\parametric_edges.json", "r") as file:
    data = json.load(file)
#
# # 提取曲线控制点和直线端点
# curves_ctl_pts = data["curves_ctl_pts"]
# lines_end_pts = data["lines_end_pts"]
#
# # 创建三维图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制曲线
# for curve in curves_ctl_pts:
#     curve = list(zip(*curve))  # 转置为每列分别为 x, y, z
#     ax.plot(curve[0], curve[1], curve[2], label="Curve", color="blue")
#
# # 绘制直线
# for line in lines_end_pts:
#     x = [line[0], line[3]]
#     y = [line[1], line[4]]
#     z = [line[2], line[5]]
#     ax.plot(x, y, z, label="Line", color="red")
#
# # 设置图形参数
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("3D Edges Visualization")
# plt.legend(["Curve", "Line"])
# plt.show()




# 加载 JSON 文件
# with open("parametric_edges.json", "r") as file:
#     data = json.load(file)

# 提取曲线控制点和直线端点
curves_ctl_pts = data["curves_ctl_pts"]
lines_end_pts = data["lines_end_pts"]

# 创建一个绘图对象
fig = go.Figure()

# 绘制曲线
for curve in curves_ctl_pts:
    x, y, z = zip(*curve)  # 提取 x, y, z 坐标
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="blue", width=3), name="Curve"))

# 绘制直线
for line in lines_end_pts:
    x = [line[0], line[3]]
    y = [line[1], line[4]]
    z = [line[2], line[5]]
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="red", width=3), name="Line"))

# 设置布局，背景为纯白
fig.update_layout(
    scene=dict(
        xaxis=dict(backgroundcolor="white", showgrid=False, zeroline=False),
        yaxis=dict(backgroundcolor="white", showgrid=False, zeroline=False),
        zaxis=dict(backgroundcolor="white", showgrid=False, zeroline=False),
    ),
    scene_aspectmode="data",  # 保持真实比例
    margin=dict(l=0, r=0, t=0, b=0),  # 去掉多余的边距
    showlegend=True,
)

# 显示图形
fig.show()
