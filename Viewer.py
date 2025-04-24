# ply viewer
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


def load_ply(filepath):
    """
    从 PLY 文件中加载点云数据。

    参数:
        filepath (str): PLY 文件的路径。

    返回:
        positions (np.array): 点的位置数组，形状为 (N, 3)。
        colors (np.array): 点的颜色数组，形状为 (N, 3)。
        normals (np.array): 点的法向量数组，形状为 (N, 3)。
        visible (np.array): 点的可见性数组，形状为 (N,)，值为 0 或 1。
    """
    positions = []
    colors = []
    normals = []
    visible = []

    with open(filepath, 'r') as file:
        lines = file.readlines()

        # 找到头部结束的位置
        header_end = 0
        for i, line in enumerate(lines):
            if "end_header" in line:
                header_end = i + 1
                break

        # 解析点数据
        for line in lines[header_end:]:
            parts = line.strip().split()
            if len(parts) >= 10:  # 确保有位置、颜色、法向量和 visible 属性
                x, y, z = map(float, parts[:3])
                r, g, b = map(int, parts[3:6])
                nx, ny, nz = map(float, parts[6:9])
                vis = int(parts[9])  # 读取 visible 属性
                positions.append([x, y, z])
                colors.append([r / 255, g / 255, b / 255])  # 归一化到 [0, 1]
                normals.append([nx, ny, nz])
                visible.append(vis)

    return np.array(positions), np.array(colors), np.array(normals), np.array(visible)


def is_view_tangent(view_direction, normal):
    """
    判断法向量是否与视线方向垂直。

    参数:
        view_direction (np.array): 视线方向向量。
        normal (np.array): 点的法向量。

    返回:
        bool: 如果法向量与视线方向垂直，返回 True，否则返回 False。
    """
    dot_product = np.dot(view_direction, normal)
    return abs(dot_product) < 1e-2  # 切线阈值


def draw_point_cloud_dynamic(points, colors):
    """
    使用 OpenGL 绘制点云。

    参数:
        points (np.array): 点的位置数组，形状为 (N, 3)。
        colors (np.array): 点的颜色数组，形状为 (N, 3)。
    """
    glBegin(GL_POINTS)
    for pos, col in zip(points, colors):
        glColor3f(*col)
        glVertex3f(*pos)
    glEnd()


def calculate_view_direction():
    """
    从模型视图矩阵中提取相机的视线方向。

    返回:
        np.array: 视线方向向量。
    """
    # 获取当前的模型视图矩阵
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)

    # 提取相机的视线方向（模型视图矩阵的第三列）
    view_direction = np.array([-modelview_matrix[0][2], -modelview_matrix[1][2], -modelview_matrix[2][2]])
    view_direction = view_direction / np.linalg.norm(view_direction)  # 归一化
    return view_direction


def main(background_color=(1.0, 1.0, 1.0, 1.0)):
    """
    主函数，初始化 Pygame 和 OpenGL，加载点云并显示。

    参数:
        background_color (tuple): 背景颜色 (R, G, B, A)，默认为白色。
    """
    # 初始化 Pygame 和 OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    # 加载 PLY 文件
    # filepath = r"D:\workspace\ND-NeRF\exp\DTU\random_point_cloud.ply"
    filepath = r'D:\workspace\ND-NeRF\exp\DTU\scan105\emap\results\RIND_points.ply'  # 替换为你的 PLY 文件路径
    positions, colors, normals, visible = load_ply(filepath)

    # 启用 OpenGL 功能
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(2)

    # 设置背景颜色
    glClearColor(*background_color)

    clock = pygame.time.Clock()

    # 相机旋转变量
    rotation_x, rotation_y = 0, 0
    last_mouse_pos = None

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    last_mouse_pos = event.pos
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:  # 左键
                    last_mouse_pos = None
            elif event.type == MOUSEMOTION:
                if last_mouse_pos:
                    dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                    rotation_x += dy * 0.5
                    rotation_y += dx * 0.5
                    last_mouse_pos = event.pos

        # 清屏
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()

        # 应用相机旋转
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        # 计算视线方向（从模型视图矩阵中提取）
        view_direction = calculate_view_direction()

        # 根据 visible 属性和法向量确定可见点
        visible_points = []
        visible_colors = []
        for pos, norm, col, vis in zip(positions, normals, colors, visible):
            if vis == 1:  # visible 为 1 的点始终显示
                visible_points.append(pos)
                visible_colors.append(col)
            elif vis == 0 and is_view_tangent(view_direction, norm):  # visible 为 0 的点需要判断法向量
                visible_points.append(pos)
                visible_colors.append(col)

        # 绘制可见点
        draw_point_cloud_dynamic(visible_points, visible_colors)

        glPopMatrix()
        pygame.display.flip()
        clock.tick(60)

        # 打印当前旋转角度
        print(f"Rotation X: {rotation_x:.2f}, Rotation Y: {rotation_y:.2f}", end="\r")


if __name__ == "__main__":
    main()