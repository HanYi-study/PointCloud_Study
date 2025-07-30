import open3d as o3d  # 读取 .ply 点云文件。
import numpy as np

'''
该脚本实现了将 .ply 格式的点云数据转换为 .pcd 格式（Point Cloud Data），输出的数据包含字段：x, y, z, intensity。

其主要流程是：
    使用 Open3D 加载 .ply 点云文件。
    提取点坐标数据。
    生成 .pcd 文件的文本格式（ASCII格式），并写入坐标数据和固定 intensity 字段。

输入文件： .ply 文件（只读点的 x, y, z）
输出文件： .pcd 文件，添加了 intensity=0 字段

使用库： Open3D、NumPy
适用场景： 需要将点云从 .ply 转换为 .pcd 以适配不同工具（如 PCL）时


'''


path=r"/home/yunl/Data/paper_data/fps/ori.ply"  #输入文件路径，.ply 格式。
ply = o3d.io.read_triangle_mesh(path)  #read_triangle_mesh()：从 .ply 文件读取一个三角网格对象（但点云也可以这样读取点坐标）。

points = np.array(ply.vertices) #转为 (N, 3) 矩阵, 提取的是顶点坐标，即每个点的 x, y, z 值。
'''
for element in points:
    for data in element.data:
        print(data)
''' #调试用，输出每个点的坐标
save_pcd_path=r"/home/yunl/Data/paper_data/fps/ori2.pcd"  # .pcd 文件保存路径。--------------可指定输出路径

HEADER = '''\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4 
TYPE F F F F 
COUNT 1 1 1 1 
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
'''
#构建 .pcd 文件头部
#FIELDS x y z intensity	表明每个点包含 4 个字段
#TYPE F F F F	数据类型为 float（前三个）和 float（第四个 intensity）
#SIZE 4 4 4 4	每个字段 4 字节
#COUNT 1 1 1 1	每个点每个字段有一个值
#WIDTH, POINTS	点的数量
#DATA ascii	数据存储为 ASCII


'''

def write_pcd(points, save_pcd_path):
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z, i, is_g = points[i]
        lines.append('{:.6f} {:.6f} {:.6f} {}'.format( \
            x, y, z, i))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))
'''


def write_pcd(points, save_pcd_path):
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(len(points), len(points)) + '\n')
        np.savetxt(f, points, delimiter = ' ', fmt = '%f %f %f %d')
