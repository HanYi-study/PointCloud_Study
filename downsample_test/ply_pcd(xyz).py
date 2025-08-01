'''

from plyfile import PlyData,PlyElement  # 用于解析 .ply 文件,plyfile 是专门用于读取 .ply 文件的 Python 库；
import pcl  # 导入 PCL 点云库（但本脚本未使用）
import os  # 导入 PCL 点云库（但本脚本未使用）
import numpy as np 
import pandas as pd

file=r"/home/yunl/Data/paper_data/fps/ori.ply"  # file 是待转换的 .ply 文件路径；
PCD_FILE_PATH=r"/home/yunl/Data/paper_data/fps/ori2.pcd"  #PCD_FILE_PATH 是目标 .pcd 文件路径；
if os.path.exists(PCD_FILE_PATH):
    os.remove(PCD_FILE_PATH)  #如果 .pcd 文件已存在，就先删除它，避免内容叠加或格式错误。


# 写文件句柄
handle = open(PCD_FILE_PATH, 'a')
# pcd头部
handle.write(
    '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\ncolor r g b\nTYPE F F F\nCOUNT 1 1 1')
string = '\nWIDTH ' + str(65536)
handle.write(string)
handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
string = '\nPOINTS ' + str(65536)
handle.write(string)
handle.write('\nDATA ascii')
 #有问题


plydata=PlyData.read(file)  #读
elements=plydata.elements  #elements 是一个列表，通常第一个元素是 vertex，表示点云数据。

# 依次写入点
for i in range(65536):  # 这里我只用到了前三列，故只需展示0，1，2三列 读者可根据自身需要写入其余列
    string = '\n' + str(elements[i, 0]) + ' ' + str(elements[i, 1]) + ' ' + str(elements[i, 2])
    handle.write(string)
handle.close()




# plydata=PlyData.read(file)

print(plydata)
print("*************************************************")
#第一种读取方法
elements=plydata.elements
for element in elements:
    for data in element.data:
        print(data)
#遍历每个元素（如 vertex/face），逐行打印每个点的数据。

print("*************************************************")

#第二种读取方法
# vertex_data=elements[0].data
# face_data =elements[1].data
# print(vertex_data)
# print(face_data)

#若 .ply 中包含多个元素（如顶点和面），可分别提取处理；
#多用于三角网格的几何处理。

'''

from plyfile import PlyData  #引入 plyfile 模块中的 PlyData 类，用于读取 .ply 格式的点云数据。
#.ply 是 Polygon File Format（多边形文件格式）或 Stanford Triangle Format，常用于 3D 扫描点云。
import numpy as np  #引入 NumPy，用于将点云坐标组合为矩阵进行批量处理。
import struct  

#作用: 读取 .ply 点云文件中的坐标信息，转换为 .pcd 格式并保存
#输入格式: .ply 文件（文本或二进制），包含至少三个字段：x, y, z
#输出格式: .pcd 文件（ASCII 文本格式），标准 PCD v0.7 格式，仅包含 x y z 三列

def write_pcd(points, save_path):  # 定义一个函数 write_pcd()，将点云的 x/y/z 数据写入 .pcd 文件。
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
#构造 .pcd 文件的标准头部：
#FIELDS x y z：点的三个坐标分量。
#TYPE F F F：三者都是 float 类型。
#WIDTH 和 POINTS 都是点的数量 len(points)。
#DATA ascii：采用文本格式（而非二进制）。

    with open(save_path, 'w') as f:  #打开目标文件 save_path，写入头部和点数据。
        f.write(header) 
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")  #每个点一行，格式为 "x y z"，值之间用空格分隔。

ply_path = "/home/yunl/Data/paper_data/fps/ori.ply"  #读取.ply文件，定位到文件的路径
pcd_path = "/home/yunl/Data/paper_data/fps/ori2.pcd"  #写入.pcd，定位到文件的路径

plydata = PlyData.read(ply_path)  #读取 .ply 文件，得到 PlyData 对象，它包含点云所有的属性字段（如 x, y, z, r, g, b, label 等）。
vertex = plydata['vertex']  #提取 .ply 文件中的 vertex 数据块，也就是点坐标。
points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)  #将 x、y、z 三列拼接成一个 (N, 3) 的 NumPy 数组。
write_pcd(points, pcd_path)  #调用上面定义的函数，将 .ply 中提取的坐标写入 .pcd 文件。
