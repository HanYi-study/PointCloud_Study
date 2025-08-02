'''#----------------------------------------------------------------------------------版本1.0
import matplotlib.pyplot as plt
from numpy import dtype
from plyfile import *
import numpy as np
import open3d as o3d
import sys
import importlib
importlib.reload(sys)

from mpl_toolkits.mplot3d import Axes3D

def makePlyFile_rgb_none(xyzs1, labels1, xyzs2, labels2, fileName):
    '''Make a ply file for open3d.visualization.draw_geometries
    :param xyzs:    numpy array of point clouds 3D coordinate, shape (numpoints, 3).
    :param labels:  numpy array of point label, shape (numpoints, ).
    '''
    numpoints_ground = len(xyzs1)
    numpoints = len(xyzs2)
    numpoints_sum = numpoints_ground + numpoints
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(numpoints_sum))
        f.write('property float64 x\n')
        f.write('property float64 y\n')
        f.write('property float64 z\n')
        # f.write('property int16 red\n')
        # f.write('property int16 green\n')
        # f.write('property int16 blue\n')
        f.write('property int8 label\n')
        f.write('end_header\n')
        for i in range(numpoints_ground):
            x_g, y_g, z_g = xyzs1[i]
            # r, g, b = rgbs[i]
            label_g = labels1[i]
            # f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, int(r), int(g), int(b), int(label)))
            f.write('{} {} {} {}\n'.format(x_g, y_g, z_g, int(label_g)))

        for i in range(numpoints):
            x, y, z = xyzs2[i]
            # r, g, b = rgbs[i]
            label = labels2[i]
            # f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, int(r), int(g), int(b), int(label)))
            f.write('{} {} {} {}\n'.format(x, y, z, int(label)))



def read_ply(ground_point_ply, net_predict_ply):
    # random_number = rs_after
    # 1、读取PLY数据
    Ground_Plydata = PlyData.read(ground_point_ply)
    Ground_Ply_Geometry = o3d.geometry.PointCloud()
    # 2、将原始PLY数据转换成numpy数组PointscloudArray
    PointscloudArray = np.array(Ground_Plydata)
    Ground_Plydata_Header = Ground_Plydata.header
    # 3、在numpy数组中提取坐标点的信息（x,y,z,r,g,b,label）并组成新的数组BeforeSampling_Points
    Ground_Points = []
    for point_temp in Ground_Plydata.elements[0].data:
        Ground_Points.append(
            [point_temp[0], point_temp[1], point_temp[2], point_temp[3]])
    Ground_Points = np.array(Ground_Points)
    # 6、将数组AfterSampling_Points进行拆分，以便将数组写入到输出文件中
    xyzs_g = Ground_Points[:, 0:3]  # xyz坐标数据
    # rgbs_g = BeforeSampling_Points[:, 3:6]  # rgb颜色数据
    labels_g = Ground_Points[:, 3]  # 标签数据

    # 1、读取PLY数据
    Ori_Plydata = PlyData.read(net_predict_ply)
    Ori_Ply_Geometry = o3d.geometry.PointCloud()
    # 2、将原始PLY数据转换成numpy数组PointscloudArray
    PointscloudArray = np.array(Ori_Plydata)
    Ori_Plydata_Header = Ori_Plydata.header
    # 3、在numpy数组中提取坐标点的信息（x,y,z,r,g,b,label）并组成新的数组BeforeSampling_Points
    BeforeSampling_Points = []
    for point_temp in Ori_Plydata.elements[0].data:
        BeforeSampling_Points.append(
            [point_temp[0], point_temp[1], point_temp[2], point_temp[3]])
    BeforeSampling_Points = np.array(BeforeSampling_Points)
    xyzs = BeforeSampling_Points[:, 0:3]  # xyz坐标数据
    # rgbs = BeforeSampling_Points[:, 3:6]  # rgb颜色数据
    labels = BeforeSampling_Points[:, 3]  # 标签数据

    # 7、将拆分的数组写入到输出文件
    makePlyFile_rgb_none(xyzs_g, labels_g, xyzs, labels, 'combine' + str(net_predict_ply))

if __name__ == "__main__":
    # 1.载入地面点数据
    # file_ply = ['131-132', '132-133', '133-134', '134-135', '135-136', '136-137', '137-138', '138-137', '139-140',
    #             '140-141', '141-142']
    # for i in file_ply:
    #     read_ply(i+'_g.ply', i+'.ply')
    # read_ply('ori16384.ply', 'ori1024.ply')
    read_ply('131132ex.ply', '132133ex.ply')
'''

import os
from plyfile import PlyData
import numpy as np

def extract_xyz_label(points):
    """
    自动判断点结构，提取xyz和label
    支持[x, y, z, label]或[x, y, z, r, g, b, label]
    """
    arr = np.array(points)
    if arr.shape[1] == 4:
        xyzs = arr[:, 0:3]
        labels = arr[:, 3]
    elif arr.shape[1] == 7:
        xyzs = arr[:, 0:3]
        labels = arr[:, 6]
    else:
        raise ValueError("点云数据格式不支持，仅支持4列或7列结构")
    return xyzs, labels

def makePlyFile_xyz_label(xyzs1, labels1, xyzs2, labels2, fileName):
    numpoints_ground = len(xyzs1)
    numpoints = len(xyzs2)
    numpoints_sum = numpoints_ground + numpoints
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(numpoints_sum))
        f.write('property float64 x\n')
        f.write('property float64 y\n')
        f.write('property float64 z\n')
        f.write('property int8 label\n')
        f.write('end_header\n')
        for i in range(numpoints_ground):
            x_g, y_g, z_g = xyzs1[i]
            label_g = labels1[i]
            f.write('{} {} {} {}\n'.format(x_g, y_g, z_g, int(label_g)))
        for i in range(numpoints):
            x, y, z = xyzs2[i]
            label = labels2[i]
            f.write('{} {} {} {}\n'.format(x, y, z, int(label)))

def read_ply(ground_point_ply, net_predict_ply, out_path, suffix='_combined'):
    # 读取地面点PLY
    Ground_Plydata = PlyData.read(ground_point_ply)
    ground_points = []
    for point_temp in Ground_Plydata.elements[0].data:
        ground_points.append(list(point_temp))
    xyzs_g, labels_g = extract_xyz_label(ground_points)

    # 读取预测点PLY
    Ori_Plydata = PlyData.read(net_predict_ply)
    predict_points = []
    for point_temp in Ori_Plydata.elements[0].data:
        predict_points.append(list(point_temp))
    xyzs, labels = extract_xyz_label(predict_points)

    base_name = os.path.splitext(os.path.basename(net_predict_ply))[0]
    out_file = os.path.join(out_path, base_name + suffix + '.ply')
    makePlyFile_xyz_label(xyzs_g, labels_g, xyzs, labels, out_file)
    print(f' 合并后的文件已保存到: {out_file}')

def main():
    print("=== 点云PLY合并工具（自动识别RGB，合并时不需要RGB / 可以根据实际需求再进行调整） ===")
    print("合并的是输入ply文件中的x y z label")
    print("支持输入的PLY文件每个点的数据结构：x, y, z, label 或 x, y, z, r, g, b, label\n")
    ground_ply = input("请输入地面点PLY文件路径：\n> ").strip()
    predict_ply = input("请输入预测点PLY文件路径：\n> ").strip()
    out_path = input("请输入保存合并后文件的文件夹路径（默认当前文件夹）：\n> ").strip()
    if out_path == '':
        out_path = '.'
    suffix = input("请输入输出文件名后缀（默认 _combined）：\n> ").strip()
    if suffix == '':
        suffix = '_combined'
    read_ply(ground_ply, predict_ply, out_path, suffix)

if __name__ == "__main__":
    main()

'''
示例输出：
=== 点云PLY合并工具（自动识别RGB，合并时不需要RGB / 可以根据实际需求再进行调整） ===
支持输入的PLY文件每个点的数据结构：x, y, z, label 或 x, y, z, r, g, b, label

请输入地面点PLY文件路径：
> 131132ex.ply
请输入预测点PLY文件路径：
> 132133ex.ply
请输入保存合并后文件的文件夹路径（默认当前文件夹）：
> D:\PointCloud\PointClout_hy\combine_ply_6\output
请输入输出文件名后缀（默认 _combined）：
> _merged

✅ 合并后的文件已保存到: D:\PointCloud\PointClout_hy\combine_ply_6\output\132133ex_merged.ply

'''