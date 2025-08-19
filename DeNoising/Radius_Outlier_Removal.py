# 半径离群点去除
'''
 半径离群点去噪（Radius Outlier Removal）是一种点云去噪方法。
 它通过检查每个点在指定半径范围内的邻居点数量，如果某点的邻居数少于设定阈值，则认为该点是离群点（噪声点），将其移除。

 输入文件：
 内容：点云数据，通常为每个点的三维坐标（x, y, z），有时还包含颜色（r, g, b）等属性。
 格式：常见为.ply、.txt、.npy、.xyz等，内容为点的坐标及可选属性。
 至少需要3列（x, y, z），颜色等属性可选。
 输出文件：
 内容：去除了离群点后的点云数据，格式与输入一致。
 格式：同输入（如.ply、.txt、.npy等），包含点的坐标及可选属性。

 如何操作？
 1.读取点云数据。
 2.对每个点，统计其在指定半径内的邻居点数量。
 3.如果邻居数小于阈值，则判定为离群点并移除。
 4.保留剩余点，保存为输出文件。

 操作完的数据变成了什么样：
 点云中的孤立点、噪声点被有效去除。
 点云整体更干净，结构更连贯，便于后续处理和分析。


'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

def radius_outlier_removal(points, radius=0.05, min_neighbors=5):  # 半径离群点去除
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(points)  # 使用自动算法找到最近邻
    neighbors = nbrs.radius_neighbors(points, return_distance=False)  # 计算每个点的邻居
    # return_distance=False 表示只返回邻居点的索引，不返回距离
    # neighbors存储了每个点在指定半径内的邻居点索引列表
    mask = np.array([len(n) > min_neighbors for n in neighbors])  # 创建掩码，保留邻居数量大于 min_neighbors 的点
    return points[mask]  # 返回去除离群点后的点云

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云文件（txt/npy/ply）")
    parser.add_argument("--output", required=True, help="输出点云文件（同输入格式）")
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--min_neighbors", type=int, default=5)
    args = parser.parse_args()

    from data_prepare.help_ply import read_point_cloud, write_point_cloud

    points = read_point_cloud(args.input)
    filtered = radius_outlier_removal(points, args.radius, args.min_neighbors)
    write_point_cloud(args.output, filtered)