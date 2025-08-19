# 统计离群点去除
'''
 统计离群点去噪（Statistical Outlier Removal）是一种点云去噪方法。
 它通过统计每个点与其邻居点的距离，判断哪些点与周围点距离异常大，将这些点视为离群点（噪声点）并移除。

 输入文件：
 内容：点云数据，通常为每个点的三维坐标（x, y, z），有时还包含颜色（r, g, b）等属性。
 格式：常见为.ply、.txt、.npy、.xyz等，内容为点的坐标及可选属性。
 至少需要3列（x, y, z），颜色等属性可选。
 输出文件：
 内容：去除了离群点后的点云数据，格式与输入一致。
 格式：同输入（如.ply、.txt、.npy等），包含点的坐标及可选属性。

 如何对输入的数据进行操作
 1.读取点云数据。
 2.对每个点，计算其与最近若干个邻居点的平均距离。
 3.统计所有点的平均距离分布，计算全局均值和标准差。
 4.如果某点的平均距离大于全局均值加上若干倍标准差，则判定为离群点并移除。
 5.保留剩余点，保存为输出文件。

 操作完的数据变成了什么样
 点云中的孤立点、异常点被有效去除。
 点云整体更干净，结构更连贯，便于后续处理和分析。


'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

def statistical_outlier_removal(points, k=16, std_ratio=2.0):  # 统计离群点去除
    # std_ratio是标准差倍数，用于判断离群点的阈值
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)   # 使用自动算法找到最近邻
    distances, _ = nbrs.kneighbors(points)  # 计算每个点到其 k 个最近邻的距离
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 计算每个点的平均距离（忽略自身）
    # distances[:, 1:]是每个点到其 k 个最近邻的距离，[:, 1:]表示忽略第一个距离（点自身到自己的距离为0）
    # np.mean(distances[:, 1:], axis=1)计算每个点的平均距离，axis=1表示按行计算平均值
    mean = np.mean(avg_distances)  # 计算所有点与其邻域点们之间距离的平均值
    std = np.std(avg_distances)  # 计算所有点与其邻域点们之间距离的标准差
    mask = avg_distances < mean + std_ratio * std  # 创建掩码，保留平均距离小于 mean + std_ratio * std 的点
    return points[mask]  # 返回去除离群点后的点云

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云文件（txt/npy/ply）")
    parser.add_argument("--output", required=True, help="输出点云文件（同输入格式）")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    args = parser.parse_args()

    # 这里假设你有 help_ply.py 或 txt_to_npy.py 等通用读写函数
    from data_prepare.help_ply import read_point_cloud, write_point_cloud

    points = read_point_cloud(args.input)
    filtered = statistical_outlier_removal(points, args.k, args.std_ratio)
    write_point_cloud(args.output, filtered)