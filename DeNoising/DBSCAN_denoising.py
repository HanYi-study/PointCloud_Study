
# DNSCAN聚类去噪（保留最大连通分量）
'''
DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法。
利用DBSCAN算法对点云进行聚类，将密度较低的点（即噪声点）识别出来并去除，只保留最大连通分量（即最大簇），从而达到去除孤立点、离群点的目的。

输入格式：支持 .txt、.npy、.ply 等常见点云格式（由 read_point_cloud 决定）。
输入内容：通常为点云的坐标（x, y, z），也可以包含颜色（r, g, b）或标签等信息。
需要几列数据：最少需要3列（x, y, z），如果有颜色或标签，额外的列会被保留，但聚类通常只用前三列。
输出格式：与输入格式一致（.txt、.npy、.ply）。
输出内容：去除了噪声点后的点云数据，保留最大连通分量。

如何对输入的数据进行操作:
1.读取点云数据（如N×3或N×6的数组）
2.用DBSCAN算法聚类，将点分为若干簇和噪声点（标签为-1）。
3.统计每个簇的点数，找到最大簇（最大连通分量）。
4.只保留最大簇的点，其余点（包括噪声点和小簇）全部去除。
5.保存处理后的点云数据。

操作完的数据变成了什么样:
1.点数减少：孤立点、离群点、稀疏小簇被去除，只剩下主簇（最大连通分量）。
2.点云密度提高：去除了噪声点后，点云的整体密度得到了提升。
3.结构更完整：主结构被保留，噪声点被清理。
4.处理速度更快：去除了冗余点后，后续处理（如重建、分割等）的速度得到了提升。

如何判断哪些点构成一个连通分量的？
1.密度可达性：
  对于每个点，DBSCAN 会查找其 eps 半径内的所有邻居点。
  如果某个点的邻居数（包括自身）≥ min_samples，则这个点被认为是“核心点”。
2.边界点和噪声点：
  如果某个点的邻居数（包括自身）< min_samples，则这个点被认为是“边界点”或“噪声点”。
  边界点是指在核心点的邻域内，但自身不是核心点的点。
  噪声点是指不在任何核心点的邻域内的点。
DBSCAN 通过“密度可达性”原则，把空间上彼此靠近、密度足够的点归为同一个簇，这个簇就是一个连通分量。
这些点之间没有实际连线，但它们在空间上通过核心点间接“连通”。
最终，属于同一个簇的点标签相同，构成一个连通分量。

'''

import numpy as np
from sklearn.cluster import DBSCAN
import argparse

def dbscan_denoising(points, eps=0.05, min_samples=10):  
    # points是一个N×3或N×6的数组
    # eps是DBSCAN算法中的邻域半径
    # min_samples是每个簇的最小样本数
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)  # 使用DBSCAN进行聚类
    # labels是一个长度为N的数组，表示每个点所属的簇标签，噪声点的标签为-1
    # fit_predict()的作用是先拟合模型，然后返回每个样本的标签，此函数是sklearn.cluster.DBSCAN 类自带的方法，它由scikit-learn库自动提供。
    unique, counts = np.unique(labels, return_counts=True)  # 计算每个标签的唯一值及其计数
    # unique是每个簇的标签（不包括噪声点 -1）
    # counts是每个簇的点数
    # unique实现了对labels数组中所有唯一值的提取，并返回这些唯一值的数组。
    # return_counts=True参数使得函数同时返回每个唯一值在原数组中出现的次数。
    main_label = unique[np.argmax(counts[unique != -1])]  # 找到出现次数最多的标签（排除噪声点 -1）
    # main_label保存的是最大连通分量的标签
    # argmax的作用是返回counts数组中最大值的索引位置
    # counts[unique != -1]的作用是过滤掉噪声点 -1 的计数
    # 为什么要去掉噪声点的计数：如果噪声点数量较多，最大簇可能会被错误地选为噪声点集合（-1），而不是实际的主结构。这样会导致最终保留下来的不是主结构点，而是噪声点，违背了去噪的目的。
    # 如果有两个或以上的簇点数完全相同且都达到最大值，那么它们都可以被视为“最大连通分量”。不过，常规写法（如你的代码）只会保留其中第一个被找到的最大簇。
    mask = labels == main_label  # 创建掩码，保留最大连通分量的点
    # 得到的 mask 是一个布尔型数组，长度与点云点数相同，True 表示该点属于最大连通分量，False 表示不属于。
    # 假设 labels = [1, 2, 1, -1, 1, 2]，main_label = 1，则 mask = [True, False, True, False, True, False]，只保留标签为 1 的点。
    return points[mask]  # 返回去除噪声后的点云

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云文件（txt/npy/ply）")
    parser.add_argument("--output", required=True, help="输出点云文件（同输入格式）")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--min_samples", type=int, default=10)
    args = parser.parse_args()

    from data_prepare.help_ply import read_point_cloud, write_point_cloud  # 读写点云文件的帮助函数

    points = read_point_cloud(args.input)  # 读取输入点云文件
    filtered = dbscan_denoising(points, args.eps, args.min_samples)  # 应用DBSCAN去噪
    write_point_cloud(args.output, filtered)  # 写入输出点云文件