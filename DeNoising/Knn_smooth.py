# KNN平滑去噪
'''
KNN平滑去噪是一种基于K近邻（K-Nearest Neighbors, KNN）思想的点云去噪方法。
它通过将每个点的位置替换为其K个最近邻点的平均值，从而减弱孤立点和局部噪声，使点云整体更加平滑。

输入文件内容：点云数据，通常为N行、3列或更多（如x, y, z, r, g, b, label）。
输入文件格式：常见为.txt、.npy、.ply等，内容为点的坐标及可选属性。
需要几列数据：至少3列（x, y, z），可包含更多属性（如颜色、标签等）。
输出文件内容：与输入格式一致，内容为经过KNN平滑处理后的点云数据。
输出文件格式：与输入相同（.txt、.npy、.ply等）

如何对输入的数据进行操作：
1.读取点云数据（如N×3或N×6的数组）。
2.对每个点，找到其K个最近邻点。
3.用这K个最近邻点的平均值替换当前点的位置。
4.对所有点重复上述操作，得到平滑后的点云。
5.保存处理后的点云数据，格式与输入一致。

操作完的数据变成了什么样：
点云整体更加平滑，孤立点和局部噪声被减弱。
点的空间分布变得更连续，但可能会导致边缘细节变模糊。
数据格式和列数不变，只是点的坐标等数值发生了平滑变化。
经过KNN的平滑处理，原点云的坐标点数值会发生变化，每个点的新坐标是它和周围邻域点的加权平均值（权重由KNN决定），所以点的位置会变得更加平滑、连续，数值与原始点云不同。

'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

def knn_smooth(points, k=16):  # KNN平滑去噪
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)  # 使用自动算法找到最近邻
    # nbrs是一个 KNN 模型，用于查找最近邻点
    # n_neighbors是要查找的邻居数量，这里是 k+1，因为第一个邻居是点本身
    # algorithm='auto'表示让算法自动选择最合适的方式来计算最近邻
    # fit是用来训练模型的方法，这里是用点云数据来训练 KNN 模型
    _, indices = nbrs.kneighbors(points)  # 计算每个点的 k 个最近邻
    # _是每个点到其 k 个最近邻的距离（这里不需要用到，所以用 _ 忽略）
    smoothed = np.array([points[neighbors[1:]].mean(axis=0) for neighbors in indices])  # 对每个点的 k 个最近邻进行平均
    return smoothed  # 返回平滑后的点云

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云文件（txt/npy/ply）")
    parser.add_argument("--output", required=True, help="输出点云文件（同输入格式）")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    from data_prepare.help_ply import read_point_cloud, write_point_cloud

    points = read_point_cloud(args.input)
    filtered = knn_smooth(points, args.k)
    write_point_cloud(args.output, filtered)