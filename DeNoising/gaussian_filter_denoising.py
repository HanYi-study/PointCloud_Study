# 高斯滤波去噪
'''
高斯滤波去噪是一种利用高斯核对点云数据进行平滑处理的方法。
它通过对点云的每一列（通常是 x、y、z 坐标）进行一维高斯滤波，减弱孤立点和局部噪声，使点云整体更加平滑。

输入文件内容：点云数据，通常为 N 行、3 列或更多（如 x, y, z, r, g, b, label）。
输入文件格式：常见为 .txt、.npy、.ply 等，内容为点的坐标及可选属性。
需要几列数据：至少 3 列（x, y, z），可包含更多属性（如颜色、标签等）。
输出文件内容：与输入格式一致，内容为经过高斯滤波处理后的点云数据。
输出文件格式：与输入相同（.txt、.npy、.ply 等）

如何对输入的数据进行操作：
1.读取点云数据（如 N×3 或 N×6 的数组）。
2.对每一列（如 x、y、z）分别应用一维高斯滤波，即用高斯核对每一列进行平滑处理。
3.得到平滑后的点云数据。
4.保存处理后的点云数据，格式与输入一致。

操作完的数据变成了什么样：
点云整体更加平滑，孤立点和局部噪声被减弱。
点的空间分布变得更连续，但可能会导致边缘细节变模糊。
数据格式和列数不变，只是点的坐标等数值发生了平滑变化。
经过高斯核的平滑处理，原点云的坐标点数值会发生变化，每个点的新坐标是它和周围邻域点的加权平均值（权重由高斯分布决定），所以点的位置会变得更加平滑、连续，数值与原始点云不同。

'''

import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse

def gaussian_filter_denoising(points, sigma=1.0):
    # points是一个N×3或N×6的数组
    # sigma是高斯滤波的标准差
    return gaussian_filter1d(points, sigma=sigma, axis=0)  # 高斯滤波去噪
    # gaussian_filter1d 是 SciPy 提供的高斯一维滤波函数，用于对数组的指定轴进行高斯平滑处理。
    '''
    高斯滤波的基本思想：
    高斯滤波是一种常用的信号平滑方法，用高斯分布（钟形曲线）作为权重，对每个数据点及其邻域进行加权平均，距离越近权重越大，距离越远权重越小。

    gaussian_filter1d的工作方式：
    这个函数会对 points 数组的每一列（如 x、y、z）分别进行一维高斯滤波。因为axis=0表示对列进行操作。
    对于每一列，函数会用一个长度为 2*truncate*sigma+1 的高斯核，在该列上滑动，对每个点及其邻域点做加权平均.
    （详解这一步：高斯滤波时，会先生成一个高斯分布形状的“权重窗口”（即高斯核），这个窗口的长度由参数 sigma（标准差）和 truncate（截断系数，默认值为4.0）共同决定，长度为 2*truncate*sigma+1。
    然后，这个高斯核会在每一列数据上从头到尾滑动。对于每一个点，都会用这个高斯核覆盖它和它周围的邻域点，把这些点的数值按照高斯核的权重做加权平均，作为该点的新值。
    这样，距离当前点越近的邻域点权重越大，越远的权重越小，实现平滑效果。）
    sigma 控制高斯核的宽度，值越大，平滑效果越强
    （因为 sigma 是高斯分布的标准差，它决定了高斯核的“宽度”——也就是参与加权平均的邻域范围有多大。
    sigma 越大，高斯核越宽，每个点的新值会参考更远的邻域点，平均的范围更大，数据变化会被“拉平”，所以平滑效果更强，噪声被抑制得更多，但细节也更容易被模糊。
    sigma 越小，高斯核越窄，只参考很近的邻域点，平滑效果较弱，细节保留更多。）

    有点抽象

    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云文件（txt/npy/ply）")
    parser.add_argument("--output", required=True, help="输出点云文件（同输入格式）")
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    from data_prepare.help_ply import read_point_cloud, write_point_cloud

    points = read_point_cloud(args.input)
    filtered = gaussian_filter_denoising(points, args.sigma)
    write_point_cloud(args.output, filtered)