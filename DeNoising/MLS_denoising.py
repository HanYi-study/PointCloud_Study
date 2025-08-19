# MLS（Moving Least Squares）表面重建去噪（伪代码，需依赖open3d）
'''
MLS（Moving Least Squares，移动最小二乘）表面重建去噪是一种点云平滑和重建方法。
它通过在每个点的邻域内拟合一个局部平滑曲面，然后将该点投影到这个曲面上，从而消除噪声、平滑点云表面，并能重建缺失或不规则的表面结构。

输入文件内容：点云数据，通常为N行、3列（x, y, z），也可包含颜色（r, g, b）等属性。
输入文件格式：常见为.ply、.txt、.npy等，内容为点的坐标及可选属性。
需要几列数据：至少3列（x, y, z），颜色等属性可选。
输出文件内容：与输入格式一致，内容为经过MLS平滑和重建后的点云数据。
输出文件格式：与输入相同（.ply、.txt、.npy等）。

如何对输入的数据进行操作：
1.读取点云数据（如N×3的数组）。
2.为每个点查找其邻域点（通常用半径搜索）。
3.在邻域内用最小二乘法拟合一个局部平滑曲面。
4.将该点投影到拟合的曲面上，得到新的点位置。
5.对所有点重复上述操作，得到平滑后的点云。
6.保存处理后的点云数据，格式与输入一致。

操作完的数据变成了什么样：
点云表面更加光滑、连续，噪声和离群点被有效去除。
局部表面结构得到修复，点的位置可能发生较大变化。
点的数量和属性结构通常保持不变，但点的位置更贴合于局部平滑曲面。

伪代码：
在Python中，常用的Open3D库只提供了接口（如create_from_point_cloud_mls），而没有详细的算法步骤代码。
伪代码只是给出调用流程和主要步骤，具体实现细节依赖于第三方库的底层C++实现，普通Python代码难以手写高效的MLS算法。

'''

import argparse
import open3d as o3d

def mls_denoising(input_path, output_path, search_radius=0.05):  # MLS表面重建去噪
    pcd = o3d.io.read_point_cloud(input_path)  # 读取点云文件
    pcd = pcd.voxel_down_sample(voxel_size=search_radius/2)  # 体素下采样
    # pcd = pcd.remove_non_finite_points()  # 移除无穷大点 
    # voxel_size=search_radius/2 是体素边长，决定下采样后点的稀疏程度，值越大点越稀疏。
    # 取该值是为了让下采样后的点云密度与后续MLS算法的邻域搜索半径search_radius相适应
    # 将点云划分成一系列立方体网格（体素），每个体素内只保留一个代表点，从而减少点的数量，降低点云密度
    # 体素太大点云会过于稀疏，影响后面表面拟合的精度；体素太小点云太密集，计算量大，效率低
    # 取search_radius/2是一个经验值，能兼顾效率和效果，保证每个MLS邻域内有足够的点参与拟合，但不会太密。
    pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]  # 统计离群点去除
    # nb_neighbors=20：对于每个点，统计其最近的20个邻居点。20个邻居是点云处理中常用的经验值，能较好地反映局部结构。
    # std_ratio=2.0：标准差倍数阈值。若某点到其邻居的平均距离大于全局平均距离加2倍标准差，则认为该点为离群点并移除。2.0倍标准差是常见的异常值判定阈值，既能有效去除孤立点，又不会误删正常点。
    pcd = pcd.uniform_down_sample(every_k_points=1)  # 均匀下采样
    # 对点云进行均匀下采样（Uniform Downsampling），即每隔一定数量的点保留一个点，从而进一步降低点云密度，使点分布更加均匀。
    # every_k_points=1：表示每隔1个点保留一个点，也就是不做实际下采样，所有点都保留。
    # 本句代码当前没有实际效果，仅为占位或方便后续调整采样率。
    pcd = pcd.estimate_normals()  # 估计法线
    # 为点云中的每个点估计法向量（normal vector）
    # estimate_normals()函数的功能是：通过分析每个点的邻域（通常是最近的若干个点），拟合一个局部平面，并将该平面的法向量作为该点的法线。
    pcd = o3d.geometry.PointCloud.create_from_point_cloud_mls(pcd, search_radius=search_radius)  # MLS表面重建
    # o3d.geometry.PointCloud.create_from_point_cloud_mls 是Open3D库提供的MLS算法接口。
    # pcd：输入的点云对象，作为待平滑和重建的原始点云。
    # search_radius=search_radius：MLS算法中用于查找每个点邻域的半径。每个点会在该半径范围内找到邻居点，然后用最小二乘法拟合局部平滑曲面，并将该点投影到该曲面上，实现去噪和平滑。
    # 参数search_radius控制拟合局部曲面时的邻域大小。
    o3d.io.write_point_cloud(output_path, pcd)  # 写入点云文件

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云文件（txt/npy/ply）")
    parser.add_argument("--output", required=True, help="输出点云文件（同输入格式）")
    parser.add_argument("--search_radius", type=float, default=0.05)
    args = parser.parse_args()
    mls_denoising(args.input, args.output, args.search_radius)

'''
拟合曲面是什么:
在点云的每个点附近，找到一堆邻居点，这些点大致分布在一个小区域内。
用数学方法（最小二乘法）在这些邻居点中“拟合”出一个光滑的小曲面（比如一个小平面或二次曲面），让这个曲面尽量贴合这些点。
这个曲面就是“拟合曲面”，它代表了该点附近的理想、平滑的表面形状。

为什么要用拟合曲面:
原始点云有噪声，点的位置可能偏离真实表面。
拟合曲面能反映出局部的真实表面形状。
把点“投影”到拟合曲面上，相当于把点拉回到更合理的位置，从而消除噪声，让点云表面更光滑。

为什么还要用拟合曲面的法向量:
曲面的法向量就是该点表面“朝向”的方向。
法向量对后续的表面重建、渲染、特征提取等操作非常重要。
有了法向量，可以更好地描述点云的几何结构，比如判断表面是凸还是凹、光照如何反射等。

得到法线后能做什么:
可以用来重建网格表面（如三角网格），让点云变成连续的3D模型。
可以用于点云分割、特征提取、物体识别等任务。
可以提升渲染效果，让3D显示更真实。

这些操作和去噪的关系:
噪声点通常偏离真实表面，拟合曲面能“忽略”这些异常点，恢复真实表面形状。
投影到拟合曲面上，相当于把点“拉回”到合理位置，去除了噪声。
法向量的估计也更准确，进一步提升点云质量。
拟合曲面和法向量的计算，是为了让点云表面更光滑、结构更合理，从而实现去噪和平滑。

'''