import numpy as np  #NumPy用于数组和数学计算
import pcl  #PCL（Point Cloud Library）用于处理点云数据，Python PCL（Point Cloud Library）接口
#print(pcl.__version__)
from plyfile import PlyData, PlyElement  #PlyData和PlyElement用于处理PLY文件格式

"""
KNN_Noise_points.py 是一个用于点云噪声点提取的工具脚本。
它的主要目的是：通过比较原始点云和滤波后的点云，使用 KNN（K近邻）方法判断哪些点是噪声点，并将这些噪声点保存为新的 .ply 文件。
功能：从原始点云和滤波点云中，利用 KNN 近邻算法，找出疑似“噪声点”（即被滤波器去除的部分）
输入：原始点云文件（.ply 格式）和滤波后的点云文件（.ply 格式）
输出：噪声点云文件（.ply 格式），仅包含所有被认为是噪声的点
核心方法：读取 PLY → 构建 KDTree → 计算近邻差异 → 提取噪声 → 保存 PLY
"""

def load_ply(filename):  #加载 .ply 文件为 pcl.PointCloud 对象
    # Load PLY file into pcl.PointCloud
    plydata = PlyData.read(filename)  #读取 PLY 文件
    point_cloud = pcl.PointCloud()  #创建 pcl.PointCloud 对象
    point_cloud.from_array(np.hstack((plydata['vertex']['x'][:, None],
                                       plydata['vertex']['y'][:, None],
                                       plydata['vertex']['z'][:, None])).astype(np.float32))  #将 PLY 数据转换为 pcl.PointCloud 格式，提取了其中的x,y,z坐标
    return point_cloud  #合并成点云数组后用 pcl.PointCloud 对象包装，返回。

def save_ply(filename, cloud):  #保存 pcl.PointCloud 对象为 .ply 文件
    # Save pcl.PointCloud to PLY file
    array = cloud.to_array()  #将 pcl.PointCloud 对象转换为 NumPy 数组
    data = np.empty(len(array), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])  #创建一个空的 NumPy 数组，指定数据类型为 float32,构造 PLY 文件结构（字段为 x/y/z）。
    for i in range(len(data)):  #遍历数组，将点云数据填充到 NumPy 数组中
        data[i] = tuple(array[i])  #将 pcl.PointCloud 中的点云数据转换为元组形式
    el = PlyElement.describe(data, 'vertex')  #创建 PlyElement 对象，描述点云数据
    PlyData([el]).write(filename)  #将 PlyElement 对象写入到指定的 PLY 文件中

def find_noise(original_cloud, filtered_cloud, k=10, threshold=0.01):  #利用 KNN 方法找出噪声点
    # Use K nearest neighbors to find noise points
    kdtree = original_cloud.make_kdtree_flann()  #创建 KDTree，用于快速查找近邻点
    noise_points = pcl.PointCloud()  #创建一个新的 pcl.PointCloud 对象，用于存储噪声点
    indices, _ = kdtree.nearest_k_search_for_cloud(filtered_cloud, k)  #对滤波后点云中每个点进行 KNN 搜索，找到每个点的 k 个近邻点
    for i in range(len(original_cloud)):  #遍历原始点云中的每个点
        if np.abs(np.mean(original_cloud[i]) - np.mean(filtered_cloud[indices[i]])) > threshold:  #如果原始点云点与其近邻点的平均值差异超过阈值，则认为该点是噪声点
            noise_points.append(original_cloud[i])  #将该点添加到噪声点云中
    return noise_points  #返回找到的噪声点云

def main():
    original_cloud = load_ply('/home/zxb/Poppy/Data/cloudviewer/patent/shebei/shebei.ply')
    filtered_cloud = load_ply('/home/zxb/Poppy/Data/cloudviewer/patent/shebei/hy/shebei_hy_20_0.1.ply')
    noise_points = find_noise(original_cloud, filtered_cloud)
    save_ply('noise_points.ply', noise_points)

if __name__ == "__main__":
    main()
