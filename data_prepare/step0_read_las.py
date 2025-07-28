import laspy
import numpy as np

def read(file_name):
    las = laspy.read(file_name)
    # 读取 .las 文件，返回一个 laspy.LasData 对象，包含点云的所有信息（坐标、颜色、分类等）。
    points_x = las.x.array * las.x.scale + las.x.offset - las.header.x_min
    points_y = las.y.array * las.y.scale + las.y.offset - las.header.y_min
    points_z = las.z.array * las.z.scale + las.z.offset - las.header.z_min
    # 读取点云的 x、y、z 坐标，并进行缩放和偏移处理，确保坐标值在正确的范围内。
    # las.x.array：获取 x 坐标的原始数组,即为原始整数坐标
    # las.x.scale 和 las.x.offset：应用缩放和偏移，确保坐标值正确。scale 和 offset 是 .las 文件中的坐标缩放和偏移信息。
    # las.header.x_min：将点云坐标归一化（通常为了让点云坐标起始从 (0,0,0) 左右开始，便于计算/显示）。
    min = np.array([las.header.x_min, las.header.y_min, las.header.z_min])
    # 获取点云的最小坐标值，用于后续处理或归一化。

    #提取点云颜色信息（RGB)
    points_red = las.red.reshape(-1, 1)
    points_green = las.green.reshape(-1, 1)
    points_blue = las.blue.reshape(-1, 1)

    print(min) # 打印点云原始的最小坐标值（x_min, y_min, z_min），用于调试或记录原始数据起点。
    points_x = points_x.reshape(-1, 1)
    points_y = points_y.reshape(-1, 1)
    points_z = points_z.reshape(-1, 1)
    # 将 x、y、z 坐标转换为列向量，以便后续处理。
    # points_x.shape, points_y.shape, points_z.shape 从一维数组转换为了二维数组
    # points_x.shape, points_y.shape, points_z.shape 分别为 (7269967, 1)，表示每个点的坐标都是一个单独的列向量。
    # points_xyz.shape, points_color.shape 分别为 (7269967, 3)，表示每个点的坐标和颜色都是一个三维向量。即点云坐标和颜色信息的矩阵

    #合并成最终输出格式
    points_xyz  = np.hstack((points_x, points_y, points_z))  #points_xyz 是 (N, 3) 的坐标矩阵。
    points_color = np.hstack((points_red, points_green, points_blue))  #points_color 是 (N, 3) 的颜色矩阵。
    points_label = las.classification.array  #points_label 是 (N,) 的每个点的分类标签（ground, building, tree 等）。


    # print('points_xyz----', type(points_xyz), points_xyz.shape)
    # print('points_label----', type(points_label), points_label.shape)
    # points_xyz---- <class 'numpy.ndarray'> (7269967, 3)
    # points_label---- <class 'numpy.ndarray'> (7269967,)
    return points_xyz.astype(np.float32), points_label, min, points_color
    #points_xyz：归一化后的点云坐标。  points_label：每个点的语义标签。  min：原始坐标最小值（可能用于恢复全局坐标）。  points_color：RGB颜色信息。

if __name__ == "__main__":
    points_xyz, points_label, points_min, points_color = read('/home/hy/projects/PointCloud_Code/Data/Data_prepare/3-4(3_4).las')
    #新添加了下面几行代码，因为我想把他保存为输出文件，且是.txt的文件
    output_path = '/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step0_result/3-4(3_4).txt'
    data_out = np.hstack((points_xyz, points_color, np.zeros((points_xyz.shape[0], 1)), points_label.reshape(-1, 1)))
    # 插入一个全为0的 instance_id 列
    np.savetxt(output_path, data_out, fmt='%.6f')

    #las文件应该是一个带标签的、带颜色的点云数据集，可能来自三维扫描平台如 LiDAR 或 photogrammetry。
    #这个路径指向一个 .las 文件，这是标准的激光雷达点云数据格式。
    #他应该具备以下结构：
   # | x    | y   | z    | red   | green  | blue   | classification |
   # | 浮点 | 浮点 | 浮点 | 整型  | 整型    | 整型   | 整型            |
   #read（）会返回什么：points_xyz：(N, 3) 的点云坐标矩阵     points_label：(N,) 的整型分类标签数组（每个点一个类别）  points_min：(3,) 的三维最小坐标值，用于后续还原或偏移   points_color：(N, 3) 的颜色矩阵（R, G, B）值
