import numpy as np
import os
"""
功能是 批量处理 .txt 格式的点云文件，为每个文件添加一定比例的高斯噪声点，且确保噪声点的坐标不会超出原始点云的 XYZ 范围。输出是添加了噪声的新点云文件。

作用：
1）为每个 .txt 点云文件添加 5% 数量的高斯噪声点（原始点运输量的5%）
2）每个噪声点的 XYZ 坐标由高斯分布采样，但限制在原始点云的边界范围内
3）输出文件名加后缀 "_gauss5%_0.1m"，表示添加了高斯噪声点，且噪声强度为 0.1m
4）噪声点的标签设置为 1，表示这些点是噪声点
5）保留原始点云的强度和颜色信息
6）输出文件格式与原始点云一致，保留 7 列数据

输入格式：每个 .txt 文件是一个 N × 7 的点云数组
输出格式：每个 .txt 文件变成了 (N + 5%N) × 7，其中新增的点是噪声点，标签为 1
"""

def add_gaussian_noise_to_files(input_folder, output_folder, noise_limit=0.1):
    #input_folder: 输入文件夹路径
    #output_folder: 输出文件夹路径
    #noise_limit: 高斯噪声的标准差，默认值为 0.1m，这个值越小表示噪声越小，越接近原始点云
    if not os.path.exists(output_folder):  #检查输出文件夹是否存在，如果不存在则创建
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):  #遍历输入文件夹中的所有.txt文件
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)  #构建每个文件的完整路径
            #加载点云数据，假设每行格式为 x y z intensity r g b label，将.txt点云数据加载为 NumPy 数组
            data = np.loadtxt(file_path)

            # 计算要添加的噪声点的数量（约为原点数的 5%）
            num_noise_points = int(0.05 * data.shape[0])

            # 计算点云的XYZ轴的最大和最小值（用于限制噪声点的范围）
            xyz_min = np.min(data[:, 0:3], axis=0)
            xyz_max = np.max(data[:, 0:3], axis=0)

            # 创建一个空数组用于存储噪声点。
            noise_data_new = np.zeros((num_noise_points, 7))

            for i in range(num_noise_points):
                # 随机选择一个原始点（继承其强度和颜色信息）
                # 这里的 idx 是随机选择的原始点的索引
                # orig_point 是原始点的完整数据（包括强度和颜色信息）
                idx = np.random.randint(0, data.shape[0])
                orig_point = data[idx]

                # 创建高斯噪声点
                noise = np.random.normal(0, noise_limit, 3)  #生成一个均值为0，标准差为 noise_limit 的高斯噪声向量
                noise_point = orig_point[0:3] + noise  #将噪声添加到原始点的坐标上

                # 确保噪声点在XYZ轴的范围内
                noise_point = np.maximum(np.minimum(noise_point, xyz_max), xyz_min)  

                noise_data_new[i, 0:3] = noise_point  #将噪声点的坐标赋值
                noise_data_new[i, 3:6] = orig_point[3:6]  #将原始点的强度和颜色信息复制到噪声点

            noise_data_new[:, 6] = 1  # 所有噪声点的类别标签统一为 1。
            merged_data_new = np.vstack((data, noise_data_new))  #将原始点云数据和噪声点数据合并

            output_file_path = os.path.join(output_folder, filename.replace('.txt', '_gauss5%_0.1m.txt'))  #构建输出文件的路径，添加后缀 _gauss5%_0.1m
            #将合并后的数据保存到输出文件中，格式为浮点数
            np.savetxt(output_file_path, merged_data_new, fmt='%f')


if __name__ == "__main__":
    add_gaussian_noise_to_files('/home/zxb/Poppy/Data/paper_deepleaning_lvbo/fenlei_3/version3_data/lvbo_2us2.5_hy20_0.1/point cloud apart_txt/txt',
                                '/home/zxb/Poppy/Data/paper_deepleaning_lvbo/fenlei_3/version3_data/lvbo_2us2.5_hy20_0.1_apart_guass5%/txt_guass5%_0.1m')

