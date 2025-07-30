import numpy as np  #NumPy用于数组和数学计算
import os  #os用于文件和目录操作

"""
  现实中的点云数据，例如来自 LiDAR、RGB-D摄像头、结构光扫描仪等，通常包含以下问题：
    1. 点位偏差（由测量误差造成）
    2. 传感器噪声（如激光回波不稳定）
    3. 重建误差（如SLAM系统重定位）
    4. 数据稀疏或遮挡
    为了让模型更鲁棒，我们模拟这种不完美的数据环境，让网络学会“在噪声环境中也能识别结构”。

高斯噪声（Gaussian Noise），又称为正态分布噪声：
   每个坐标值上加上一个服从正态分布的随机扰动，使得点的位置略微变化，就像自然抖动一样。
   需要在原坐标值上分别添加均值为0，标准差为某个比例的随机数，即可实现高斯噪声的添加

  添加高斯噪声的核心目的：提升模型的泛化能力
  添加高斯噪声是一种数据增强手段
  该脚本批量读取一个文件夹下的 .txt 点云数据文件，为每个点云添加模拟的高斯噪声点（数量约为原点数的 5%），然后将结果保存到另一个指定的输出文件夹中，文件名加后缀 _gauss。
"""
def add_gaussian_noise_to_files(input_folder, output_folder):  #主函数：将input_folder中的所有.txt文件添加高斯噪声并保存到output_folder
    if not os.path.exists(output_folder):  #检查输出文件夹是否存在，如果不存在则创建
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):  #遍历输入文件夹中的所有.txt文件
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)  #构建每个文件的完整路径
            data = np.loadtxt(file_path)  #加载点云数据，假设每行格式为 x y z intensity r g b label，将.txt点云数据加载为 NumPy 数组

            # 计算要添加的噪声点的数量（约为原点数的 5%）  即要添加这么多个噪声点
            num_noise_points = int(0.05 * data.shape[0])

            # 计算点云的中心和半径
            center = np.mean(data[:, 0:3], axis=0)  #计算点云的中心点
            radii = np.linalg.norm(data[:, 0:3] - center, axis=1)  #计算每个点到中心点的距离
            mean_radius = np.mean(radii)  #计算距离的平均值     均值
            std_radius = np.std(radii)  #计算距离的标准差       标准差
            #得到所有半径的平均值 mean_radius 和标准差 std_radius，用于生成噪声强度

            # 初始化一个新的数组来存储生成的噪声点（与原始点维度一致，7维）
            noise_data_new = np.zeros((num_noise_points, 7))

            for i in range(num_noise_points):  #为每个噪声点生成高斯噪声
                idx = np.random.randint(0, data.shape[0])
                orig_point = data[idx]  #随机选择一个原始点作为噪声点的基础

                random_direction = np.random.randn(3)  #生成一个随机方向向量
                random_direction /= np.linalg.norm(random_direction)  #归一化该向量
                #上面两行代码生成一个随机方向向量，并将其归一化，使其长度为 1，即单位随机方向向量，用于偏移

                noise_magnitude = mean_radius + np.random.normal(0, std_radius * 0.1)  #生成噪声的强度，基于平均半径和标准差
                noise = noise_magnitude * random_direction  #生成的噪声强度与方向向量相乘得到实际偏移量。

                noise_data_new[i, 0:3] = orig_point[0:3] + noise  #将噪声添加到原始点的坐标上
                noise_data_new[i, 3:6] = orig_point[3:6]  #将原始点的强度和颜色信息复制到噪声点

            noise_data_new[:, 6] = 1  #将噪声点的标签（第七列/第七维）设置为 1，表示这些点是噪声点
            merged_data_new = np.vstack((data, noise_data_new))  #将原始点云数据和噪声点数据合并

            output_file_path = os.path.join(output_folder, filename.replace('.txt', '_gauss.txt'))  #构建输出文件的路径，添加后缀 _gauss
            np.savetxt(output_file_path, merged_data_new, fmt='%f')  #将合并后的数据保存到输出文件中，格式为浮点数

# 将一个文件夹中的所有txt进行批量处理,txt_insert_guass
if __name__ == "__main__":
    add_gaussian_noise_to_files('/home/zxb/Poppy/Data/paper_deepleaning_lvbo/fenlei_3/version3_data/lvbo_2us2.5_hy20_0.1/apartment/txt',
                                '/home/zxb/Poppy/Data/paper_deepleaning_lvbo/fenlei_3/version3_data/lvbo_2us2.5_hy20_0.1_apart_guass5%/txt')
"""
输入文件夹：包含多个.txt 点云数据文件的文件夹，每个.txt文件包含点云数据，每行代表一个点，格式为 x y z intensity r g b label。
输出文件夹：处理后的点云数据文件将保存到此文件夹中，文件名将添加后缀 _gauss。对应的.txt 文件将包含原始点云数据和添加的高斯噪声点，格式与输入文件相同。
每个输出文件的点数约等于原点数×1.05，噪声点的数量约为原点数的 5%。每一行包含点的坐标 (x, y, z)、强度 (intensity)、颜色 (r, g, b) 和标签 (label)，噪声点的标签为 1（最后一列标记是否为噪声点）。
"""
