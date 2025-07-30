import numpy as np

'''
功能是：在原始点云中随机加入 10% 的高斯噪声点（有标签），并将新生成的数据保存为一个 .txt 文件。

输入：原始点云文件（.txt），每行 7 个值（x, y, z, r, g, b, label）
输出：保存一个新 .txt 文件，格式仍为 7 列。
操作：在原始点云的基础上添加一定数量的伪造“噪声点”，这些点在空间中位置扰动，但颜色值不变，且标签为 1。


'''


# 1. 读取文件并加载点云数据
data = np.loadtxt("/mnt/data/j22_1_fenlei_3.txt")  #原始点云数据.txt文件

# 2. 计算要添加的噪声点的数量
num_noise_points = int(0.10 * data.shape[0])  #设定：要生成的噪声点数目是原始点数量的 10%。

# 计算点云的中心和半径
center = np.mean(data[:, 0:3], axis=0)  #center：原始点云的中心（x, y, z 的平均值）
radii = np.linalg.norm(data[:, 0:3] - center, axis=1)  #radii：每个点到中心的距离（欧几里得距离）
mean_radius = np.mean(radii)  #mean_radius：平均距离
std_radius = np.std(radii)  #std_radius：距离的标准差

# 3. 创建一个新的噪声数据数组
noise_data_new = np.zeros((num_noise_points, 7))  #创建空数组 noise_data_new 用于存放噪声点，格式保持与原始数据一致（7列）。

# 对每一个噪声点，随机选择一个原始点，然后在其位置上添加一个小的高斯噪声
for i in range(num_noise_points):  # 开始生成每一个噪声点。
    # 随机选择一个原始点
    idx = np.random.randint(0, data.shape[0])
    orig_point = data[idx]  #从原始点中随机选一个点作为“基点”。

    # 随机选择一个方向并添加高斯噪声
    random_direction = np.random.randn(3)  # 3D随机方向
    random_direction /= np.linalg.norm(random_direction)  # 单位化
    #生成一个随机方向向量（三维），然后单位化，使其长度为 1。
    noise_magnitude = mean_radius + np.random.normal(0, std_radius * 0.1)  # 控制噪声大小  #生成一个符合高斯分布的扰动量（std_radius * 0.1 控制标准差）
    noise = noise_magnitude * random_direction    #沿着随机方向应用扰动（生成噪声点的位置偏移）。

    # 将噪声添加到原始点的位置
    noise_data_new[i, 0:3] = orig_point[0:3] + noise  #位置加上扰动后的新坐标
    noise_data_new[i, 3:6] = orig_point[3:6]  #颜色保持原始点的 r, g, b 不变

# 4. 设置噪声标签为1
noise_data_new[:, 6] = 1

# 5. 合并原始点云和噪声点云
merged_data_new = np.vstack((data, noise_data_new))  #将原始数据和生成的噪声数据按行堆叠（vstack），变成一个新的点云数据集。

# 6. 保存到新的文件
output_path = "/path/to/save/j22_1_gauss_v2.txt"   #将最终结果保存为 .txt 文件，每行 7 列（x, y, z, r, g, b, label），格式为浮点数。
np.savetxt(output_path, merged_data_new, fmt='%f')
