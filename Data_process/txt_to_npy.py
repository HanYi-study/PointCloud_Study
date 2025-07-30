# 批量txt转化为npy（一个文件夹内的所有txt）
import os
import numpy as np

'''
该脚本的主要功能是：
    批量将一个文件夹中的所有 .txt 点云数据文件转换为 .npy 格式文件，并保存在指定目录下。
    这是点云数据处理中的常见预处理步骤之一，用于将文本格式的点数据高效转换为 NumPy 二进制格式，加快后续的数据加载与训练速度。


将多个 .txt 点云文件（通常是 xyz、xyzrgb、xyzirgb 格式）转换为 .npy 格式；
用于：
    后续模型训练（如 RandLA-Net、KPConv）；
    加速数据加载；
    数据归档或备份；


'''


# 设置原始文件夹和目标文件夹的路径
input_folder = '.txt 格式的点云数据所在目录'     # txt文件所在文件夹
output_folder = '.npy 文件保存的目标目录'    # npy文件所在文件

# 如果目标文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历原始文件夹中的所有txt文件，并将其转换为npy文件并保存到目标文件夹中
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):  
        # 读取txt文件中的数据
        filepath = os.path.join(input_folder, filename)  #构造 .txt 文件的完整路径；
        data = np.loadtxt(filepath)   #使用 np.loadtxt 加载数据为二维数组，默认以空格/Tab 分隔。

        # 将数据保存为npy文件
        output_filename = os.path.splitext(filename)[0] + '.npy'  #将 .txt 改为 .npy
        output_filepath = os.path.join(output_folder, output_filename)   #生成完整路径。
        np.save(output_filepath, data)   #使用 NumPy 内置函数 np.save 保存为 .npy 格式；
        #.npy 是 NumPy 的高效二进制格式，适用于快速读写大规模数组。



