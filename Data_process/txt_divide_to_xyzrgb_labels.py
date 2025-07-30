import os
import numpy as np

'''
该脚本的作用是：
    将一个包含 xyzrgb+label 的 .txt 点云文件进行分离处理：
        提取前6列保存为 xyzrgb 格式；
        提取最后一列保存为 .labels 格式。

该操作常用于 语义分割任务中的数据预处理阶段，例如：
    RandLA-Net、KPConv 等模型都需要 .xyzrgb 和 .labels 分开处理。


输入文件：  .txt格式
     1.234 2.345 3.456 120 140 180 2
     4.321 1.876 0.765 100 130 160 0
     
输出文件：  （1）xyzrgb文件-------.txt格式，文件名与原始输入文件同名；
           1.234000 2.345000 3.456000 120 140 180
           4.321000 1.876000 0.765000 100 130 160
           float float float int int int
           （2）label标签文件-----.labels文件，文件名与原始输入文件同名；
           2
           0

           

'''

source_folder = "含有完整数据的 .txt 文件（通常包含 x y z r g b label）所在目录的路径"
xyzrgb_folder = "将前六列x y z r g b分离出来后的保存目录"
label_folder = "将最后一列（标签，labels）保存成 .labels 文件的目录，所在路径"

# 创建目标文件夹（如果它们不存在）
if not os.path.exists(xyzrgb_folder):
    os.makedirs(xyzrgb_folder)

if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# 获取源文件夹中的所有 txt 文件
txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]

for txt_file in txt_files:
    # 读取 txt 文件内容
    data = np.loadtxt(os.path.join(source_folder, txt_file))  #读取每一个 .txt 文件为二维数组

    # 提取前六列并保存到新的 txt 文件
    xyzrgb_data = data[:, :6]   #提取前6列：x y z r g b；
    np.savetxt(os.path.join(xyzrgb_folder, txt_file), xyzrgb_data, fmt="%f %f %f %d %d %d")  #写入对应的 .txt 文件到 xyzrgb_folder；   前三列为浮点，后三列为整数（颜色通道）。

    # 提取最后一列，转换为 int，并保存到新的 .labels 文件
    labels = data[:, -1].astype(int)   #提取最后一列作为标签，强制转换为整数；
    label_filename = os.path.splitext(txt_file)[0] + ".labels"   #使用 .labels 后缀保存，格式为每行一个类别值。
    np.savetxt(os.path.join(label_folder, label_filename), labels, fmt="%d")

