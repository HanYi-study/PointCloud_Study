import os
import numpy as np  #numpy: 用于加载 .npy 文件和保存 .txt

'''
该脚本的作用是：
将深度学习中模型生成的 .npy 文件（通常是预测结果或特征）转换为普通 .txt 文本文件，方便查看、编辑、手动分析或用于其他程序处理。

输入的是.npy文件（内容为一个数组）：
array([[1.1, 2.2, 3.3],
       [4.4, 5.5, 6.6]], dtype=float32)
输出的是.txt文件（内容为文本格式的数组）：
1.1 2.2 3.3
4.4 5.5 6.6

应用场景：
1.模型输出转换：将模型预测的标签、特征等从二进制 .npy 格式转换为可读的文本格式，便于后续处理。
2.数据可视化：将 .npy 数据转换为 .txt 格式后，可以使用文本编辑器或其他工具查看数据内容。



'''

# 设置原始文件夹和目标文件夹的路径
input_folder = '存放模型生成的npy文件的目录的路径'     # npy文件所在文件夹
output_folder = '该路径用于存放转换后的 .txt 文件'    # txt文件所在文件夹

# 如果目标文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历原始文件夹中的所有npy文件，并将其转换为txt文件并保存到目标文件夹中
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        # 读取npy文件中的数据
        filepath = os.path.join(input_folder, filename)
        data = np.load(filepath)  #读取当前 .npy 文件中的数据，通常是模型输出，如标签数组、特征向量、概率等


        # 将数据保存为txt文件
        output_filename = os.path.splitext(filename)[0] + '.txt'  #将文件后缀从 .npy 改成 .txt
        output_filepath = os.path.join(output_folder, output_filename)  #并构造输出路径
        np.savetxt(output_filepath, data)  #使用 numpy.savetxt() 将数据保存为 .txt 文件（以默认格式写出，通常是空格分隔）


