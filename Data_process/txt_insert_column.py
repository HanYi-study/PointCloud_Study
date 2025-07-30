import os
import numpy as np

'''
该脚本的主要功能是：
    批量处理一个文件夹下的所有 .txt 点云文件，在每个文件的第三列和第四列之间插入一列全为 0 的值，最后将结果保存到另一个目标目录中。

    输入数据格式一般为形如 [x, y, z, r, g, b]。
    插入之后变为 [x, y, z, 0, r, g, b]，即添加一个“伪 intensity”或空白列用于后续处理。

适合以下任务：
    在没有 intensity（强度）通道的点云中插入虚拟强度值；
    为了对齐某些训练/测试模型所要求的 xyzirgb 数据格式；
    为了构建统一的数据维度做预处理。




'''

source_folder = "源文件夹路径，源文件夹中待处理文件是x y z r g b"
target_folder = "输出文件夹路径，处理后文件的内容是x y z i r g b"

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 获取源文件夹中的所有 txt 文件
txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]

for txt_file in txt_files:
    # 读取 txt 文件内容
    data = np.loadtxt(os.path.join(source_folder, txt_file))   #加载点云数据（每行为一个点，多列如 x, y, z, r, g, b）。

    # 创建一个全为0的列
    zero_column = np.zeros((data.shape[0], 1))  #构造一个长度与点数一致的全零列。

    # 在第三列和第四列之间插入这个列
    data = np.hstack((data[:, :3], zero_column, data[:, 3:]))  #将 zero_column 插入原数据的第 3 和第 4 列之间，即结构从 [x y z r g b] → [x y z 0 r g b]。

    # 保存修改后的数据到新的文件夹
    np.savetxt(os.path.join(target_folder, txt_file), data, fmt="%f")

