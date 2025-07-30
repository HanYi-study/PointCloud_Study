import os



'''

def convert_ply_to_txt(ply_filename, txt_filename):
    with open(ply_filename, 'r') as ply_file, open(txt_filename, 'w') as txt_file:  # 打开PLY文件和TXT文件
        # 跳过头部
        for line in ply_file:  # 逐行读取PLY文件
            if line.strip() == "end_header":  # 找到头部结束标志
                break  # 跳出循环，开始复制数据

        # 复制数据
        for line in ply_file:  # 继续读取剩余的行
            txt_file.write(line)  # 将数据写入TXT文件

def convert_folder_ply_to_txt(folder_path):
    for filename in os.listdir(folder_path):  # 遍历文件夹中的所有文件
        if filename.endswith(".ply"):  # 检查文件是否为PLY格式
            ply_filename = os.path.join(folder_path, filename)  # 构建PLY文件的完整路径
            txt_filename = os.path.join(folder_path, filename[:-4] + ".txt")
            convert_ply_to_txt(ply_filename, txt_filename)
            print(f"Converted {ply_filename} to {txt_filename}")

# 修改为包含PLY文件的文件夹的路径
folder_path = '/home/zxb/Poppy/Data/paper_deepleaning_lvbo/fenlei_3/version3_data/lvbo_2us2.5/update_ok/hy20_0.1_ply'
convert_folder_ply_to_txt(folder_path)
'''

from plyfile import PlyData
import numpy as np


def convert_ply_to_txt_correctly(ply_path, txt_path):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']

    # 提取 x, y, z，如果有 r, g, b 也一并提取
    fields = ['x', 'y', 'z']
    if all(c in vertex_data.data.dtype.names for c in ['red', 'green', 'blue']):
        fields += ['red', 'green', 'blue']

    points = np.stack([vertex_data[c] for c in fields], axis=-1)
    np.savetxt(txt_path, points, fmt='%.6f')

    print(f"✅ Converted binary PLY: {ply_path} → {txt_path}")

def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".ply"):
            ply_path = os.path.join(input_dir, fname)
            txt_path = os.path.join(output_dir, fname.replace(".ply", ".txt"))
            convert_ply_to_txt_correctly(ply_path, txt_path)

# 修改路径
input_dir = '/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step1_H3D_result/points-1'
output_dir = '/home/hy/projects/PointCloud_Code/Results/data_conversion_result/H3D_points-1_ply_to_txt'

batch_convert(input_dir, output_dir)

