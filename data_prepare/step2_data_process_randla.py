import os
import numpy as np

def process_files(input_dir, output_dir):
    """
        该函数的作用是批量处理指定目录下的所有 .txt 文件，将每个文件的内容进行特定的转换，并将结果保存到另一个目录中。
        对每个点云文件提取标签（最后一列），修改点云特征(在每行的第四列插入0),并将处理后的数据和标签分别保存到输出目录中。
        输入：
        input_dir:一个目录路径，里面放的是点云数据的文本文件,每个文件的结构通常如下(x, y, z, intensity, label)，其中最后一列是标签。
        x, y, z 是点云的空间坐标,intensity 是反射强度值或其他特征,label 是类别标签。
        这些 .txt 文件被视为点云采样数据的格式化存储方式。

        输出：是保存在output_dir目录下的两个文件，一个是处理后的点云特征数据文件，另一个是标签文件。
        output_dir:
        (1)处理后的特征数据文件(同名.txt文件),把原数据中除了最后一列(label)以外的列都提取出来，在第4列插入值0，用于之后的模型输入、特征工程、数据增强等
        (2)标签文件(同名.label文件),只保存每一个点的标签（最后一列0,通常用于分类监督学习模型的目标值

    """

    if not os.path.exists(output_dir):  #检查目录是否已存在。
        os.makedirs(output_dir)  #如果不存在，则创建该目录（包括任何必要的父目录）。
    

    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]  #获取指定目录下所有以.txt结尾的文件名列表。
    # os.listdir(input_dir)：列出输入目录中的所有文件。
    # f.endswith('.txt')：筛选出所有以.txt结尾的文件。
    # 最终 txt_files 是一个包含所有 .txt 文件名的列表。

    for txt_file in txt_files:  #遍历每个文件名。
        data = np.loadtxt(os.path.join(input_dir, txt_file))  #加载每个 .txt 文件的数据。
        # os.path.join(input_dir, txt_file)：构建完整的文件路径。
        # np.loadtxt()：读取文件内容并将其转换为 NumPy 数组
        labels = data[:, -1].astype(int)  #提取每行的最后一列作为标签，并将其转换为整数类型。
        # data[:, -1]：获取每行的最后一列。
        # .astype(int)：将标签转换为整数类型。
        new_data = np.insert(data[:, :-1], 3, 0, axis=1)  #在每行的第四列插入0，axis=1表示沿着列方向插入。
        # data[:, :-1]：获取每行的所有列，除了最后一列
        # np.insert(data[:, :-1], 3, 0, axis=1)：在第3列位置插入一个值为 0 的新列（索引从0开始，所以这是“第4列”）。axis=1：表示按列操作。
        #插入后生成的新数组命名为 new_data。

        np.savetxt(os.path.join(output_dir, txt_file), new_data, fmt='%f') #将处理后的数据保存到输出目录中，文件名与原文件相同，但内容已修改。
        # os.path.join(output_dir, txt_file)：构建输出文件的完整路径。
        # np.savetxt()：将 NumPy 数组保存为文本文件，fmt='%f' 表示以浮点数格式保存。
        #保存的是新特征数据文件
        np.savetxt(os.path.join(output_dir, txt_file.replace('.txt', '.label')), labels, fmt='%d')
        #将标签保存为单独的 .label 文件。
        #txt_file.replace('.txt', '.label')：将文件扩展名从 .txt 改为 .label。
        #fmt='%d'：保存为整数格式。
        #保存的是标签文件





# replace 'input_dir' and 'output_dir' with your actual directories
# process_files('input_dir', 'output_dir')
if __name__ == '__main__':
    process_files('/home/hy/projects/PointCloud_Code/Results/data_conversion_result/H3D_points-1_ply_to_txt', '/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step2_result')