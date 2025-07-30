import os
import shutil  #用于文件移动

'''
该脚本的作用是： 将点云 .txt 文件与对应的 .labels 文件按行对齐后合并成一个新的 _merged.txt 文件，并保存在指定目录中（combine 文件夹）。

这种操作常见于：
    预测结果可视化前的数据整合；
    将点云数据（如 xyzrgb）与每个点的预测标签拼接；
    准备下游任务的数据格式（如用于评估、转换成 .ply 等）。

数据转换之前：
    每个点云数据文件（.txt）和对应的标签文件（.labels）是分开的。（以原始点云.txt文件及其对应的.labels文件为例）
    原始.txt文件（点云数据）：
    1.23 4.56 7.89 100 100 100
    2.34 5.67 8.90 120 120 120
    对应.labels文件（标签）：
    0
    1
    合并后的_merged.txt文件：
    1.23 4.56 7.89 100 100 100 0
    2.34 5.67 8.90 120 120 120 1

这种合并操作常见于：
    1.模型预测后：将 .labels（预测结果）拼接回点云，用于可视化或评估
    2.准备 .ply 文件：有些 .ply 格式需要 xyzrgb+label 组合
    3.统一数据结构：下游评估函数或绘图工具要求一行表示一个完整点，包括预测


'''


# 原始文件夹路径和目标文件夹路径
source_folder = "输入的文件夹包含.txt文件和.labels文件，输入的是该文件夹的路径"    #输入目录路径
destination_folder = "合并后的文件存放的目录路径"                                #输出目录路径

# 获取原始文件夹中的所有文件，并将获取的文件名存放在一个列表中
files = os.listdir(source_folder)

# 创建一个空字典用于存放文件配对信息：键是公共前缀，值是对应的 .txt 和 .labels 文件路径元组。
file_dict = {}

# 遍历文件列表，将同名的.txt和.labels文件配对
for filename in files:
    if filename.endswith(".txt"):  #只处理.txt文件
        base_name = os.path.splitext(filename)[0]  #去掉文件后缀名，获取公共前缀
        txt_path = os.path.join(source_folder, filename)  #构建完整的 .txt 文件路径
        labels_filename = base_name + ".labels"  #构建对应的 .labels 文件名
        labels_path = os.path.join(source_folder, labels_filename)  #构建完整的 .labels 文件路径

        if os.path.exists(labels_path):  #检查对应的 .labels 文件是否存在
            file_dict[base_name] = (txt_path, labels_path)  #若存在，将 .txt 和 .labels 文件路径存入字典中

# 遍历配对好的文件字典，合并内容并将新文件放入目标文件夹
for base_name, (txt_path, labels_path) in file_dict.items():  # 遍历字典中的每个键值对
    new_filename = base_name + "_merged.txt"  #新文件名为原文件名加上 _merged 后缀
    new_path = os.path.join(destination_folder, new_filename)  #构建新文件的完整路径

    with open(txt_path, 'r') as txt_file, open(labels_path, 'r') as labels_file, open(new_path, 'w') as new_file:  #同时打开 .txt 和 .labels 文件，以及新文件
        for txt_line, labels_line in zip(txt_file, labels_file):  #逐行读取 .txt 和 .labels 文件
            merged_line = txt_line.strip() + " " + labels_line.strip() + "\n"  #将两行内容合并，中间用空格分隔，并添加换行符
            new_file.write(merged_line)  #写入合并后的内容到新文件中   每一行拼接为：原txt内容 + 空格 + 标签值

'''
# 移动新文件到目标文件夹
for base_name in file_dict.keys():  #
    new_filename = base_name + "_merged.txt"
    new_path = os.path.join(destination_folder, new_filename)
    shutil.move(new_path, destination_folder)

'''#这一段代码冗余

print("success！")
