import os

'''
change_3and5_txt.py 脚本是一个用于批量处理点云数据文本文件中颜色信息的转换脚本。  
只修改了每行的第4、5、6列（RGB颜色值），将其从浮点数转换为整数。（对应的序列号为3、4、5）

转换前第4~6列原始内容（R、G、B值）是浮点数格式（例如：0.123456），转换处理后第4-6列都变为了整数
变化是对RGB值进行了取整处理

处理的原因：
1.标准化 RGB 数据：
    很多点云处理模型（包括 RandLA-Net 可视化）要求颜色值为 [0, 255] 的整数，浮点数可能引起格式错误或显示不正确。
2.兼容常见工具：
    如 PCL、Open3D、MeshLab 等软件读取 .txt 或 .ply 点云时，RGB 是整数更容易渲染。
3.压缩数据大小：
    整数格式通常比浮点数占用更少的存储空间，尤其在大规模点云数据中。但是不是主要目的。

输入的是txt文件
输出的是txt文件


'''


# 输入和输出文件夹路径
input_folder = 'txt文件上层目录文件夹'  # 替换为您的输入文件夹路径
output_folder = 'txt文件上层目录文件夹'  # 替换为您的输出文件夹路径
#input_folder 包含原始 .txt 点云数据文件的目录
#output_folder 用于保存处理后的 .txt 文件的目录



# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#如果输出目录 xyzirgb 不存在，则创建它，以便后续能写入新文件。


# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder, filename)  # 构建输入文件的完整路径
        output_file_path = os.path.join(output_folder, filename)  # 构建输出文件的完整路径

        # 读取输入文件并处理内容
        with open(input_file_path, 'r') as input_file:  #逐行读取点云数据文本文件，每行其中前3列是 XYZ，第4-6列是 RGB（颜色信息）。
            lines = input_file.readlines()  # 读取所有行

        processed_lines = []  # 初始化一个空列表，用来存储处理后的每一行。
        for line in lines:  # 遍历每一行
            columns = line.strip().split()  # 按空格分割行内容  strip()：移除行尾换行符或空格；  split()：按空格分割成单个字段列表
            if len(columns) >= 6:   #如果这行有 至少6个元素（即 XYZRGB）才进行处理。
                try:
                    columns[3] = str(int(float(columns[3])))  # 将第四列转换为整数
                    columns[4] = str(int(float(columns[4])))  # 将第五列转换为整数
                    columns[5] = str(int(float(columns[5])))  # 将第六列转换为整数
                    #尝试将第4、5、6列（RGB）先转为 float，再转为 int，然后再转回 str，用于后面拼接。
                except ValueError:
                    pass  # 如果有任何值无法转换为浮点数/整数（例如值是 nan 或字母），就跳过这一行。
            processed_line = ' '.join(columns) + '\n'  #将处理好的 columns 拼接为一行字符串，并添加换行符 \n
            processed_lines.append(processed_line)  #然后加入到最终要写入的列表中

        # 将处理后的内容写入输出文件
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(processed_lines)

        print(f"Processed {filename} and saved to {output_file_path}")
