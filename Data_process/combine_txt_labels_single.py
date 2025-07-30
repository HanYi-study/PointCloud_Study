import os

'''
该脚本用于：
    将一对 .txt 文件（点云数据）和 .label 文件（标签）逐行拼接合并，输出成 _merged.txt 文件，保存在目标目录下。

与之前的脚本不同之处在于：
    本脚本只处理单个文件对；
    包含了 文件名一致性校验（保证 .txt 和 .label 文件是配对的）；
    逻辑更清晰、函数更独立。


转换前的数据：
    假设.txt 文件内容如下（每行表示一个点的坐标和颜色）：
    0.1 0.2 0.3 120 130 140
    0.4 0.5 0.6 110 120 130
    对应 .label 内容如下：
    1
    2
    合并后的 _merged.txt：
    0.1 0.2 0.3 120 130 140 1
    0.4 0.5 0.6 110 120 130 2

适用于：
    1.预测后：拼接模型输出标签以生成 .ply、.pcd 等可视化或评估格式
    2.可视化前处理：一行合并成完整属性，便于 MeshLab、Open3D、Potree 可视化
    3.数据对齐：保证预测标签与原始点云数据一一对应


两段代码功能与使用方式的区别
特征 / 脚本	 randla_combine_pred_txt_labels.py（当前）	  randla_combine_pred_txt_labels_pi.py（之前）
🎯 目标	         合并一个 .txt 与 .label 文件	             批量合并多个 .txt 与 .labels 文件
📂 输入目录	     直接传入单个文件路径	                      扫描整个文件夹下所有成对文件
🧪 校验机制	     检查 .txt 和 .label 文件是否同名	          默认假设同名 .txt 与 .labels 成对存在
📁 输出位置	      手动指定输出路径	                          默认输出到同一个 combine 文件夹
🔁 可复用性	     用于单文件调试或测试非常好	                   更适合批处理大规模预测结果

两种文件转换前后数据格式一样吗？最终的输出结构是完全一致的，两种方式只是处理的规模不同。
'''


def merge_files(txt_path, labels_path, new_path):  #用于将单个 .txt 文件和对应的 .labels 文件合并成一个新的 _merged.txt 文件
    with open(txt_path, 'r') as txt_file, open(labels_path, 'r') as labels_file, open(new_path, 'w') as new_file:  #同时打开 .txt 和 .labels 文件，以及新文件
        for txt_line, labels_line in zip(txt_file, labels_file):  #逐行读取 .txt 和 .labels 文件
            merged_line = txt_line.strip() + " " + labels_line.strip() + "\n"  #将两行内容合并，中间用空格分隔，并添加换行符
            new_file.write(merged_line)  #写入合并后的内容到新文件中   每一行拼接为：原txt内容 + 空格 + 标签值
#txt_path：点云数据路径（每行一般为 x y z r g b）；
#labels_path：对应预测标签路径（每行一个整数）；
#new_path：合并后输出文件路径；
#用 zip 同步读取两文件对应行;
#合并成：x y z r g b label
#写入到 new_path 指定的文件中。


# 输入.txt文件路径和.labels文件路径
txt_file_path = "/home/zxb/Poppy/kpconv_pred/save_folder/fenlei3_us1.5/npy_label/j27_1_fenlei_3_gauss_us1.5.txt"   #输入的点云数据文件路径
labels_file_path = "/home/zxb/Poppy/kpconv_pred/save_folder/fenlei3_us1.5/npy_label/j27_1_fenlei_3_gauss_us1.5.label"  #输入的标签文件路径
destination_folder = "/home/zxb/Poppy/kpconv_pred/save_folder/fenlei3_us1.5/npy_label/combine"   #输出目录的路径

# 提取文件名（不包括扩展名）作为合并后文件的文件名
txt_base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
print(txt_base_name)
labels_base_name = os.path.splitext(os.path.basename(labels_file_path))[0]
print(labels_base_name)  #打印两个基础文件名，便于调试确认是否一致。

# 确保两个文件具有相同的基本文件名
if txt_base_name != labels_base_name:  #如果两个文件的基本名不一致，则提示错误。若不匹配则报错终止，不会执行合并。
    print("错误：输入的文件名不匹配！")
else:  #如果匹配，则继续执行合并操作。
    new_filename = txt_base_name + "_merged.txt"  #新文件名为原文件名加上 _merged 后缀
    new_path = os.path.join(destination_folder, new_filename)  #构建新文件的完整路径
    merge_files(txt_file_path, labels_file_path, new_path)  #调用合并函数，将 .txt 和 .labels 文件合并到新文件中

    print("合并完成！新文件已创建：", new_path)
