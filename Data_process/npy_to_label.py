import numpy as np  #numpy 用于加载 .npy 数据
import os

'''
该文件的作用：将深度学习模型（如 KPConv）输出的 .npy 文件格式标签转换为标准的 .label 文本格式文件（每行一个整数标签）。
本质：从二进制 .npy 数据格式 → 文本格式 .label

输入.npy（预测标签）：array([2., 0., 1., 1., 2.], dtype=float32)
输出.label（整数标签）：
2
0
1
1
2

常见用途：
KPConv、RandLA-Net 等模型输出 .npy 标签后转换为 .label 格式用于：
    1.可视化工具（如 SemanticKITTI, Open3D, CloudCompare）加载 .label
    2.与 ground truth .label 进行 mIoU/acc 评估
    3.其他框架继续处理（如投影回原图、构建点云颜色、渲染语义标签等）

转换的原因：
1.标准化标签格式：.label 是常用的点云语义分割标签格式，便于与其他工具兼容。
2.简化处理：.npy 格式通常用于模型训练/推理，而 .label 格式更适合后处理和可视化。
3.便于评估：.label 格式可以直接用于计算指标（如 mIoU, accuracy）与 ground truth 比较。

'''

input_folder = "模型输出的.npy格式的文件所在的目录路径"  # 替换为包含npy文件的文件夹路径
output_folder = "你想把转换为.label格式的文件存在什么目录下？就把这个目录的路径写在这里"  # 替换为保存.label格式文件的文件夹路径

if not os.path.exists(output_folder):
    os.makedirs(output_folder)  #如果输出路径不存在，就创建它（自动建目录）

npy_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]  #遍历输入文件夹中所有 .npy 文件，构成列表

for npy_file in npy_files:  #对每个 .npy 文件进行处理
    npy_path = os.path.join(input_folder, npy_file)  #构建完整的文件路径
    data = np.load(npy_path)  #使用 np.load 加载该 .npy 文件（通常是一维数组或二维预测标签）

    # 将浮点数转换为整数，这里可以根据需要进行调整
    data_int = data.astype(int)  #将 .npy 中的值（如果是浮点）转换为整数，符合 .label 文件格式要求

    # 生成保存.label格式文件的路径
    label_file = os.path.splitext(npy_file)[0] + ".label"  #构造 .label 文件名与路径：将原 .npy 文件名改后缀为 .label
    label_path = os.path.join(output_folder, label_file)
    # 将整数数据写入.label格式文件
    with open(label_path, "w") as f:  #打开输出文件，并将数据按行写入（每行一个整数标签）
        for value in data_int.flatten():
            f.write(str(value) + "\n")

print("转换完成！")
