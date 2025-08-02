'''
import os 
import numpy as np  # 数值计算
from sklearn.metrics import confusion_matrix  # 混淆矩阵评估


功能简介：
  对一个文件夹中的多个预测文件（*_pred.txt）与对应的真实标签文件（*_label.txt）进行批量比对，计算总体的 mIoU（mean Intersection over Union）和 OA（Overall Accuracy），
  并输出各个类别的 IoU 分数。适用于点云分类任务的结果评估。

输入文件格式：
存在于同一个目录中，成对命名：
  scene01_label.txt
  scene01_pred.txt
  scene02_label.txt
  scene02_pred.txt
  ...
每个文件格式为纯文本 .txt，每行一个整数标签（可为 float 被转成 int），如下：
  0
  0
  1
  2
  1
  0
输出文件格式：
  写入一个文本 .txt 文件，内容格式如：
  mIoU: 0.6784
  OA: 0.8453
  IoU for class 0: 0.7931
  IoU for class 1: 0.7047
  IoU for class 2: 0.6542
  IoU for class 3: 0.5132
  IoU for class 4: 0.6120
  IoU for class 5: 0.4437

'''

''' #--------------------------------------------------------------------------------------------------------------------------------------------------------版本1.0
def calculate_metrics(true_labels, pred_labels, num_classes=6):  # 定义评估指标计算函数，num_classes=6 表示有 6 个类别（你可以根据实际类别数量修改）。
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    # 计算混淆矩阵（confusion matrix），返回一个 num_classes x num_classes 的矩阵。
    intersection = np.diag(cm)
    # 交集：混淆矩阵对角线元素为预测正确的点数（即 TP）。
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    # 并集：预测+真实 - 重叠区域（交集）= IoU 中的分母。
    iou = intersection / union.astype(np.float32)
    # 分类别计算 IoU，得到一个形如 [0.6, 0.7, 0.5, ...] 的数组。
    miou = np.mean(iou)
    # mIoU：取 IoU 平均；
    oa = np.sum(intersection) / np.sum(cm)
    # OA：总体正确预测数 ÷ 所有样本数。
    return miou, oa, iou
    # 返回三个指标供主函数使用。


def read_labels(file_path):   
    with open(file_path, 'r') as f:  # 读取 .txt 文件中每行的标签（以整数返回）。
        return [int(float(line.strip())) for line in f]  # 这里做了 float → int 转换，处理部分小数标签格式。


def main(folder_path, output_path):  
    total_cm = np.zeros((6, 6), dtype=int)
# 主函数入口：初始化一个 6x6 的累加型混淆矩阵。
    for file in os.listdir(folder_path):
        if file.endswith("_label.txt"):
        # 遍历所有 *_label.txt 文件（作为真实标签）。
            base_name = file.rsplit("_", 1)[0]
            label_file = os.path.join(folder_path, file)
            pred_file = os.path.join(folder_path, f"{base_name}_pred.txt")
            # 从 label 文件名推导出对应的 pred 文件名。

            true_labels = read_labels(label_file)
            pred_labels = read_labels(pred_file)
            # 读取两组标签（真实 vs 预测）。

            cm = confusion_matrix(true_labels, pred_labels, labels=range(6))
            total_cm += cm
            # 计算当前文件的混淆矩阵，并累计到 total_cm 中。

    intersection = np.diag(total_cm)
    union = np.sum(total_cm, axis=0) + np.sum(total_cm, axis=1) - intersection
    iou = intersection / union.astype(np.float32)
    miou = np.mean(iou)
    oa = np.sum(intersection) / np.sum(total_cm)
    # 所有文件处理完后，整体计算 mIoU、OA、各类 IoU。

    with open(output_path, 'w') as f:
        f.write(f"mIoU: {miou:.4f}\n")
        f.write(f"OA: {oa:.4f}\n")
        for i, class_iou in enumerate(iou):
            f.write(f"IoU for class {i}: {class_iou:.4f}\n")
    # 将结果写入输出文件，格式如下：
    # mIoU: 0.6523
    # OA: 0.8342
    # IoU for class 0: 0.7123
    # IoU for class 1: 0.6540
    # ...

if __name__ == "__main__":
    # Assume the files are in a folder named 'data' and results are saved to 'results.txt'
    main('/home/yuhao/study/code/stratified_transformer/完成_实验/表2实验/pointnet++',
         '/home/yuhao/study/code/stratified_transformer/完成_实验/表2实验/pointnet++/pointnet++_results.txt')

'''

import os
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(true_labels, pred_labels, num_classes=6):
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou = intersection / union.astype(np.float32)
    miou = np.mean(iou)
    oa = np.sum(intersection) / np.sum(cm)
    return miou, oa, iou

def read_labels(file_path):
    with open(file_path, 'r') as f:
        return [int(float(line.strip())) for line in f]

# ✅ 新增函数：抽离主逻辑为一个可调用函数
def calculate_all_metrics(folder_path, output_path, num_classes=6):
    total_cm = np.zeros((num_classes, num_classes), dtype=int)

    for file in os.listdir(folder_path):
        if file.endswith("_label.txt"):
            base_name = file.rsplit("_", 1)[0]
            label_file = os.path.join(folder_path, file)
            pred_file = os.path.join(folder_path, f"{base_name}_pred.txt")

            if not os.path.exists(pred_file):
                print(f" 缺失预测文件：{pred_file}，跳过该对")
                continue

            true_labels = read_labels(label_file)
            pred_labels = read_labels(pred_file)

            cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
            total_cm += cm

    intersection = np.diag(total_cm)
    union = np.sum(total_cm, axis=0) + np.sum(total_cm, axis=1) - intersection
    iou = intersection / union.astype(np.float32)
    miou = np.mean(iou)
    oa = np.sum(intersection) / np.sum(total_cm)

    with open(output_path, 'w') as f:
        f.write(f"mIoU: {miou:.4f}\n")
        f.write(f"OA: {oa:.4f}\n")
        for i, class_iou in enumerate(iou):
            f.write(f"IoU for class {i}: {class_iou:.4f}\n")

    print("\n 指标计算完成：")
    print(f"  - mIoU: {miou:.4f}")
    print(f"  - OA:   {oa:.4f}")
    print(f" 结果已保存到：{output_path}")

# ✅ 替换 main：改为控制台交互版本
def main():
    print("=== 点云 mIoU / OA 评估工具 ===")
    print("_label.txt 是模型预测后存放 真实标签 的文件")
    print("_pred.txt 是模型预测后存放 预测结果 的文件")
    folder_path = input("请输入包含 *_label.txt 和 *_pred.txt 的文件夹路径：\n> ").strip()
    output_path = input("请输入结果保存路径（例如：result.txt）：\n> ").strip()
    num_classes = int(input("请输入类别总数（如 6）：\n> ").strip())

    if not os.path.isdir(folder_path):
        print(" 输入路径无效，目录不存在！")
        return

    calculate_all_metrics(folder_path, output_path, num_classes)

if __name__ == "__main__":
    main()
