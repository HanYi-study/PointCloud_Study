import os
import numpy as np
import matplotlib.pyplot as plt

'''
混淆矩阵（Confusion Matrix）是用于评估分类模型性能的矩阵，反映模型预测标签与真实标签的对应关系。
矩阵结构中：行（Y轴）代表真实标签（ground truth） / 列（X轴）代表模型预测标签（prediction）
         预测类别
        A   B   C
真实 A [20,  3,  2]
     B [ 4, 30,  1]
     C [ 1,  2, 25]
[0][0]=20：表示真实为 A 且预测为 A 的样本数（真正类 True Positive）
[0][1]=3：真实为 A 但预测为 B 的样本数（误分类）
[1][0]=4：真实为 B 但预测为 A 的样本数
每一个数值： 表示某一类真实标签被预测为另一类的样本数，是对分类正确性和混淆程度的直观刻画。

py文件的作用：
该脚本是一个“混淆矩阵可视化工具”，其核心功能是将已有的混淆矩阵数据（.txt 或 .csv）读取出来
绘制为彩色图像（heatmap），可视化模型在每一类上的分类准确性与混淆情况。

混淆矩阵如何生成的？
混淆矩阵通常是模型评估时生成的
例如使用 sklearn 的 confusion_matrix(y_true, y_pred) / 或自己构建二维数组 [真实标签][预测标签] += 1

'''


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DrawConfusionMatrix:
    def __init__(self, matrix, labels_name, x_labels_name, num_label, cmap='viridis', show_colorbar=True):
        self.matrix = matrix
        self.labels_name = labels_name
        self.x_labels_name = x_labels_name
        self.num_label = num_label
        self.cmap = cmap
        self.show_colorbar = show_colorbar

    def draw(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(self.matrix, cmap=self.cmap)

        if self.show_colorbar:
            cbar = plt.colorbar(cax)
            cbar.ax.tick_params(labelsize=12)

        ax.set_xticks(np.arange(self.num_label))
        ax.set_yticks(np.arange(len(self.labels_name)))
        ax.set_xticklabels(self.x_labels_name, fontsize=12)
        ax.set_yticklabels(self.labels_name, fontsize=12)

        for i in range(len(self.labels_name)):
            for j in range(self.num_label):
                ax.text(j, i, f"{self.matrix[i][j]:.2f}", ha='center', va='center',
                        fontsize=11, color='black')

        plt.xlabel('预测类别', fontsize=14)
        plt.ylabel('真实类别', fontsize=14)
        plt.title('混淆矩阵图', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ 混淆矩阵图已保存到：{save_path}")
        else:
            plt.show()


def load_matrix(file_path):
    try:
        return np.loadtxt(file_path, delimiter=',')
    except:
        return np.loadtxt(file_path)


def get_manual_matrix():
    print("\n请输入混淆矩阵（每行输入一行数字，空行结束）：")
    matrix_lines = []
    while True:
        line = input()
        if line.strip() == '':
            break
        try:
            row = list(map(float, line.strip().split()))
            matrix_lines.append(row)
        except ValueError:
            print("⚠️ 输入格式错误，请输入空格分隔的数字。")
    return np.array(matrix_lines)


def prompt_labels(num_classes, default_labels):
    print(f"\n🧠 检测到混淆矩阵包含 {num_classes} 个真实类别标签，以及 OA / IoU。")
    print(f"是否使用默认类别标签（如：{default_labels}）？输入 y 使用默认，输入 n 手动输入：")
    label_input = input("> ").strip().lower()
    if label_input == 'n':
        labels_name = []
        for i in range(num_classes):
            label = input(f"请输入类别 {i} 的标签名称：\n> ").strip()
            labels_name.append(label)
    else:
        labels_name = default_labels[:num_classes]
    return labels_name

def main():
    print("=== 混淆矩阵图像生成工具 ===")
    mode = input("请选择数据输入方式：\n1 - 读取文件或文件夹\n2 - 手动输入矩阵\n> ").strip()

    if mode == '1':
        input_path = input("请输入待处理文件或文件夹路径：\n> ").strip()
        use_manual_matrix = False
    elif mode == '2':
        matrix = get_manual_matrix()
        use_manual_matrix = True
        input_path = None
    else:
        print("❌ 输入无效，退出程序。")
        return

    output_dir = input("请输入处理后图像保存的目录路径：\n> ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    default_labels = ['聚集人群', '灌木', '建筑物', '地面', '树木', '车辆']

    # 自动识别类别数量
    if use_manual_matrix:
        matrix_shape = matrix.shape
    else:
        # 先读取一个矩阵用于识别类别数量
        if os.path.isdir(input_path):
            files = [f for f in os.listdir(input_path) if f.endswith('.txt') or f.endswith('.csv')]
            if not files:
                print("❌ 未找到有效的混淆矩阵文件。")
                return
            matrix_shape = load_matrix(os.path.join(input_path, files[0])).shape
        else:
            matrix_shape = load_matrix(input_path).shape

    num_classes = matrix_shape[0]
    num_label = matrix_shape[1]
    # OA/IoU列通常在最后两列
    if num_label > num_classes:
        oa_iou_count = num_label - num_classes
    else:
        oa_iou_count = 0

    labels_name = prompt_labels(num_classes, default_labels)
    x_labels_name = labels_name + ['OA', 'IoU'][:oa_iou_count]
    num_label = len(x_labels_name)

    cmap = input("请输入 colormap（默认 viridis，可选 inferno, coolwarm, plasma 等）\n> ").strip()
    if cmap == '':
        cmap = 'viridis'

    colorbar_input = input("是否显示颜色条？[y/n] 默认 y：\n> ").strip().lower()
    show_colorbar = colorbar_input != 'n'

    if use_manual_matrix:
        dcm = DrawConfusionMatrix(matrix, labels_name, x_labels_name, num_label, cmap, show_colorbar)
        save_name = input("请输入保存文件名（不含扩展名）：\n> ").strip()
        out_path = os.path.join(output_dir, save_name + '_matrix.png')
        dcm.draw(save_path=out_path)
    else:
        if os.path.isdir(input_path):
            files = [f for f in os.listdir(input_path) if f.endswith('.txt') or f.endswith('.csv')]
            for fname in files:
                fpath = os.path.join(input_path, fname)
                matrix = load_matrix(fpath)
                dcm = DrawConfusionMatrix(matrix, labels_name, x_labels_name, num_label, cmap, show_colorbar)

                base = os.path.splitext(fname)[0]
                out_path = os.path.join(output_dir, base + '_matrix.png')
                dcm.draw(save_path=out_path)
        else:
            matrix = load_matrix(input_path)
            dcm = DrawConfusionMatrix(matrix, labels_name, x_labels_name, num_label, cmap, show_colorbar)

            base = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(output_dir, base + '_matrix.png')
            dcm.draw(save_path=out_path)

    print("\n🎉 所有混淆矩阵处理完成！")

if __name__
