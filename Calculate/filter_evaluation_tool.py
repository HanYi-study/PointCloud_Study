''' #----------------------------------------------------------------------------版本1.0

# Given values
On = 1325642
Ot = 67403
Rn = 1324767
Rt = 4660

# Calculate removed points
Dn = On - Rn
Dt = Ot - Rt

# Calculate intersection over union for non-noise points (IOU0)
IOU0 = Rn / (Rn + Dn + Rt)

# Calculate intersection over union for noise points (IOU1)
IOU1 = Dt / (Dt + Dn + Rt)

# Calculate noise error (Ea)
Ea = Rt / Ot

# Calculate total error (Eb)
Eb = (Dn + Rt) / (On + Ot)

# Calculate retained target point ratio (I)
I = Rn / On

# Calculate noise removal ratio (Qn)
Qn = Dt / Ot

# Calculate accuracy (Acc)
Acc = (Rn + Dt) / (On + Ot)

# Print calculated values
print("Non-noise Points Removed (Dn):", Dn)
print("Noise Points Removed (Dt):", Dt)
print("IOU for Non-noise Points (IOU0):", IOU0)
print("IOU for Noise Points (IOU1):", IOU1)
print("Noise Removal Ratio (Qn):", Qn)
print("Accuracy (Acc):", Acc)
print("Total Error (Et):", Eb)

# print("Noise Error (Ea):", Ea)
# print("Retained Target Point Ratio (I):", I)

'''

import os

def calculate_filter_metrics(On, Ot, Rn, Rt):
    Dn = On - Rn
    Dt = Ot - Rt
    IOU0 = Rn / (Rn + Dn + Rt)
    IOU1 = Dt / (Dt + Dn + Rt)
    Ea = Rt / Ot if Ot > 0 else 0
    Eb = (Dn + Rt) / (On + Ot)
    I = Rn / On if On > 0 else 0
    Qn = Dt / Ot if Ot > 0 else 0
    Acc = (Rn + Dt) / (On + Ot)

    result = (
        f">>> 输入参数:\n"
        f"原始非噪声点 On: {On}\n"
        f"原始噪声点   Ot: {Ot}\n"
        f"保留非噪声点 Rn: {Rn}\n"
        f"保留噪声点   Rt: {Rt}\n\n"
        f">>> 评估指标:\n"
        f"Non-noise Points Removed (Dn): {Dn}\n"
        f"Noise Points Removed (Dt): {Dt}\n"
        f"IOU for Non-noise Points (IOU0): {IOU0:.4f}\n"
        f"IOU for Noise Points (IOU1): {IOU1:.4f}\n"
        f"Noise Removal Ratio (Qn): {Qn:.4f}\n"
        f"Accuracy (Acc): {Acc:.4f}\n"
        f"Total Error (Eb): {Eb:.4f}\n"
    )

    return result


def main():
    print("=== 点云滤波性能评估工具 ===")
    print("IOU0（非噪声 IOU）:衡量真正保留的目标点与误删点、误保噪声的比值")
    print("IOU1（噪声 IOU）:衡量正确去除的噪声点与误删点、误保点的比值")
    print("Qn（噪声去除率）:成功剔除的噪声点占全部噪声点的比例")
    print("Acc（总体准确率）:正确操作的点（保留/删除）占总点数的比例")
    print("Eb（误差率）:误处理的点占总点数的比例")
    print("Dn/Dt:被误删/正确删的点数量，作为算法代价衡量\n")

    try:
        On = int(input("请输入原始非噪声点数量 On："))
        Ot = int(input("请输入原始噪声点数量 Ot："))
        Rn = int(input("请输入滤波后保留的非噪声点数量 Rn："))
        Rt = int(input("请输入滤波后保留的噪声点数量 Rt："))
    except ValueError:
        print(">>> 输入无效，必须是整数！")
        return

    result_str = calculate_filter_metrics(On, Ot, Rn, Rt)
    print("\n" + result_str)

    save_choice = input("是否保存结果？1 - 不保存，2 - 保存\n> ").strip()
    if save_choice == '2':
        save_path = input("请输入保存结果的文件路径（如 result.txt）：\n> ").strip()
        try:
            with open(save_path, 'w') as f:
                f.write(result_str)
            print(f">>> 结果已保存至 {save_path}")
        except Exception as e:
            print(f">>> 保存失败：{e}")
    else:
        print(">>> 已完成评估。未保存结果。")

if __name__ == "__main__":
    main()

'''
示例输出：
=== 点云滤波性能评估工具 ===
请输入原始非噪声点数量 On：10000
请输入原始噪声点数量 Ot：3000
请输入滤波后保留的非噪声点数量 Rn：9800
请输入滤波后保留的噪声点数量 Rt：200

>>> 输入参数:
原始非噪声点 On: 10000
原始噪声点   Ot: 3000
保留非噪声点 Rn: 9800
保留噪声点   Rt: 200

>>> 评估指标:
Non-noise Points Removed (Dn): 200
Noise Points Removed (Dt): 2800
IOU for Non-noise Points (IOU0): 0.9510
IOU for Noise Points (IOU1): 0.9020
Noise Removal Ratio (Qn): 0.9333
Accuracy (Acc): 0.9429
Total Error (Eb): 0.0571

是否保存结果？1 - 不保存，2 - 保存
> 2
请输入保存结果的文件路径（如 result.txt）：
> ./filter_eval_0802.txt
>>> 结果已保存至 ./filter_eval_0802.txt
'''
