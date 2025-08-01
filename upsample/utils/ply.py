import numpy as np


def read_ply(filepath):
    """简化版 PLY 读取函数，仅用于 ascii 格式 .ply 文件"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 找 header 结束位置
    i = 0
    while not lines[i].strip().startswith("end_header"):
        i += 1
    start = i + 1

    # 读取数据字段名
    header_lines = lines[:start]
    props = []
    for line in header_lines:
        if line.startswith('property'):
            props.append(line.strip().split()[-1])

    # 读取点数据
    data = np.loadtxt(lines[start:], dtype=np.float64)
    data_dict = {}
    for idx, key in enumerate(props):
        data_dict[key] = data[:, idx]

    # 转为结构化数组
    dtype = [(k, data_dict[k].dtype) for k in data_dict]
    structured = np.zeros(len(data_dict[props[0]]), dtype=dtype)
    for k in data_dict:
        structured[k] = data_dict[k]

    return structured
