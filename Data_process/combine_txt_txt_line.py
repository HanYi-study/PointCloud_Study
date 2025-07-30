import os

'''
该函数 merge_files(a_path, b_path, c_path) 的作用是将两个逐行对应的 .txt 文件合并，将它们每一行拼接后写入一个新的文件中。

合并方式：
    对文件 A 和 B 进行 逐行 zip，即第 i 行和第 i 行配对；
    每对行拼接成：a_line + ' ' + b_line + '\n'
    写入目标文件C

当前文件：
    将两个文件的内容逐行合并
另一个文件：
    同名文件进行文件的和ing，不是逐行合并

| 方面         | 上一版本（遍历多个文件）                     | 当前版本（合并单个文件）            |
| ----------  | ------------------------------------------ | --------------------------------- |
| 输入对象     | `folder_a`、`folder_b`，自动查找所有同名文件 | 只针对**单个文件路径**进行合并      |
| 合并模式     | file\_a 内容 + 换行 + file\_b 内容          | **逐行 zip** 拼接（逐行内容合并）   |
| 输出路径处理 | `folder_c` 中保留原始文件名                  | 需要手动指定输出文件完整路径         |
| 场景         | 批量合并目录下的所有同名文件                 | 只想快速合并两个单文件的行（逐行对齐）|

'''


def merge_files(a_path, b_path, c_path):
    with open(a_path, 'r') as a_file, open(b_path, 'r') as b_file, open(c_path, 'w') as c_file:  #将两个逐行对应的 .txt 文件合并，将它们每一行拼接后写入一个新的文件中。
        for a_line, b_line in zip(a_file, b_file):
            a_line = a_line.rstrip('\n')  
            b_line = b_line.rstrip('\n')  #对文件 A 和 B 进行 逐行 zip，即第 i 行和第 i 行配对；
            c_line = a_line + ' ' + b_line + '\n'  #每对行拼接成：a_line + ' ' + b_line + '\n'   
            c_file.write(c_line)

# 使用函数
merge_files('输入的是txt文件,a文件的路径',
            '输入的是txt文件,b文件的路径',
            '输出的是txt文件，c文件的路径')
