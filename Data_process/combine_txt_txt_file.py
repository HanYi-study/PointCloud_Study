import os

'''
该脚本用于将两个不同目录下的同名 .txt 文件合并，并将合并后的结果保存到第三个目录中。
具体操作为：
    1.遍历 folder_a 和 folder_b 中的文件；
    2.找出两者文件名相同的 .txt 文件；
    3.将两个文件的内容顺序拼接起来（中间加一个换行符）；
    4.将合并后的结果保存到 folder_c。

'''

def merge_files(folder_a, folder_b, folder_c):
      

    # 获取文件夹a和b中的文件列表
    files_in_a = set(os.listdir(folder_a))    #第一个 .txt 文件目录（上半段内容）
    files_in_b = set(os.listdir(folder_b))    #第二个 .txt 文件目录（下半段内容）

    # 查找两个文件夹中名称相同的文件
    common_files = files_in_a.intersection(files_in_b)   #获取两个目录下的所有文件名，转为集合后求交集 → 得到文件名相同的 .txt 文件。

    for file_name in common_files:  #对每一个同名文件

        path_a = os.path.join(folder_a, file_name)  
        path_b = os.path.join(folder_b, file_name)    #打开 a 和 b 目录中的原文件；
        path_c = os.path.join(folder_c, file_name)    

        # 合并文件并保存到文件夹c
        with open(path_a, 'r') as file_a, open(path_b, 'r') as file_b, open(path_c, 'w') as file_c:
            file_c.write(file_a.read())
            file_c.write("\n")
            file_c.write(file_b.read())
            #把 a 中的内容写入新文件，中间插入一个换行，再写入 b 中的内容；
            #输出文件保存到 folder_c 中，保持原文件名不变。

        '''
           每个合并后的txt文件:
            <内容来自folder_a中该txt文件>
                         #中间有一个换行
            <内容来自folder_b中该txt文件>
        '''
    
        print(f"Merged {file_name} and saved to folder c.")

# 使用方法：
folder_a = '因为合并的是整个a文件夹下所有的txt文件，所以需要输入a文件夹的目录路径' # 替换为你的实际路径
folder_b = '因为合并的是整个b文件夹下所有的txt文件，所以需要输入b文件夹的目录路径'
folder_c = '输出目录的路径'
merge_files(folder_a, folder_b, folder_c)