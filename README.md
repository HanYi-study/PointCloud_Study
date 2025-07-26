# 点云代码学习
此文档用于记录点云代码学习笔记

## 目录
 - 文件功能说明
 - 文件结构说明
 - 处理流程/逻辑展示



 
## 文件功能说明  
 **1. data-prepare文件夹:**  
 （1）step0_read_las.py  
 （2）step1_data_prepare_H3D.py  
 （3）step1_data_prepare_V3D.py  
 （4）step2_data_process_randla.py  
 （5）help_tool.py  
 （6）help_ply.py
 
 - step0_read_las.py：  
 输入：原始.las 格式点云数据（含空间坐标、颜色、分类标签等）   
 输出：内存中的numpy数组（x,y,z,RGB,label）、min_xyz起始坐标（后续可用于还原位置信息）  
数据状态：原始格式转换为标准数组  
 处理：  
    - a）解析.las文件结构      
    - b）提取xyz坐标（归一化到起点）、RGB颜色、标签（classification）

 - step1_data_prepare_H3D.py / step1_data_prepare_V3D.py：  
 两文件的区别：  
 H3D.py 处理的是 H3D 格式的点云数据（含 RGB、instance、label、可能还带 intensity）  
 V3D.py 针对的是 V3D 格式的点云数据（结构简洁，可能只有 XYZ 和 label）  
 输入：step0产出的数组数据  
 输出：.ply格式的点云文件、KD-Tree或其他中间结构缓存（如 .npy/.pkl 文件）  
 数据状态：已变为可视化或深度学习框架可读格式  
 操作：  
    - a）构建KD-Tree空间结构  
    - b）写出.ply文件：XYZ+RGB+Label  

 - step2_data_process_randla.py:  
 输入：.ply格式的点云文件  
 输出：是保存在output_dir目录下的两个文件，一个是处理后的点云特征数据文件，另一个是标签文件。  
 a）处理后的特征数据文件(同名.txt文件),把原数据中除了最后一列(label)以外的列都提取出来，在第4列插入值0，用于之后的模型输入、特征工程、数据增强等。  
 b）标签文件(同名.label文件),只保存每一个点的标签（最后一列0,通常用于分类监督学习模型的目标值。  
 数据状态：结构化、标准化、块化后的训练输入数据。  
 操作：  
    - a）对每个点云文件提取标签（最后一列）。  
    - b）修改点云特征（在每行的第四列插入0）。  
    - c）将处理后的数据和标签分别保存到输出目录中。





 ## 文件结构说明  


 ## 处理流程/逻辑展示
 
