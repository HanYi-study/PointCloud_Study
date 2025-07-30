import numpy as np

'''
作用：将点云数据（TXT格式）转换为 PLY 格式的带颜色+标签的点云文件

输入： .txt 点云数据，包含 xyz + RGB + label 共7列
输出： .ply 格式的点云文件，含 RGB 颜色和标签信息
处理： 拆分并组合坐标、颜色、标签
应用： 适用于将分类后的 txt 点云文件转换为可视化友好的 .ply 格式

输入：
该文件内容格式如下（每行是一个点）：
x, y, z, r, g, b, label
12.3, 7.8, 0.5, 255, 0, 0, 2

输出：
12.300000 7.800000 0.500000 255 0 0 2
并在文件开头加上 .ply 文件的头部信息（Header）：
ply
format binary_little_endian 1.0
element vertex 43867
property float64 x
property float64 y
property float64 z
property uint16 red
property uint16 green
property uint16 blue
property uint8 label
end_header
...

eg:
输入：（.txt文件）
1.0, 2.0, 3.0, 255, 0, 0, 1
4.0, 5.0, 6.0, 0, 255, 0, 2
7.0, 8.0, 9.0, 0, 0, 255, 3
输出：（.ply文件）
ply
format binary_little_endian 1.0
element vertex 3
property float64 x
property float64 y
property float64 z
property uint16 red
property uint16 green
property uint16 blue
property uint8 label
end_header
1.000000 2.000000 3.000000 255 0 0 1
4.000000 5.000000 6.000000 0 255 0 2
7.000000 8.000000 9.000000 0 0 255 3


'''


# Function to create point cloud file
def create_output(vertices, rgb, labels, filename):
#将传入的坐标、颜色和标签合并，保存为 .ply 文件。
#写入的是ASCII格式点数据（用 np.savetxt()）

    # colors = colors.reshape(-1, 3)
    # vertices = np.hstack([vertices.reshape(-1, 3), labels])
    vertices = np.concatenate((vertices, rgb, labels[:, None]), axis=1)
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    #合并为一行七列：x y z R G B label
    ply_header = '''ply
format binary_little_endian 1.0
element vertex %(vert_num)d
property float64 x
property float64 y
property float64 z
property uint16 red
property uint16 green
property uint16 blue
property uint8 label
end_header
'''
    #PLY文件的头部信息，描述了点的数量、每个字段的类型

    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)
    #将刚刚写入的数据前面插入PLY头部信息


if __name__ == '__main__':
    # Define name for output file
    output_file = '/home/yunl/Data/paper_data/fps/results/135-136.ply' # 输出的文件的详细路径
    a = np.loadtxt('/home/yunl/Data/paper_data/fps/132-133(132_133).txt', skiprows=1,delimiter=',', usecols=(0, 1, 2), unpack=False)  #坐标
    b = np.float64(a)  # 转为 float64 精度
    e = np.loadtxt('/home/yunl/Data/paper_data/fps/132-133(132_133).txt', skiprows=1, dtype='uint16',delimiter=',', usecols=(3, 4, 5), unpack=False)  #RGB 颜色（uint16）
    c = np.loadtxt('/home/yunl/Data/paper_data/fps/132-133(132_133).txt', skiprows=1, dtype='uint8',delimiter=',', usecols=(6), unpack=False)  #
    #这些步骤从同一个 .txt 文件中读取出：  b：x, y, z 坐标 (float64)   e：RGB颜色 (uint16)    c：标签 (uint8)

    # d = np.int32(c)
#   43867是我的点云的数量，用的时候记得改成自己的
#     one = np.ones((43867,3))
#     one = np.float32(one)*255
#    points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
#    colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    # Generate point cloud
    print("\n Creating the output file... \n")
#    create_output(points_3D, colors, output_file)
    create_output(b, e, c, output_file)   #生成 .ply 点云文件。
