import laspy
import numpy as np

save_las_file = r"/home/yunl/Data/paper_data/save_las.las"

# 以x y z r g b的方式定义点云数据
my_data = np.array([[8, 5, 3, 255, 0, 0],
                    [9, 0, 1, 255, 0, 0],
                    [2, 5, 3, 0, 255, 0],
                    [0, 4, 2, 0, 255, 0],
                    [7, 2, 9, 0, 0, 255],
                    [8, 8, 4, 0, 0, 255],
                    [9, 5, 8, 255, 0, 255],
                    [2, 5, 9, 255, 0, 255],
                    [0, 7, 5, 255, 255, 0],
                    [11, 2, 8, 255, 255, 0],
                    [10, 9, 0, 255, 255, 0]
                    ])

# 创建点云文件
las = laspy.create(file_version="1.2", points_format=3)
las.x = my_data[:, 0]
las.y = my_data[:, 1]
las.z = my_data[:, 2]
las.red = my_data[:, 3]
las.green = my_data[:, 4]
las.blue = my_data[:, 5]

# 保存las文件
las.write(save_las_file)
