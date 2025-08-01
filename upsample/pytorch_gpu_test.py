import time
import torch

if __name__ == '__main__':
    print('torch版本：'+torch.__version__)
    print('cuda是否可用：'+str(torch.cuda.is_available()))
    print('cuda版本：'+str(torch.version.cuda))
    print('cuda数量:'+str(torch.cuda.device_count()))
    print('GPU名称：'+str(torch.cuda.get_device_name()))
    print('当前设备索引：'+str(torch.cuda.current_device()))

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    print(torch.rand(3, 3).cuda())

    for i in range(1,100000):
        start = time.time()
        a = torch.randn(i*100, 100, 100)
        a = a.cuda() #a = a
        a = torch.matmul(a,a)
        end = time.time() - start
        # print(end)
        print(i)

'''
主要功能：
  检查 PyTorch 的 GPU 使用环境是否可用，包括：
  当前安装的 PyTorch 版本；
  是否检测到 CUDA；
  CUDA 的版本；
  GPU 数量；
  GPU 名称；
  当前使用的 GPU 索引。

执行一次简单的 GPU 张量计算示例：
  创建一个随机的 3×3 张量并将其移动到 GPU 上；
  打印这个张量（验证 GPU 是否在使用）。

进行性能压力测试（循环计算张量乘法）：
  从小规模张量开始，每次扩大张量数量（i*100 × 100 × 100）；
  将张量送入 GPU 上计算矩阵乘法；
  统计每次的运行时间；
  打印当前的循环次数 i（运行时间 end 被注释掉了，但可以用来分析耗时）。

'''