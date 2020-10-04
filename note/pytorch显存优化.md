# 第一步-搭建pytorch分类网络
[网络如何搭建参考来源](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
## 自定义网络

## 进行迁移学习

## 关于分类网络训练和推理的关键点
训练的时候,需要做数据预处理,把像素值缩放到[-1,1],对数据进行shuffle和batch
推理的时候,需要做数据预处理,把像素值缩放到[-1,1],对数据进行batch
上面两步属于data pipeline阶段,我们可以朝多线程方向优化速度

接下来是网络结构定义,以及损失函数和优化器定义

然后是训练,这里要注意一个点,训练的时候我们一般采用GPU,所以需要把数据和网络都传到GPU设备上
在推理的时候,我们可以把数据和网络都放在CPU上,就是速度会慢些

# 第二步-跟踪模型训练过程中显存利用变化
三篇参考文章:
[浅谈深度学习:如何计算模型以及中间变量的显存占用大小](https://oldpan.me/archives/how-to-calculate-gpu-memory)
[如何在Pytorch中精细化利用显存](https://oldpan.me/archives/how-to-use-memory-pytorch)
[再次浅谈Pytorch中的显存利用问题(附完善显存跟踪代码)](https://oldpan.me/archives/pytorch-gpu-memory-usage-track)
卷积核参数,output_feature_map,forward和backward过程中产生的中间变量,优化器,主要是这几个点占用显存

# 第三步-优化显存利用
占用显存比较多空间的并不是我们输入图像,而是神经网络中的中间变量以及使用optimizer算法时产生的巨量的中间参数
中间变量在backward的时候会翻倍

比如及时清空中间变量,优化代码,减少batch

消费级显卡对单精度计算有优化,服务器级别显卡对双精度计算有优化

比较需要注意的点是模型准确率不能降太多

CPU调用和GPU调用优化问题