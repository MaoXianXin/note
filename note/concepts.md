# concepts
## FLOPS
floating point operations per second
用于衡量计算机性能的一个指标，同时也是一个衡量模型在推理的时候所需要的计算量

TeraFlops   每秒1万亿次浮点运算
PetaFlops   每秒1千万亿次浮点运算

![test2020122415:21:15](https://maoxianxin1996.oss-accelerate.aliyuncs.com/ai/test2020122415:21:15.png)

T4 gpu
上图是T4显卡，FP32(一般是训练阶段)=8.1 TeraFlops，INT8(推理阶段)=130 TeraFlops

![test2020122415:21:27](https://maoxianxin1996.oss-accelerate.aliyuncs.com/ai/test2020122415:21:27.png)

V100多了一个东西 Tensor Performance，利用了tensor core，大大的加速了计算

模型实际速度还会受限于上图中出现的memory bandwidth，也就是说模型实际推理速度受Flops影响，也会受访存带宽等影响

然后这里的GPU内存也存在区别，T4是GDDR6，v100是HBM2(Hign Bandwidth Memory)


![test2020122417:56:08](https://maoxianxin1996.oss-accelerate.aliyuncs.com/ai/test2020122417:56:08.png)

## FLOPs

![11](https://maoxianxin1996.oss-accelerate.aliyuncs.com/ai/testWeChat%20Image_20201225153249.jpg)
上面这张图的features[2]的MAC占比为11.9%，但是in_channel=out_channel=64，和后面的512相比还是比较小的，这里计算量大的原因是output feature map较大
features[7]，features[12]，features[14]，features[19]，features[21]的in_channel=out_channel，同时MAC占都是11.9%左右，比后面的features[24]，features[26]，features[28]的MAC=2.98%大，但是后者的通道数为512是最大的，造成这个的原因是在卷积过程中maxpooling做了降维处理，使output feature map不断的变小
还有一个点是，全连接层的计算量看起来其实不大，但是参数量特别大，所以最终的model size也很大
2*MAC=FLOPs，两倍的说法是因为在编码的时候。芯片内部会用mmac指令进行乘加计算。对应一个乘法+一个加法

## Model Performance

![12](https://maoxianxin1996.oss-accelerate.aliyuncs.com/ai/testWeChat%20Image_20201225153054.png)