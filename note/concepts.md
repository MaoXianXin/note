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