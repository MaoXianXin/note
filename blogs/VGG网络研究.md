# VGG网络研究
## Very Deep Convolutional Networks for Large-Scale Image Recognition
In this work we investigate the effect of the convolutional network **depth on its accuracy** in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of **increasing depth using an architecture with very small (3x3) convolution filters**, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to **16-19 weight layers**. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our **representations generalise well** to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of **deep visual representations** in computer vision

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/ai1/20201231112511.png)
从上图可以看出通道数翻倍发生在max pooling之后，经过pooling之后output feature map会变小，从而降低网络的计算量，此时我们增加通道数，虽然会增大计算量，但是也会让网络提取到更多样性的特征

## 研究点
1. depth  16到19   accuracy的变化   flower_photos
### depth
2. convolution filters     3x3   5x5   7x7    params和FLOPs的变化
### filters
3. deep visual representations 可视化
### representations
4. AutoML 对比 accuracy
### AutoML
5. image resolution 32x32  64x64  128x128  224x224  256x256   accuracy
### resolution
6. model optimzation accuracy相同  优化latency和model size
### optimization
7. multi-scale training 和data augmentation里面的zoom是否存在关系
### multi-scale
8. conv3x3 对比 conv1x1
### filter size accuracy
