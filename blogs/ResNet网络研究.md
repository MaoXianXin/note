# ResNet网络研究
skip connection: 解决网络加深，网络性能得不到提升的问题

VGG16
VGG19                网络不能够盲目的加深，否则会增大计算量和模型体积
ResNet50

这时候就要从网络结构设计做文章

ResNet18   BasicBlock
ResNet50   Bottleneck          conv1x1        channel大    通道减小
                                              conv3x3        channel小
                                              conv1x1        channel大    通道放大
                                                           减少参数量和计算量

模型设计从accuract   model size   MAC(latency)三个角度考虑