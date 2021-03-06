#### 量化训练:
1. post-training quantization: 剪枝，稀疏编码，对模型存储体积进行压缩
2. quatization-aware training: forward F32==>int8 映射，backward F32梯度更新，保存模型int8，quantize/dequantize
3. 还有一种训练和推理都用int8
4. 在训练过程中引入精度带来的误差，然后整个网络在训练过程中进行修正

模型大小不仅是内存容量问题，也是内存带宽问题
量化就是将神经网络的浮点算法转化为定点

花哨的研究往往是过于棘手或前提假设过强，以至于几乎无法引入工业界的软件栈
Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference揭示了量化感知训练的诸多细节
为什么量化是有效的(具有足够好的预测准确度)，尤其是将FP32转换为INT8时已经丢失了信息？直觉解释是**神经网络被过度参数化**，进而包含足够的冗余信息，裁剪这些冗余信息不会导致明显的准确度下降。相关证据表明对于给定的量化方法，FP32网络和INT8网络之间的准确度差距对于大型网络来说较小，因为大型网络过度参数化的程度更高

#### 可能有用的github上的一些东西:
1. [Graph Transform Tool(tensorflow)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md)
2. [Converter command line examples(tensorflow)](https://www.tensorflow.org/lite/convert/cmdline_examples)

#### Tensorflow官网的一些东西:
1. [TensorFlow Lite and TensorFlow operator compatibility](https://www.tensorflow.org/lite/guide/ops_compatibility)

#### 目前工作进展: 
1. 把网络换成VGG，人脸表情识别(传统方法)，要实现的是用CNN来实现检测的量化训练
2. 第一要务是跑通检测模型，后面要检测下模型大小的问题(可以运行下tf-slim来对比下)

#### 量化训练技巧: 
1. 在你定义好网路结构之后，加上下面这句话，即可量化训练: tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=200)
2. 论文中提到，为了使量化训练有更好的精度，推荐使用relu6，让输出限制在较小的范围内
3. tf.contrib.quantize.create_eval_graph()和tf.contrib.quantize.create_training_graph()不能同时出现在同一程序中，不然会出问题
4. 基于已经训好的网络去做模拟量化实验的，不基于预训练模型训不起来，可能还有坑要踩，而且在模拟量化训练过程中bn层参数固定，融合bn参数也是用已经训练好的移动均值和方差，而不是用每个batch的均值和方差
5. 重写后的 eval 图与训练图并非平凡地等价，这是因为量化操作会影响 batchnorm 这一步骤
6. 对于卷积层之后带batchnorm的网络，因为一般在实际使用阶段，为了优化速度，batchnorm的参数都会提前融合进卷积层的参数中，所以训练模拟量化的过程也要按照这个流程．首先把batchnorm的参数与卷积层的参数融合，然后再对这个参数做量化
7. 对于权值的量化分通道进行求缩放因子，然后对于激活值的量化整体求一个缩放因子，这样的效果最好

#### 模型checkpoint相关:
1. 如果不知道输入输出，可采用如下方式查看.pb(图模型):
```
import tensorflow as tf
gf = tf.GraphDef()
gf.ParseFromString(open('./testpb/test.pb','rb').read())
for n in gf.node:
    print ( n.name +' ===> '+n.op )  
```
2. 保存图模型，不含有权重信息:
```
g = tf.get_default_graph()
graph_def = g.as_graph_def()
tf.train.write_graph(graph_def, "./model", 'graph.pb', as_text=False)
```
3. 保存训练期间的权重信息:
```
saver = tf.train.Saver()
saver.save(sess, os.path.join( ‘./model’, 'model.ckpt'))
```
#### 理论文章:
1. [tensorflow的量化教程(2)(csdn)](https://blog.csdn.net/u012101561/article/details/86321621)
2. [卷积神经网络训练模拟量化实践(oschina)](https://my.oschina.net/Ldpe2G/blog/3000810)
3. [神经网络量化简介(黎明灰烬)](https://jackwish.net/neural-network-quantization-introduction-chn.html)

#### 实践文章:
1. 用于coral TPU的预测     [Object detection and image classification with Google Coral USB Accelerator(pyImageSearch)](https://www.pyimagesearch.com/2019/05/13/object-detection-and-image-classification-with-google-coral-usb-accelerator/)
2. 基于tfslim的方式量化训练    [Quantizing neural networks to 8-bit using TensorFlow(armDevelop)](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/quantizing-neural-networks-to-8-bit-using-tensorflow)
3. [【Tensorflow系列】使用Inception_resnet_v2训练自己的数据集并用Tensorboard监控(cnblogs)](https://www.cnblogs.com/andre-ma/p/8458172.html)
