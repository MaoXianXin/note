# subclass_model
```
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
mnist = tf.keras.datasets.mnist
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCH = 5
for epoch in range(EPOCH):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
```

## experiments
### mnist
```
Epoch 63, Loss: 3.4395288395216994e-08, Accuracy: 100.0, Test Loss: 0.19204725325107574, Test Accuracy: 98.54999542236328
Epoch 64, Loss: 2.225022299739976e-08, Accuracy: 100.0, Test Loss: 0.19182617962360382, Test Accuracy: 98.55999755859375
Epoch 65, Loss: 1.4092463196391236e-08, Accuracy: 100.0, Test Loss: 0.19167029857635498, Test Accuracy: 98.56999969482422
Epoch 66, Loss: 8.829417197375733e-09, Accuracy: 100.0, Test Loss: 0.19166195392608643, Test Accuracy: 98.56999969482422
Epoch 67, Loss: 5.545214065705295e-09, Accuracy: 100.0, Test Loss: 0.19192759692668915, Test Accuracy: 98.58000183105469
Epoch 68, Loss: 3.451107621899041e-09, Accuracy: 100.0, Test Loss: 0.19261902570724487, Test Accuracy: 98.5999984741211
Epoch 69, Loss: 2.1080170942155974e-09, Accuracy: 100.0, Test Loss: 0.19399333000183105, Test Accuracy: 98.58999633789062
Epoch 70, Loss: 1.2675920313398592e-09, Accuracy: 100.0, Test Loss: 0.19623185694217682, Test Accuracy: 98.61000061035156
```
### cifar10
```
Epoch 64, Loss: 0.037418823689222336, Accuracy: 98.86000061035156, Test Loss: 5.988936901092529, Test Accuracy: 55.61000061035156
Epoch 65, Loss: 0.0335138663649559, Accuracy: 98.9520034790039, Test Loss: 6.026555061340332, Test Accuracy: 55.23999786376953
Epoch 66, Loss: 0.03817056491971016, Accuracy: 98.8239974975586, Test Loss: 6.112489223480225, Test Accuracy: 55.089996337890625
Epoch 67, Loss: 0.03954439237713814, Accuracy: 98.77999877929688, Test Loss: 6.182155132293701, Test Accuracy: 54.77000045776367
Epoch 68, Loss: 0.041161421686410904, Accuracy: 98.81399536132812, Test Loss: 6.382829666137695, Test Accuracy: 55.72999572753906
Epoch 69, Loss: 0.0326593741774559, Accuracy: 98.99800109863281, Test Loss: 6.233616828918457, Test Accuracy: 55.659996032714844
Epoch 70, Loss: 0.03823954984545708, Accuracy: 98.88800048828125, Test Loss: 6.562615394592285, Test Accuracy: 55.58000183105469
```

简单的Conv2D+Dense，在mnist上效果不错，但是面对cifar10等开始变复杂的图片时，还是不够，所以需要增大网络，但是对比单纯的Dense来说，效果还是有提升的