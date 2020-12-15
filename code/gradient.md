## original code
```
import tensorflow as tf
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


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


kwargs = {'experimental_steps_per_execution': 256}
model.compile(optimizer = 'sgd',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'],
              **kwargs
              )
model.fit(x_train, y_train, epochs=70, validation_split=0.3)
model.evaluate(x_test, y_test)
```

## class weight
```
def fit(self,
      x=None,
      y=None,
      batch_size=None,
      epochs=1,
      verbose=1,
      callbacks=None,
      validation_split=0.,
      validation_data=None,
      shuffle=True,
      class_weight=None,
      sample_weight=None,
      initial_epoch=0,
      steps_per_epoch=None,
      validation_steps=None,
      validation_batch_size=None,
      validation_freq=1,
      max_queue_size=10,
      workers=1,
      use_multiprocessing=False):
```
```
class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
```

## experiments
### mnist
```
Epoch 65/70
1313/1313 [==============================] - 1s 553us/step - loss: 0.0716 - accuracy: 0.9791 - val_loss: 0.0953 - val_accuracy: 0.9731
Epoch 66/70
1313/1313 [==============================] - 1s 575us/step - loss: 0.0703 - accuracy: 0.9805 - val_loss: 0.0960 - val_accuracy: 0.9719
Epoch 67/70
1313/1313 [==============================] - 1s 560us/step - loss: 0.0698 - accuracy: 0.9798 - val_loss: 0.0948 - val_accuracy: 0.9722
Epoch 68/70
1313/1313 [==============================] - 1s 571us/step - loss: 0.0680 - accuracy: 0.9795 - val_loss: 0.0946 - val_accuracy: 0.9725
Epoch 69/70
1313/1313 [==============================] - 1s 587us/step - loss: 0.0680 - accuracy: 0.9809 - val_loss: 0.0946 - val_accuracy: 0.9726
Epoch 70/70
1313/1313 [==============================] - 1s 570us/step - loss: 0.0680 - accuracy: 0.9800 - val_loss: 0.0940 - val_accuracy: 0.9721
313/313 [==============================] - 0s 72us/step - loss: 0.0811 - accuracy: 0.9751
```
### cifar10
```
Epoch 67/70
1094/1094 [==============================] - 1s 663us/step - loss: 1.2096 - accuracy: 0.5749 - val_loss: 1.4059 - val_accuracy: 0.5097
Epoch 68/70
1094/1094 [==============================] - 1s 653us/step - loss: 1.2039 - accuracy: 0.5757 - val_loss: 1.4473 - val_accuracy: 0.4998
Epoch 69/70
1094/1094 [==============================] - 1s 672us/step - loss: 1.2015 - accuracy: 0.5749 - val_loss: 1.4095 - val_accuracy: 0.5081
Epoch 70/70
1094/1094 [==============================] - 1s 652us/step - loss: 1.1947 - accuracy: 0.5806 - val_loss: 1.4026 - val_accuracy: 0.5138
313/313 [==============================] - 0s 93us/step - loss: 1.3869 - accuracy: 0.5122
```

单纯的Dense在mnist上效果不错，但面对cifar10等开始变复杂的图片时，凸显出能力不足了