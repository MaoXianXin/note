## original code
```
import tensorflow as tf
mnist = tf.keras.datasets.mnist


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