# clr_gradient_accumulation
```
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from clr_callback import CyclicLR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 64
img_height = 224
img_width = 224
strategy = tf.distribute.MirroredStrategy()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/mao/PycharmProjects/convolution-network-natural-scenes/dataset/natural-scenes/seg_train",
    # validation_split=0.2,
    # subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/mao/PycharmProjects/convolution-network-natural-scenes/dataset/natural-scenes/seg_test",
    # validation_split=0.2,
    # subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# image_batch, labels_batch = next(iter(train_ds))
# first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))

with strategy.scope():
    num_classes = 6
    data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1),
            layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
            layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    base_model = tf.keras.applications.resnet.ResNet50(input_shape=(img_height, img_width, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    initial_epoch = 40
    kwargs = {'experimental_steps_per_execution': 256}
    # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)]
    clr = CyclicLR(base_lr=1e-6, max_lr=1e-3, step_size=1)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'],
                  **kwargs
                  )
model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch, callbacks=[clr])
print('--------')
model.evaluate(val_ds)
print('--------')

with strategy.scope():
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    fine_tune_at = int(len(base_model.layers) * 0.4)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    clr = CyclicLR(base_lr=1e-6, max_lr=1e-5, step_size=1)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-4),
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'],
                  **kwargs
                  )
fine_tune_epochs=80
total_epochs = initial_epoch + fine_tune_epochs
model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=initial_epoch, callbacks=[clr])
model.evaluate(val_ds)
```