# classification_report
```
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from clr_callback import CyclicLR
from sklearn.metrics import classification_report
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

batch_size = 64
img_height = 224
img_width = 224
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street",]
strategy = tf.distribute.MirroredStrategy()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/mao/PycharmProjects/convolution-network-natural-scenes/dataset/natural-scenes/seg_train",
    image_size=(256, 256),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/mao/PycharmProjects/convolution-network-natural-scenes/dataset/natural-scenes/seg_test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(5000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
random_crop = layers.experimental.preprocessing.RandomCrop(height=224, width=224)
train_ds = train_ds.map(lambda x, y: (random_crop(x), y))

with strategy.scope():
    num_classes = 6
    data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
            layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
            layers.experimental.preprocessing.RandomContrast(0.1),
            layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1),
    ])
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=(img_height, img_width, 3),
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

    initial_epoch = 250
    # kwargs = {'experimental_steps_per_execution': 1}
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    clr = CyclicLR(base_lr=1e-5, max_lr=1e-4, step_size=110)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'],
                  # **kwargs
                  )
model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch, callbacks=[clr, earlystop])
print('--------')
model.evaluate(val_ds)
model.save_weights('base.h5')
print('--------')

with strategy.scope():
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    fine_tune_at = int(len(base_model.layers) * 0.6)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    clr = CyclicLR(base_lr=1e-6, max_lr=1e-5, step_size=110)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-4),
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'],
                  # **kwargs
                  )
fine_tune_epochs=250
total_epochs = initial_epoch + fine_tune_epochs
model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=initial_epoch, callbacks=[clr, earlystop])
model.evaluate(val_ds)

predictions = []
groundtruth = []
for images, labels in val_ds:
    y_pred = np.argmax(model.predict(images), axis=1)
    predictions.extend(y_pred)
    groundtruth.extend(np.asarray(labels))
print(classification_report(groundtruth, predictions, target_names=class_names))
```