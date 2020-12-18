# clr_learning_rate
```
from clr_callback import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
import matplotlib.pyplot as plt

inp = Input(shape=(15,))
x = Dense(10, activation='relu')(inp)
x = Dense(1, activation='sigmoid')(x)

model = Model(inp, x)

X = np.random.rand(2000, 15)
Y = np.random.randint(0, 2, size=2000)

kwargs = {'experimental_steps_per_execution': 1}
clr = CyclicLR(base_lr=1e-2, max_lr=1e-1, step_size=500)
model.compile(optimizer=SGD(0.1), loss='binary_crossentropy', metrics=['accuracy'], **kwargs)
model.fit(X, Y, batch_size=1, epochs=10, callbacks=[clr], verbose=0)

plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("CLR - 'triangular' Policy")
plt.plot(clr.history['iterations'], clr.history['lr'])
plt.show()
```