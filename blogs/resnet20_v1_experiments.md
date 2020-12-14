# resnet20_v1_experiments
## learning rate
0.9236-cifar-cifar_resnet20_v1-113-best.params
0.9094-cifar-cifar_resnet20_v1-91-best.params

| learning rate | test accuracy | epoch |     |     |
| ------------- | ------------- | ----- | --- | --- |
| 1e-1          | 0.9236        | 113   |     |     |
| 1e-2          | 0.9094        | 91    |     |     |
|               |               |       |     |     |
|               |               |       |     |     |
|               |               |       |     |     |
|               |               |       |     |     |
|               |               |       |     |     |
|               |               |       |     |     |
|               |               |       |     |     |
|               |               |       |     |     |

from the picture above, we can find out that set a larger initial learning rate, **helps skip local optimum value**

## batch size
0.9172-cifar-cifar_resnet20_v1-112-best.params
0.9236-cifar-cifar_resnet20_v1-113-best.params
0.9218-cifar-cifar_resnet20_v1-108-best.params

| batch size | test accuracy | epoch | learning rate |     |
| ---------- | ------------- | ----- | ------------- | --- |
| 80         | 0.9172        | 112   | 1e-1          |     |
| 160        | 0.9236        | 113   | 1e-1          |     |
| 320        | 0.9218        | 108   | 1e-1          |     |
|            |               |       |               |     |
|            |               |       |               |     |
|            |               |       |               |     |
|            |               |       |               |     |
|            |               |       |               |     |
|            |               |       |               |     |
|            |               |       |               |     |
|            |               |       |               |     |

from the picture above, larger batch size **helps increase model accuracy**

## optimizer
0.9236-cifar-cifar_resnet20_v1-113-best.params
0.9178-cifar-cifar_resnet20_v1-112-best.params
0.9106-cifar-cifar_resnet20_v1-117-best.params

| optimizer | test accuracy | epoch | learning rate |     |
| --------- | ------------- | ----- | ------------- | --- |
| sgd       | 0.9236        | 113   | 1e-1          |     |
| nag       | 0.9178        | 112   | 1e-1          |     |
| adam      | 0.9106        | 117   | 1e-3          |     |
|           |               |       |               |     |
|           |               |       |               |     |
|           |               |       |               |     |
|           |               |       |               |     |
|           |               |       |               |     |
|           |               |       |               |     |
|           |               |       |               |     |

from the picture above, **sgd has the highest accuracy**

## data augmentation
0.9103-cifar-cifar_resnet20_v1-111-best.params
0.9236-cifar-cifar_resnet20_v1-113-best.params
0.8984-cifar-cifar_resnet20_v1-119-best.params
0.9127-cifar-cifar_resnet20_v1-117-best.params
0.9077-cifar-cifar_resnet20_v1-113-best.params
0.9086-cifar-cifar_resnet20_v1-118-best.params
0.9033-cifar-cifar_resnet20_v1-108-best.params
0.9040-cifar-cifar_resnet20_v1-114-best.params
0.9132-cifar-cifar_resnet20_v1-116-best.params
0.8885-cifar-cifar_resnet20_v1-109-best.params

|  augmentation type  | test accuracy | epoch |     |     |
| ------------------- | ------------- | ----- | --- | --- |
| None                | 0.9103        | 111   |     |     |
| RandomFlipLeftRight | 0.9236        | 113   |     |     |
| RandomRotation      | 0.8984        | 119   |     |     |
| RandomBrightness    | 0.9127        | 117   |     |     |
| RandomLighting      | 0.9077        | 113   |     |     |
| RandomContrast      | 0.9086        | 118   |     |     |
| RandomHue           | 0.9033        | 108   |     |     |
| RandomSaturation    | 0.9040        | 114   |     |     |
| RandomColorJitter   | 0.9132        | 116   |     |     |
| RandomFlipTopBottom | 0.8885        | 109   |     |     |
|                     |               |       |     |     |

from the picture above, **RandomFlipLeftRight, RandomBrightness, RandomColorJitter** has a better effect