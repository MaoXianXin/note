# speed_test
## inference speed
```
import torch
import numpy as np
import torchvision.models as models

model = models.vgg16()
device = torch.device("cuda")
model.to(device)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224, dtype = torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
repetitions = 300
timings = np.zeros((repetitions, 1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)
print(std_syn)
```

| network architecture |      time(ms)      |     |     |     |
| -------------------- | ------------------ | --- | --- | --- |
| resnet18             | 2.1302903445561725 |     |     |     |
| resnet34             | 3.7018090613683063 |     |     |     |
| resnet50             | 4.872166611353556  |     |     |     |
| resnet101            | 9.480131746927897  |     |     |     |
| resnet152            | 14.211658674875896 |     |     |     |
| resnext50_32x4d      | 7.2375160519282025 |     |     |     |
| resnext101_32x8d     | 15.577580699920654 |     |     |     |
| wide_resnet50_2      | 6.986582193374634  |     |     |     |
| wide_resnet101_2     | 12.818789520263671 |     |     |     |
|                      |                    |     |     |     |
| vgg11                | 3.50147488117218   |     |     |     |
| vgg11_bn             | 3.7368227235476175 |     |     |     |
| vgg13                | 4.396036580403646  |     |     |     |
| vgg13_bn             | 4.687968006134033  |     |     |     |
| vgg16                | 5.568336432774862  |     |     |     |
| vgg16_bn             | 5.703175349235535  |     |     |     |
| vgg19_bn             | 6.789220476150513  |     |     |     |
| vgg19                | 6.493688427607219  |     |     |     |
|                      |                    |     |     |     |
| squeezenet1_0        | 1.9096861894925434 |     |     |     |
| squeezenet1_1        | 1.9023944493134817 |     |     |     |
|                      |                    |     |     |     |
| densenet121          | 12.328606392542522 |     |     |     |
| densenet169          | 17.094060300191245 |     |     |     |
| densenet201          | 20.76851058959961  |     |     |     |
| densenet161          | 16.675101165771483 |     |     |     |
|                      |                    |     |     |     |
| shufflenet_v2_x0_5   | 4.92272340297699   |     |     |     |
| shufflenet_v2_x1_0   | 5.112371303240458  |     |     |     |
| shufflenet_v2_x1_5   | 5.121642880439758  |     |     |     |
| shufflenet_v2_x2_0   | 5.1313991435368855 |     |     |     |
|                      |                    |     |     |     |
| mobilenet_v2         | 4.2075329081217445 |     |     |     |
|                      |                    |     |     |     |
| mnasnet0_5           | 4.011432846387227  |     |     |     |
| mnasnet0_75          | 4.0848979226748146 |     |     |     |
| mnasnet1_0           | 3.894804159005483  |     |     |     |
| mnasnet1_3           | 4.087255358695984  |     |     |     |
|                      |                    |     |     |     |
|                      |                    |     |     |     |
|                      |                    |     |     |     |
|                      |                    |     |     |     |


## inference thoughput
```
import torch
import torchvision.models as models

optimal_batch_size = 1
model = models.resnet18()
device = torch.device("cuda")
model.to(device)
model.eval()
dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype = torch.float).to(device)

#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)

repetitions = 100
total_time = 0
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender) / 1000
     total_time += curr_time
Throughput = (repetitions * optimal_batch_size) / total_time
print('Final Throughput:', Throughput)
```

the network architecture is resnet18

| optimal_batch_size | Final Throughput(samples/sec) |     |     |     |
| ------------------ | ----------------------------- | --- | --- | --- |
| 1                  | 464                           |     |     |     |
| 2                  | 898                           |     |     |     |
| 4                  | 1166                          |     |     |     |
| 8                  | 1399                          |     |     |     |
| 16                 | 1809                          |     |     |     |
| 32                 | 1936                          |     |     |     |
| 64                 | 2077                          |     |     |     |
| 128                | 2119                          |     |     |     |
|                    |                               |     |     |     |