# clustering
```
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import mxnet
import os
from mxboard import SummaryWriter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', default=6, type=int)
parser.add_argument("--path", default="/home/mao/Github/datasets/natural-scenes", type=str)
args = parser.parse_args()

batch_size = 4
num_workers = 4

transform_train = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

path = args.path
train_path = os.path.join(path, "seg_train")
test_path = os.path.join(path, "seg_test")
train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='rollover'
)
test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='rollover'
)

initialized = False
embedding = None
labels = None
images = None

for i, (data, label) in enumerate(train_data):
    if i >= 800:
        # only fetch the first 100 batches of images
        break
    if initialized:  # after the first batch, concatenate the current batch with the existing one
        embedding = mxnet.nd.concat(*(embedding, data.reshape(batch_size,-1)), dim=0)
        labels = mxnet.nd.concat(*(labels, label), dim=0)
        images = mxnet.nd.concat(*(images, data), dim=0)
    else:  # first batch of images, directly assign
        embedding = data.reshape(batch_size,-1)
        print(embedding.shape)
        labels = label
        print(labels.shape)
        images = data
        print(images.shape)
        initialized = True

with SummaryWriter(logdir='./logs') as sw:
    sw.add_embedding(tag='natural-scenes', embedding=embedding, labels=labels, images=images)
```