import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision.models.vgg import vgg16
transform = T.Compose([
    T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
])



class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


model = vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096, 4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096, 2))

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True
model = model.cuda()
cost = torch.nn.CrossEntropyLoss()
cost = cost.cuda()
optimizer = torch.optim.Adam(model.classifier.parameters())

dataset = DogCat('./train', transforms=transform)
dataloader = data.DataLoader(dataset, batch_size=150, shuffle=True)

for i in range(10):
    running_loss = 0.0
    print('-----epoch', i, '-----')
    for num, data in enumerate(dataloader):

        x_train, y_train = data
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        optimizer.zero_grad()
        output = model(x_train)
        loss = cost(output, y_train)
        print(num*150, '/ 25000', 'loss:', loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('-----------Epoch:', i, ', loss', running_loss, '-----------')
    torch.save(model, 'dog_cat_model.pkl')
