import cv2
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else 'cpu')


class VGG16(nn.Module):
    def __init__(self,image_size = 128):
        super( VGG16 , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride =2)
        self.FC = nn.Sequential(
            nn.Linear((image_size//(2**5))**2 * 512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,200)
        )
        # self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.maxpool(self.layer1(x))
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)
        # x = self.softmax(x)
        return x


class VGG19(nn.Module):
    def __init__(self):
        super( VGG19 , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride =2)
        self.FC = nn.Sequential(
            nn.Linear(8192,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,200)
        )
        # self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.maxpool(self.layer1(x))
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        x = self.maxpool(x)
        x = torch.flatten(x)
        x = self.FC(x)
        # x = self.softmax(x)
        return x

class VGG34(nn.Module):
    def __init__(self):
        super( VGG34 , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.FC = nn.Linear(512, 200)

    def forward(self,x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)
        return x

# MODEL_PATH = "C:/Users/AIVS/.conda/envs/YJH/pythonProject1/model_save/"
Iteration = 300000
BATCH_SIZE = 64
momentum = 0.9
weight_decay = 5e-4
lr = 0.01

train_path = r"C:\Users\AIVS\Desktop\tinyImageNet\train/"
train_list = os.listdir(train_path)
test_path = r"C:\Users\AIVS\Desktop\tinyImageNet\test/"
test_list = os.listdir(test_path)
train_gt_path = r"C:\Users\AIVS\Desktop\tinyImageNet\train_gt.txt"
f = open(train_gt_path, 'r')
train_gt = f.readlines()
test_gt_path = r"C:\Users\AIVS\Desktop\tinyImageNet\test_gt.txt"
f_gt = open(test_gt_path, 'r')
test_gt = f_gt.readlines()

model = VGG16().to(DEVICE).requires_grad_(True)
optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for i in range(Iteration):
    if i == 100000: optimizer.param_groups[0]['lr'] = 0.005
    if i == 200000: optimizer.param_groups[0]['lr'] = 0.001
    if i >= 250000:
        optimizer.param_groups[0]['lr'] = 0.001 - (i / Iteration) * 0.001

    model.train()
    train_random_idx = np.random.randint(0, len(train_list), BATCH_SIZE)
    batch_img = np.zeros((BATCH_SIZE, 3, 128, 128), dtype=np.float32)
    batch_gt = np.zeros(BATCH_SIZE, dtype=np.int64)

    for j in range(BATCH_SIZE):
        idx = train_random_idx[j]
        img_path = os.path.join(train_path, train_list[idx])
        img = cv2.imread(img_path)
        img_tmp = (img.astype(np.float32) / 255.0) * 2 - 1
        if np.random.randint(2):
            img_tmp = cv2.flip(img_tmp, 1)
        if np.random.randint(2):
            blank_img = np.zeros((224, 224, 3), dtype=np.float32)
            x0, y0 = (224 - 128) // 2, (224 - 128) // 2
            blank_img[y0:y0 + 128, x0:x0 + 128] = img_tmp
            sy = np.random.randint(0, 224 - 128 + 1)
            sx = np.random.randint(0, 224 - 128 + 1)
            img_tmp = blank_img[sy:sy + 128, sx:sx + 128]

        if np.random.randint(2):
            img_tmp = cv2.rotate(img_tmp, cv2.ROTATE_90_CLOCKWISE)

        img_tmp = np.transpose(img_tmp, (2, 0, 1))
        batch_img[j] = img_tmp
        batch_gt[j] = int(train_gt[idx]) - 1

    data = torch.from_numpy(batch_img).to(DEVICE)
    target = torch.from_numpy(batch_gt).to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss_val = criterion(output, target)
    loss_val.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f'Iter: {i}, Loss: {loss_val.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    if i % 10000 == 0 and i != 0:
        model.eval()
        count = 0
        with torch.no_grad():
            for j in range(len(test_list)):
                test_img_path = os.path.join(test_path, test_list[j])
                test_img = cv2.imread(test_img_path)
                if test_img is None: continue
                test_img_tmp = (test_img.astype(np.float32) / 255.0) * 2 - 1
                test_img_tmp = np.transpose(test_img_tmp, (2, 0, 1))
                test_img_tmp = torch.from_numpy(test_img_tmp).unsqueeze(0).to(DEVICE)
                test_output = model(test_img_tmp)
                pre = torch.argmax(test_output, -1)
                if pre.item() == int(test_gt[j]) - 1:
                    count += 1
        accuracy = count / len(test_list) * 100
        print(f"Iter: {i}, Count: {count}, Accuracy: {accuracy:.2f}%")
        torch.save(model.state_dict(), f"./model_save/Iter_{i}_acy_{accuracy:.1f}.pth")
        model.train()



