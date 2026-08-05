import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import os

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else 'cpu')

Iteration = 350000
BATCH_SIZE = 64
momentum = 0.9
lr = 0.01


class Attention_module(nn.Module):
    def __init__(self, image_size=128):
        super(Attention_module, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def bottleneck(in_c, mid_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, mid_c, 1, 1, 0), nn.BatchNorm2d(mid_c), nn.ReLU(True),
                nn.Conv2d(mid_c, mid_c, 3, stride, 1), nn.BatchNorm2d(mid_c), nn.ReLU(True),
                nn.Conv2d(mid_c, out_c, 1, 1, 0), nn.BatchNorm2d(out_c)
            )

        self.layer2 = bottleneck(64, 64, 256)
        self.layer3 = bottleneck(256, 64, 256)
        self.layer4 = bottleneck(256, 64, 256)
        self.layer5 = bottleneck(256, 128, 512, 2)
        self.layer6 = bottleneck(512, 128, 512)
        self.layer7 = bottleneck(512, 128, 512)
        self.layer8 = bottleneck(512, 128, 512)
        self.layer9 = bottleneck(512, 256, 1024, 2)
        self.layer10 = bottleneck(1024, 256, 1024)
        self.layer11 = bottleneck(1024, 256, 1024)
        self.layer12 = bottleneck(1024, 256, 1024)
        self.layer13 = bottleneck(1024, 256, 1024)
        self.layer14 = bottleneck(1024, 256, 1024)
        self.layer15 = bottleneck(1024, 512, 2048, 2)
        self.layer16 = bottleneck(2048, 512, 2048)
        self.layer17 = bottleneck(2048, 512, 2048)

        def make_cbam_components(ch):
            mlp = nn.Sequential(nn.Conv2d(ch, ch // 16, 1), nn.ReLU(True), nn.Conv2d(ch // 16, ch, 1))
            spatial = nn.Conv2d(2, 1, 7, padding=3)
            return mlp, spatial

        self.mlp2, self.spa2 = make_cbam_components(256)
        self.mlp3, self.spa3 = make_cbam_components(256)
        self.mlp4, self.spa4 = make_cbam_components(256)
        self.mlp5, self.spa5 = make_cbam_components(512)
        self.mlp6, self.spa6 = make_cbam_components(512)
        self.mlp7, self.spa7 = make_cbam_components(512)
        self.mlp8, self.spa8 = make_cbam_components(512)
        self.mlp9, self.spa9 = make_cbam_components(1024)
        self.mlp10, self.spa10 = make_cbam_components(1024)
        self.mlp11, self.spa11 = make_cbam_components(1024)
        self.mlp12, self.spa12 = make_cbam_components(1024)
        self.mlp13, self.spa13 = make_cbam_components(1024)
        self.mlp14, self.spa14 = make_cbam_components(1024)
        self.mlp15, self.spa15 = make_cbam_components(2048)
        self.mlp16, self.spa16 = make_cbam_components(2048)
        self.mlp17, self.spa17 = make_cbam_components(2048)

        self.conv2 = nn.Conv2d(64, 256, 1, 1, 0);
        self.bn_conv2 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 1, 2, 0);
        self.bn_conv5 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1024, 1, 2, 0);
        self.bn_conv9 = nn.BatchNorm2d(1024)
        self.conv16 = nn.Conv2d(1024, 2048, 1, 2, 0);
        self.bn_conv16 = nn.BatchNorm2d(2048)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avgpool_final = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Linear(2048, 200)
        self.sigmoid = nn.Sigmoid()

    def cbam(self, feat, mlp, spatial):
        # Channel Attention
        ca = self.sigmoid(mlp(self.avg_pool(feat)) + mlp(self.max_pool(feat)))
        feat = feat * ca
        # Spatial Attention
        sa = self.sigmoid(
            spatial(torch.cat([torch.mean(feat, 1, keepdim=True), torch.max(feat, 1, keepdim=True)[0]], dim=1)))
        return feat * sa

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)

        sc = self.bn_conv2(self.conv2(x))
        x = F.relu(self.cbam(self.layer2(x), self.mlp2, self.spa2) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer3(x), self.mlp3, self.spa3) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer4(x), self.mlp4, self.spa4) + sc)

        sc = self.bn_conv5(self.conv5(x))
        x = F.relu(self.cbam(self.layer5(x), self.mlp5, self.spa5) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer6(x), self.mlp6, self.spa6) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer7(x), self.mlp7, self.spa7) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer8(x), self.mlp8, self.spa8) + sc)

        sc = self.bn_conv9(self.conv9(x))
        x = F.relu(self.cbam(self.layer9(x), self.mlp9, self.spa9) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer10(x), self.mlp10, self.spa10) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer11(x), self.mlp11, self.spa11) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer12(x), self.mlp12, self.spa12) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer13(x), self.mlp13, self.spa13) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer14(x), self.mlp14, self.spa14) + sc)

        sc = self.bn_conv16(self.conv16(x))
        x = F.relu(self.cbam(self.layer15(x), self.mlp15, self.spa15) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer16(x), self.mlp16, self.spa16) + sc)
        sc = x
        x = F.relu(self.cbam(self.layer17(x), self.mlp17, self.spa17) + sc)

        x = self.avgpool_final(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)
        return x


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

model = Attention_module().to(DEVICE).requires_grad_(True)
optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for i in range(Iteration):
    if i == 100000: optimizer.param_groups[0]['lr'] = 0.005
    if i == 200000: optimizer.param_groups[0]['lr'] = 0.001
    if i >= 250000: optimizer.param_groups[0]['lr'] = 0.001 - (i / Iteration) * 0.001

    model.train()
    batch_img = np.zeros((BATCH_SIZE, 3, 128, 128), dtype=np.float32)
    batch_gt = np.zeros(BATCH_SIZE, dtype=np.int64)
    train_random_idx = np.random.randint(0, len(train_list), BATCH_SIZE)

    for j in range(BATCH_SIZE):
        idx = train_random_idx[j]
        img = cv2.imread(os.path.join(train_path, train_list[idx]))
        img_tmp = (img.astype(np.float32) / 255.0) * 2 - 1

        if np.random.randint(2): img_tmp = cv2.flip(img_tmp, 1)
        if np.random.randint(2):
            blank = np.zeros((224, 224, 3), dtype=np.float32)
            x0, y0 = (224 - 128) // 2, (224 - 128) // 2
            blank[y0:y0 + 128, x0:x0 + 128] = img_tmp
            sy, sx = np.random.randint(0, 224 - 128 + 1), np.random.randint(0, 224 - 128 + 1)
            img_tmp = blank[sy:sy + 128, sx:sx + 128]
        if np.random.randint(2): img_tmp = cv2.rotate(img_tmp, cv2.ROTATE_90_CLOCKWISE)

        batch_img[j] = np.transpose(img_tmp, (2, 0, 1))
        batch_gt[j] = int(train_gt[idx]) - 1

    data, target = torch.from_numpy(batch_img).to(DEVICE), torch.from_numpy(batch_gt).to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f'Iter: {i}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    if i % 10000 == 0 and i != 0:
        model.eval()
        count = 0
        with torch.no_grad():
            for j in range(len(test_list)):
                t_img = cv2.imread(os.path.join(test_path, test_list[j]))
                if t_img is None: continue
                t_img_tmp = (t_img.astype(np.float32) / 255.0) * 2 - 1
                t_img_tmp = torch.from_numpy(np.transpose(t_img_tmp, (2, 0, 1))).unsqueeze(0).to(DEVICE)
                pre = torch.argmax(model(t_img_tmp), -1)
                if pre.item() == int(test_gt[j]) - 1: count += 1
        acc = count / len(test_list) * 100
        print(f"Iter: {i}, Accuracy: {acc:.2f}%")
        torch.save(model.state_dict(), f"./model_save/CBAM_Iter_{i}_acc_{acc:.1f}.pth")
        model.train()
