import cv2
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
from torchvision import transforms
from PIL import Image

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else 'cpu')


# Color Jitter 설정
color_transform = transforms.ColorJitter(
    brightness=0.3,  # 30% 변동
    contrast=0.3,  # 30% 변동
    saturation=0.3,  # 30% 변동
    hue=0.1  # 10% 변동
)
def fast_hist(g, p, n):
    k = (g >= 0) & (g < n)
    return np.bincount(n * g[k].astype(int) + p[k], minlength=n ** 2).reshape(n, n)

def compute_metrics(hist):
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return acc, mean_iu

def get_voc_palette():
    palette = [
        [0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0],
        [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192],
        [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64],
        [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128],
        [128, 64, 0]
    ]
    return np.array(palette, dtype=np.uint8)


def decode_segmap(mask):
    palette = get_voc_palette()
    rgb = palette[mask]
    return rgb

class VGG16(nn.Module):
    def __init__(self, image_size=256):
        super(VGG16, self).__init__()
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
        self.f0 = nn.Sequential(
            nn.Conv2d(256, 21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(512, 21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Conv2d(4096, 21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.upscore2 = nn.ConvTranspose2d(21, 21, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore4 = nn.ConvTranspose2d(21, 21, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore32 = nn.ConvTranspose2d(21, 21, kernel_size=16, stride=8, padding=4, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.layer1(x))  # 128
        x = self.layer2(x)
        x = self.maxpool(x)  # 64
        x = self.layer3(x)
        pool3 = self.maxpool(x)  # 32
        u1 = self.f0(pool3)
        x = self.layer4(pool3)
        pool4 = self.maxpool(x)  # 16
        u2 = self.f1(pool4)
        x = self.layer5(pool4)
        pool5 = self.maxpool(x)  # 8
        x = self.fc1(pool5)
        x = self.fc2(x)
        x = self.fc3(x)
        score_pool5 = x  # 8x8
        upscore2 = self.upscore2(score_pool5)  # 16x16
        score_pool4 = upscore2 + u2  # 16x16
        upscore4 = self.upscore4(score_pool4)  # 32x32
        score_pool3 = upscore4 + u1  # 32x32
        out = self.upscore32(score_pool3)  # 256x256
        return out

Iteration = 120000
BATCH_SIZE = 64
lr = 0.01

train_path = r"C:\Users\AIVS\Desktop\VOC2012\ImageSets\Segmentation\train_aug.txt"
test_path = r"C:\Users\AIVS\Desktop\VOC2012\ImageSets\Segmentation\val.txt"
train_img_path = r"C:\Users\AIVS\Desktop\VOC2012\JPEGImages"
train_gt_path = r"C:\Users\AIVS\Desktop\VOC2012\SegmentationClassAug"

with open(train_path, 'r') as f:
    train_list = [line.strip() for line in f.readlines()]
with open(test_path, 'r') as f:
    test_list = [line.strip() for line in f.readlines()]

model = VGG16().to(DEVICE)
model.requires_grad_(True)

optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(ignore_index=255)

if not os.path.exists("./result_vis"):
    os.makedirs("./result_vis")

if not os.path.exists("./model_save"):
    os.makedirs("./model_save")

start = time.time()

for i in range(Iteration):
    model.train()
    train_random_idx = np.random.randint(0, len(train_list), BATCH_SIZE)
    batch_img = np.zeros((BATCH_SIZE, 3, 256, 256), dtype=np.float32)
    batch_gt = np.zeros((BATCH_SIZE, 256, 256), dtype=np.int64)

    for j in range(BATCH_SIZE):
        idx = train_random_idx[j]
        file_name = train_list[idx]
        img = cv2.imread(os.path.join(train_img_path, file_name + ".jpg"))
        mask = cv2.imread(os.path.join(train_gt_path, file_name + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 1. Flip
        if np.random.randint(2):
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 2. Crop
        if np.random.randint(2):
            b_img = np.zeros((352, 352, 3), dtype=np.uint8)
            b_mask = np.full((352, 352), 255, dtype=np.uint8)
            x0, y0 = 48, 48
            b_img[y0:y0 + 256, x0:x0 + 256] = img
            b_mask[y0:y0 + 256, x0:x0 + 256] = mask
            sy, sx = np.random.randint(0, 97), np.random.randint(0, 97)
            img = b_img[sy:sy + 256, sx:sx + 256]
            mask = b_mask[sy:sy + 256, sx:sx + 256]

        # Color Jitter 적용 (원본 이미지가 BGR이므로 RGB 변환 후 적용)
        if np.random.randint(2):  # 50% 확률로 Jitter 적용
            # CV2(BGR) -> PIL(RGB)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Jitter 적용
            img_pil = color_transform(img_pil)
            # PIL(RGB) -> CV2(BGR)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        batch_img[j] = np.transpose((img.astype(np.float32) / 255.0) * 2 - 1, (2, 0, 1))
        batch_gt[j] = mask.astype(np.int64)

    data, target = torch.from_numpy(batch_img).to(DEVICE), torch.from_numpy(batch_gt).to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        end = time.time()
        print(f'Iter: {i}, Loss: {loss.item():.4f}, Time: {end - start:.5f} sec')
        start = time.time()

    if i % 10000 == 0 and i != 0:
        model.eval()
        hist = np.zeros((21, 21))

        with torch.no_grad():
            test_indices = np.random.randint(0, len(test_list), 100)
            vis_count = 0

            for tidx in test_indices:
                t_name = test_list[tidx]
                t_img = cv2.imread(os.path.join(train_img_path, t_name + ".jpg"))
                t_mask = cv2.imread(os.path.join(train_gt_path, t_name + ".png"), cv2.IMREAD_GRAYSCALE)

                if t_img is None or t_mask is None: continue

                t_img = cv2.resize(t_img, (256, 256))
                t_mask = cv2.resize(t_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                t_input = torch.from_numpy(
                    np.transpose((t_img.astype(np.float32) / 255.0) * 2 - 1, (2, 0, 1))).unsqueeze(0).to(DEVICE)

                t_output = model(t_input)
                pred = torch.argmax(t_output, dim=1).squeeze(0).cpu().numpy()

                hist += fast_hist(t_mask.flatten(), pred.flatten(), 21)

                if vis_count < 5:
                    gt_vis = t_mask.copy()
                    gt_vis[gt_vis == 255] = 0
                    gt_color = decode_segmap(gt_vis)
                    pred_color = decode_segmap(pred)
                    combined = np.hstack([t_img, gt_color, pred_color])
                    save_path = f"./result_vis_aa/Iter_{i}_{t_name}.png"
                    cv2.imwrite(save_path, combined)
                    vis_count += 1

        acc, mean_iu = compute_metrics(hist)

        print(f">>> [TEST] Iter: {i}")
        print(f"    Pixel Accuracy: {acc * 100:.2f}%")
        print(f"    Mean IoU      : {mean_iu * 100:.2f}%")
        print(f"    Images Saved to ./result_vis/")

        torch.save(model.state_dict(), f"./model_save/Iter_{i}_mIoU_{mean_iu * 100:.1f}.pth")
        start = time.time()
        model.train()