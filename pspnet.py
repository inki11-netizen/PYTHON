import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torchvision import transforms
from PIL import Image

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else 'cpu')

color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def fast_hist(g, p, n):
    k = (g >= 0) & (g < n)
    return np.bincount(n * g[k].astype(int) + p[k], minlength=n ** 2).reshape(n, n)


def compute_metrics(hist):
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return acc, mean_iu, iu


def get_voc_palette():
    palette = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
        [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ]
    return np.array(palette, dtype=np.uint8)


def decode_segmap(mask):
    palette = get_voc_palette()
    rgb = palette[mask]
    return rgb

class pspnet(nn.Module):
    def __init__(self, image_size=128):
        super(pspnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048)
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048)
        )
        self.layer17 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048)
        )
        self.conv2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.bn_conv2 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.bn_conv5 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        self.bn_conv9 = nn.BatchNorm2d(1024)
        self.conv16 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0)
        self.bn_conv16 = nn.BatchNorm2d(2048)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.final_conv = nn.Conv2d(2048, 21, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]  # 원본 이미지 크기(256x256)를 기억해둡니다.

        x = self.layer1(x)  # 128
        x = self.maxpool(x)  # 64
        shortcut1 = self.bn_conv2(self.conv2(x))
        x = self.layer2(x)  # 32
        x = F.relu(x + shortcut1)
        shortcut2 = x
        x = self.layer3(x)
        x = F.relu(x + shortcut2)
        shortcut3 = x
        x = self.layer4(x)
        x = F.relu(x + shortcut3)
        shortcut4 = self.bn_conv5(self.conv5(x))  # 16
        x = self.layer5(x)
        x = F.relu(x + shortcut4)
        shortcut5 = x
        x = self.layer6(x)
        x = F.relu(x + shortcut5)
        shortcut6 = x
        x = self.layer7(x)
        x = F.relu(x + shortcut6)
        shortcut7 = x
        x = self.layer8(x)
        x = F.relu(x + shortcut7)
        shortcut8 = self.bn_conv9(self.conv9(x))  # 8
        x = self.layer9(x)
        x = F.relu(x + shortcut8)
        shortcut9 = x
        x = self.layer10(x)
        x = F.relu(x + shortcut9)
        shortcut10 = x
        x = self.layer11(x)
        x = F.relu(x + shortcut10)
        shortcut11 = x
        x = self.layer12(x)
        x = F.relu(x + shortcut11)
        shortcut12 = x
        x = self.layer13(x)
        x = F.relu(x + shortcut12)
        shortcut13 = x
        x = self.layer14(x)
        x = F.relu(x + shortcut13)
        shortcut14 = self.bn_conv16(self.conv16(x))  # 4
        x = self.layer15(x)
        x = F.relu(x + shortcut14)
        shortcut15 = x
        x = self.layer16(x)
        x = F.relu(x + shortcut15)
        shortcut16 = x
        x = self.layer17(x)
        x = F.relu(x + shortcut16)
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x

Iteration = 120000
BATCH_SIZE = 32
lr = 0.01

train_path = r"C:\Users\dongj\Desktop\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Segmentation\train_aug.txt"
test_path = r"C:\Users\dongj\Desktop\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt"
train_img_path = r"C:\Users\dongj\Desktop\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
train_gt_path = r"C:\Users\dongj\Desktop\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClassAug"

with open(train_path, 'r') as f:
    train_list = [line.strip() for line in f.readlines()]
with open(test_path, 'r') as f:
    test_list = [line.strip() for line in f.readlines()]

model = pspnet().to(DEVICE)
print(">>> Training with user's custom backbone from scratch.")
model.requires_grad_(True)

optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(ignore_index=255)

SAVE_DIR = "./result_vis"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if not os.path.exists("./model_save"): os.makedirs("./model_save")

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

        # BGR을 RGB로 변환 (모델 학습 및 증강을 위함)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if np.random.randint(2):
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if np.random.randint(2):
            b_img = np.zeros((352, 352, 3), dtype=np.uint8)
            b_mask = np.full((352, 352), 255, dtype=np.uint8)
            x0, y0 = 48, 48
            b_img[y0:y0 + 256, x0:x0 + 256] = img
            b_mask[y0:y0 + 256, x0:x0 + 256] = mask
            sy, sx = np.random.randint(0, 97), np.random.randint(0, 97)
            img = b_img[sy:sy + 256, sx:sx + 256]
            mask = b_mask[sy:sy + 256, sx:sx + 256]

        if np.random.randint(2):
            img_pil = Image.fromarray(img)
            img_pil = color_transform(img_pil)
            img = np.array(img_pil)

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
        total_hist = np.zeros((21, 21))

        with torch.no_grad():
            test_indices = np.random.randint(0, len(test_list), 100)
            vis_count = 0
            print(f"\n>>> [Evaluation Start] Iter: {i}")

            for tidx in test_indices:
                t_name = test_list[tidx]
                t_img_path = os.path.join(train_img_path, t_name + ".jpg")
                t_mask_path = os.path.join(train_gt_path, t_name + ".png")

                t_img = cv2.imread(t_img_path)
                t_mask = cv2.imread(t_mask_path, cv2.IMREAD_GRAYSCALE)

                if t_img is None or t_mask is None: continue

                # 저장을 위해 BGR 원본 보존
                t_img_bgr = cv2.resize(t_img, (256, 256))

                # 모델 입력을 위해 RGB 변환
                t_img_rgb = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
                t_img_rgb = cv2.resize(t_img_rgb, (256, 256))
                t_mask = cv2.resize(t_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                t_input = torch.from_numpy(
                    np.transpose((t_img_rgb.astype(np.float32) / 255.0) * 2 - 1, (2, 0, 1))
                ).unsqueeze(0).to(DEVICE)

                t_output = model(t_input)
                pred = torch.argmax(t_output, dim=1).squeeze(0).cpu().numpy()

                cur_hist = fast_hist(t_mask.flatten(), pred.flatten(), 21)
                total_hist += cur_hist

                cur_acc, cur_mean_iu, cur_iu = compute_metrics(cur_hist)
                if vis_count < 5:
                    print(f"   [Image {vis_count + 1}] {t_name}")
                    print(f"     -> Pixel Acc: {cur_acc * 100:.2f}% | mIoU: {cur_mean_iu * 100:.2f}%")

                    unique_classes = np.unique(t_mask)
                    print(f"     -> Class-wise IoU:")
                    for cls_idx in unique_classes:
                        if cls_idx == 255: continue
                        if cls_idx < len(VOC_CLASSES):
                            cls_name = VOC_CLASSES[cls_idx]
                            cls_iou = cur_iu[cls_idx] * 100
                            print(f"        - {cls_name:<12}: {cls_iou:.2f}%")
                    print("-" * 30)

                    gt_vis = t_mask.copy()
                    gt_vis[gt_vis == 255] = 0

                    gt_color_rgb = decode_segmap(gt_vis)
                    pred_color_rgb = decode_segmap(pred)

                    # cv2 저장을 위해 RGB를 BGR로 변환
                    gt_color_bgr = cv2.cvtColor(gt_color_rgb, cv2.COLOR_RGB2BGR)
                    pred_color_bgr = cv2.cvtColor(pred_color_rgb, cv2.COLOR_RGB2BGR)

                    combined = np.hstack([t_img_bgr, gt_color_bgr, pred_color_bgr])

                    save_name = f"Iter_{i}_{t_name}_mIoU_{cur_mean_iu * 100:.1f}.png"
                    save_path = os.path.join(SAVE_DIR, save_name)
                    cv2.imwrite(save_path, combined)

                    vis_count += 1

        total_acc, total_mean_iu, _ = compute_metrics(total_hist)
        print(f">>> [TOTAL AVG] Iter: {i}")
        print(f"    Batch Pixel Accuracy: {total_acc * 100:.2f}%")
        print(f"    Batch Mean IoU      : {total_mean_iu * 100:.2f}%")
        print(f"    Images Saved to {SAVE_DIR}/")

        torch.save(model.state_dict(), f"./model_save/Iter_{i}_mIoU_{total_mean_iu * 100:.1f}.pth")
        start = time.time()