import os
import glob
import cv2
import math
import copy
import numpy as np
import nibabel as nib
import platform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
import torch.optim as optim
import ml_collections
import csv
import random

# 💡 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 💡 클래스 이름 및 색상 매핑 사전 (총 11개 클래스)
ORGAN_NAMES = {
    0: "배경",
    1: "대동맥",
    2: "담낭",
    3: "좌측 신장",
    4: "우측 신장",
    5: "간",
    6: "췌장",
    7: "비장",
    8: "위",
    9: "간 종양(Liver Tumor)",
    10: "신장 종양(Kidney Tumor)"
}
# 🌟 시각화 전용 커스텀 헥스 컬러코드 정의
ORGAN_COLORS_HEX = [
    "#000000",  # 0: 배경 (검정)
    "#0000FF",  # 1: 대동맥 (파랑)
    "#008000",  # 2: 담낭 (초록)
    "#FF0000",  # 3: 좌측 신장 (빨강)
    "#00FFFF",  # 4: 우측 신장 (청록)
    "#FF00FF",  # 5: 간 (자홍)
    "#FFFF00",  # 6: 췌장 (노랑)
    "#FFA500",  # 7: 비장 (주황)
    "#800080",  # 8: 위 (보라)
    "#8B4513",  # 9: 간 종양 (갈색)
    "#A9A9A9"  # 10: 신장 종양 (회색)

]

# U-Net 코드에서 사용한 confusion matrix 기반 Pixel Accuracy / mIoU
def fast_hist(gt, pred, num_classes):
    """
    정답(gt)과 예측(pred)으로 confusion matrix를 계산합니다.
    ignore_index=255 픽셀은 자동으로 제외됩니다.
    """
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def compute_metrics(hist):
    """
    confusion matrix로부터 Pixel Accuracy, mIoU, 클래스별 IoU를 계산합니다.
    실제 정답에 한 번도 등장하지 않은 클래스의 IoU는 NaN으로 두고
    mIoU 평균에서 제외합니다.
    """
    total_pixels = hist.sum()
    pixel_acc = np.diag(hist).sum() / total_pixels if total_pixels > 0 else 0.0

    denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_iou = np.diag(hist) / denominator

    mean_iou = np.nanmean(class_iou) if np.any(~np.isnan(class_iou)) else 0.0
    return pixel_acc, mean_iou, class_iou

def evaluate_model(model, data_loader, seg_criterion, device, num_classes):
    """
    라벨이 존재하는 검증/테스트 데이터만 사용하여
    segmentation loss, Pixel Accuracy, mIoU, 클래스별 IoU를 계산합니다.
    """
    model.eval()
    total_hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    total_seg_loss = 0.0
    labeled_batch_count = 0
    labeled_image_count = 0

    with torch.no_grad():
        for inputs, targets, has_mask in data_loader:
            # has_mask는 배치 단위 bool tensor
            has_mask = torch.as_tensor(has_mask, dtype=torch.bool)
            if not has_mask.any():
                continue

            inputs = inputs[has_mask].to(device)
            targets = targets[has_mask].to(device)

            seg_logits, _ = model(inputs)
            seg_loss = seg_criterion(seg_logits, targets)

            predictions = torch.argmax(seg_logits, dim=1)

            total_seg_loss += seg_loss.item()
            labeled_batch_count += 1
            labeled_image_count += targets.size(0)

            targets_np = targets.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            for gt_mask, pred_mask in zip(targets_np, predictions_np):
                total_hist += fast_hist(gt_mask, pred_mask, num_classes)

    pixel_acc, mean_iou, class_iou = compute_metrics(total_hist)
    avg_seg_loss = (
        total_seg_loss / labeled_batch_count
        if labeled_batch_count > 0
        else float("nan")
    )

    return {
        "seg_loss": avg_seg_loss,
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "class_iou": class_iou,
        "hist": total_hist,
        "num_labeled_images": labeled_image_count,
    }

# 🛠️ 3D NIfTI -> 2D PNG 자동 전처리 (Slicing) 함수
def preprocess_3d_to_2d(img_dir, label_dir, out_dir, is_test=False):
    """
    3D (.nii / .nii.gz) 의료 영상을 2D 슬라이스로 잘라 저장합니다.
    - is_test=True 이면 라벨(정답지) 처리를 생략합니다.
    """
    out_img_dir = os.path.join(out_dir, "images")
    out_label_dir = os.path.join(out_dir, "labels") if not is_test else None

    os.makedirs(out_img_dir, exist_ok=True)
    if not is_test:
        os.makedirs(out_label_dir, exist_ok=True)

    # 이미 변환된 파일이 존재하는 경우 패스
    if len(os.listdir(out_img_dir)) > 0:
        print(f"✅ 이미 2D 변환이 완료되어 있습니다: {out_dir}")
        return

    print(f"⏳ 3D 파일을 2D 슬라이스로 썰고 있습니다... ({'Test' if is_test else 'Train'} 데이터)")

    # BTCV 원본 번호를 논문 규격(1~8)으로 변경하는 매핑
    btcv_to_transunet = {8: 1, 4: 2, 3: 3, 2: 4, 6: 5, 11: 6, 1: 7, 7: 8}

    # .nii 또는 .nii.gz 파일 모두 찾기
    nii_files = glob.glob(os.path.join(img_dir, "*.nii*"))

    for nii_path in nii_files:
        filename = os.path.basename(nii_path)
        img_vol = nib.load(nii_path).get_fdata()

        label_vol = None
        if not is_test and label_dir is not None:
            # Synapse 라벨명은 img0001 -> label0001 인 경우가 많음
            label_name = filename.replace("img", "label")
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):  # 이름이 똑같은 경우 대비
                label_path = os.path.join(label_dir, filename)

            if os.path.exists(label_path):
                label_vol = nib.load(label_path).get_fdata()

        # Z축 깊이(보통 3번째 차원) 만큼 반복해서 자르기
        for z in range(img_vol.shape[2]):
            slice_img = img_vol[:, :, z]

            slice_label = None
            if label_vol is not None:
                slice_label = label_vol[:, :, z]
                # 학습 시: 장기가 하나도 없는 빈 배경 슬라이스는 과감히 버려서 속도 극대화
                if np.max(slice_label) == 0:
                    continue

            # 복부 CT Windowing (-125 ~ 275) 및 정규화 (0~255)
            slice_img = np.clip(slice_img, -125, 275)
            slice_img = ((slice_img - (-125)) / 400.0 * 255.0).astype(np.uint8)

            # 이미지 저장
            base_name = filename.split('.')[0] + f"_slice_{z:03d}.png"
            cv2.imwrite(os.path.join(out_img_dir, base_name), slice_img)

            # 라벨 맵핑 및 저장
            if not is_test and slice_label is not None:
                mapped_label = np.zeros_like(slice_label, dtype=np.uint8)
                for raw_val, new_val in btcv_to_transunet.items():
                    mapped_label[slice_label == raw_val] = new_val
                cv2.imwrite(os.path.join(out_label_dir, base_name), mapped_label)

    print(f"✅ 3D -> 2D 변환 완료: {out_dir}")

# 모델 Configuration
def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = None
    config.patch_size = 16
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 11
    config.activation = 'softmax'
    return config

def get_r50_b16_config():
    config = get_b16_config()
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    config.classifier = 'seg'
    config.pretrained_path = None
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 11
    config.n_skip = 3
    config.activation = 'softmax'
    return config

CONFIGS = {'R50-ViT-B_16': get_r50_b16_config()}

# 모델 구조 (Encoder & Decoder)
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1): return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(cin, cout, stride=1): return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)

class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)
        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = self.gn_proj(self.downsample(x)) if hasattr(self, 'downsample') else x
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)

class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(OrderedDict(
            [('conv', StdConv2d(3, width, 7, 2, padding=3, bias=False)), ('gn', nn.GroupNorm(32, width)),
             ('relu', nn.ReLU(inplace=True))]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict([('unit1', PreActBottleneck(width, width * 4, width))] + [
                (f'unit{i}', PreActBottleneck(width * 4, width * 4, width)) for i in range(2, block_units[0] + 1)]))),
            ('block2', nn.Sequential(OrderedDict([('unit1', PreActBottleneck(width * 4, width * 8, width * 2, 2))] + [
                (f'unit{i}', PreActBottleneck(width * 8, width * 8, width * 2)) for i in
                range(2, block_units[1] + 1)]))),
            ('block3', nn.Sequential(OrderedDict([('unit1', PreActBottleneck(width * 8, width * 16, width * 4, 2))] + [
                (f'unit{i}', PreActBottleneck(width * 16, width * 16, width * 4)) for i in
                range(2, block_units[2] + 1)]))),
        ]))

    def forward(self, x):
        features = []
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            features.append(x)
        return self.body[-1](x), features[::-1]

class Attention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.proj_dropout(self.out(context_layer))
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = F.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = True
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels, out_channels=config.hidden_size,
                                       kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = self.dropout(x + self.position_embeddings)
        return embeddings, features

class Block(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x, weights = self.attn(self.attention_norm(x))
        x = x + h
        h = x
        x = self.ffn(self.ffn_norm(x))
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([copy.deepcopy(Block(config, vis)) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis: attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super().__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None: x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = self.config.skip_channels
        for i in range(4 - self.config.n_skip): skip_channels[3 - i] = 0

        self.blocks = nn.ModuleList([DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
                                     zip(in_channels, out_channels, skip_channels)])

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x

class VisionTransformerMultiTask(nn.Module):
    def __init__(self, config, img_size=224, num_classes=11, vis=False):
        super().__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)

        self.seg_head = SegmentationHead(in_channels=config.decoder_channels[-1], out_channels=num_classes,
                                         kernel_size=3)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config.decoder_channels[-1], 2)
        )

    def forward(self, x):
        if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)
        x, _, features = self.transformer(x)
        x = self.decoder(x, features)

        seg_logits = self.seg_head(x)
        cls_logits = self.cls_head(x)
        return seg_logits, cls_logits

# 데이터 로더 및 유틸리티

# ==============================================================================
# 데이터셋
# ==============================================================================
class AbdominalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=224, label_mapping=None, augment=False):
        self.image_files = sorted(
            glob.glob(os.path.join(image_dir, "*.png"))
            + glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.jpeg"))
        )
        self.mask_files = sorted(
            glob.glob(os.path.join(mask_dir, "*.png"))
            + glob.glob(os.path.join(mask_dir, "*.jpg"))
            + glob.glob(os.path.join(mask_dir, "*.jpeg"))
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"이미지가 없습니다: {image_dir}")
        if len(self.mask_files) == 0:
            raise RuntimeError(f"마스크가 없습니다: {mask_dir}")

        # 단순 정렬 후 자르는 방식은 파일명이 정확히 대응한다는 전제입니다.
        min_len = min(len(self.image_files), len(self.mask_files))
        self.image_files = self.image_files[:min_len]
        self.mask_files = self.mask_files[:min_len]

        self.img_size = img_size
        self.label_mapping = label_mapping
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {self.image_files[idx]}")
        if mask is None:
            raise FileNotFoundError(f"마스크를 읽을 수 없습니다: {self.mask_files[idx]}")

        if self.label_mapping is not None:
            mapped_mask = np.zeros_like(mask, dtype=np.uint8)
            for old_value, new_value in self.label_mapping.items():
                mapped_mask[mask == old_value] = new_value
            mask = mapped_mask

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 영상과 마스크에 동일한 공간 증강 적용
        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.2:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            if random.random() < 0.3:
                angle = random.uniform(-10.0, 10.0)
                center = (self.img_size / 2.0, self.img_size / 2.0)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, matrix, (self.img_size, self.img_size),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                mask = cv2.warpAffine(
                    mask, matrix, (self.img_size, self.img_size),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            if random.random() < 0.3:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-10.0, 10.0)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        image = np.repeat(image[..., None], 3, axis=2)
        mask = np.ascontiguousarray(mask, dtype=np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


# ==============================================================================
# Loss: Weighted Cross Entropy + Soft Dice
# ==============================================================================
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, exclude_background=True, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.exclude_background = exclude_background
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()

        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=dims)
        denominator = torch.sum(probabilities, dim=dims) + torch.sum(one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # 현재 배치 정답에 존재하는 클래스만 Dice 평균에 포함
        present = torch.sum(one_hot, dim=dims) > 0
        if self.exclude_background:
            present[0] = False

        if present.any():
            return 1.0 - dice[present].mean()
        return logits.sum() * 0.0


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, class_weights, num_classes, ce_ratio=0.6, dice_ratio=0.4, ignore_index=255):
        super().__init__()
        self.ce = CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            exclude_background=True
        )
        self.ce_ratio = ce_ratio
        self.dice_ratio = dice_ratio

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.ce_ratio * ce_loss + self.dice_ratio * dice_loss
        return total, ce_loss.detach(), dice_loss.detach()


# ==============================================================================
# mIoU 유틸리티
# ==============================================================================
def fast_hist(gt, pred, num_classes):
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


def compute_metrics(hist):
    total = hist.sum()
    pixel_acc = np.diag(hist).sum() / total if total > 0 else 0.0
    denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_iou = np.diag(hist) / denominator

    mean_iou = np.nanmean(class_iou) if np.any(~np.isnan(class_iou)) else 0.0
    foreground_iou = class_iou[1:]
    foreground_miou = (
        np.nanmean(foreground_iou)
        if np.any(~np.isnan(foreground_iou))
        else 0.0
    )
    return pixel_acc, mean_iou, foreground_miou, class_iou


def evaluate_model(model, loader, seg_criterion, cls_criterion, device, num_classes):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    seg_loss_sum = 0.0
    cls_loss_sum = 0.0
    cls_correct = 0
    image_count = 0
    batch_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            seg_logits, cls_logits = model(inputs)
            seg_loss, _, _ = seg_criterion(seg_logits, targets)

            cls_targets = (
                ((targets == 9) | (targets == 10))
                .view(targets.size(0), -1)
                .any(dim=1)
                .long()
            )
            cls_loss = cls_criterion(cls_logits, cls_targets)

            predictions = torch.argmax(seg_logits, dim=1)
            cls_predictions = torch.argmax(cls_logits, dim=1)

            seg_loss_sum += seg_loss.item()
            cls_loss_sum += cls_loss.item()
            cls_correct += (cls_predictions == cls_targets).sum().item()
            image_count += targets.size(0)
            batch_count += 1

            targets_np = targets.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            for gt_mask, pred_mask in zip(targets_np, predictions_np):
                hist += fast_hist(gt_mask, pred_mask, num_classes)

    pixel_acc, mean_iou, foreground_miou, class_iou = compute_metrics(hist)
    return {
        "seg_loss": seg_loss_sum / max(batch_count, 1),
        "cls_loss": cls_loss_sum / max(batch_count, 1),
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "foreground_miou": foreground_miou,
        "class_iou": class_iou,
        "cls_accuracy": cls_correct / max(image_count, 1),
        "num_images": image_count,
        "hist": hist,
    }


def split_indices(length, train_ratio, seed):
    """
    PyTorch random_split()과 동일한 방식으로 인덱스를 생성합니다.

    test 코드가 아래 방식이라면:
        random_split(dataset, [train_size, test_size],
                     generator=torch.Generator().manual_seed(seed))

    이 함수도 내부적으로 torch.randperm()을 동일한 generator로 사용하므로
    같은 데이터셋 길이와 같은 seed일 때 완전히 동일한 90%/10% 분할이 재현됩니다.
    """
    train_size = int(length * train_ratio)

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(
        length,
        generator=generator
    ).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    return train_indices, val_indices


def make_balanced_dataset_sampler(train_subsets):
    """
    Synapse/LiTS/KiTS 각각의 총 샘플링 확률이 비슷해지도록 구성합니다.
    LiTS 데이터 수가 압도적으로 많아 다른 장기를 덮어버리는 현상을 완화합니다.
    """
    sample_weights = []
    for subset in train_subsets:
        if len(subset) == 0:
            continue
        per_sample_weight = 1.0 / len(subset)
        sample_weights.extend([per_sample_weight] * len(subset))

    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def print_validation_result(name, metrics):
    print(
        f"  [{name}] Loss={metrics['seg_loss']:.4f} | "
        f"PixelAcc={metrics['pixel_acc'] * 100:.2f}% | "
        f"mIoU={metrics['mean_iou'] * 100:.2f}% | "
        f"FG-mIoU={metrics['foreground_miou'] * 100:.2f}% | "
        f"TumorAcc={metrics['cls_accuracy'] * 100:.2f}%"
    )


# ==============================================================================
# 학습 실행부
# ==============================================================================
if __name__ == "__main__":
    print("🚀 클래스 불균형 개선 TransUNet 학습 시작")

    # --------------------------------------------------------------------------
    # 1. 재현성 및 기본 설정
    # --------------------------------------------------------------------------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224
    NUM_CLASSES = 11
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    EPOCHS = 100
    TRAIN_RATIO = 0.9
    INITIAL_LR = 0.01
    WEIGHT_DECAY = 1e-4
    CLASSIFICATION_LOSS_WEIGHT = 0.3
    PATIENCE = 20

    SAVE_DIR = "./transunet_training_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, "transunet_multitask_best_miou.pth")
    FINAL_WEIGHT_PATH = "transunet_multitask_weights.pth"
    FULL_MODEL_PATH = "transunet_multitask_full.pth"
    HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.csv")

    # 기존 가중치에서 이어서 학습할 때만 경로 지정
    RESUME_PATH = None
    # 예: RESUME_PATH = r"./transunet_multitask_weights.pth"

    # --------------------------------------------------------------------------
    # 2. 데이터 경로
    # --------------------------------------------------------------------------
    SYNAPSE_RAW_TRAIN_IMG = r"C:\Users\dongj\Desktop\Abdomen\RawData\Training\img"
    SYNAPSE_RAW_TRAIN_LBL = r"C:\Users\dongj\Desktop\Abdomen\RawData\Training\label"
    PROCESSED_SYNAPSE_TRAIN = r"C:\Users\dongj\Desktop\Abdomen\Processed_2D\Train"

    LITS_IMAGE_DIR = r"C:\Users\dongj\Desktop\LITS\train_images\train_images"
    LITS_MASK_DIR = r"C:\Users\dongj\Desktop\LITS\train_masks\train_masks"

    KITS_IMAGE_DIR = r"C:\Users\dongj\Desktop\KITS23\NOT-AUGMENTED\DATASET_FINAL\JPEGImages"
    KITS_MASK_DIR = r"C:\Users\dongj\Desktop\KITS23\NOT-AUGMENTED\DATASET_FINAL\Annotations"

    LITS_MAPPING = {1: 5, 2: 9}
    KITS_MAPPING = {1: 3, 2: 10}

    if os.path.isdir(SYNAPSE_RAW_TRAIN_IMG):
        preprocess_3d_to_2d(
            SYNAPSE_RAW_TRAIN_IMG,
            SYNAPSE_RAW_TRAIN_LBL,
            PROCESSED_SYNAPSE_TRAIN,
            is_test=False
        )

    # --------------------------------------------------------------------------
    # 3. 학습/검증 데이터셋 생성
    #    test 코드의 random_split(seed=42)과 완전히 같은 방식으로 9:1 분할
    #    같은 인덱스를 사용하되 train에는 증강, validation에는 증강 없음
    # --------------------------------------------------------------------------
    train_subsets = []
    val_subsets = []
    val_loaders_by_name = {}
    dataset_summary = []

    dataset_specs = [
        (
            "Synapse",
            os.path.join(PROCESSED_SYNAPSE_TRAIN, "images"),
            os.path.join(PROCESSED_SYNAPSE_TRAIN, "labels"),
            None,
        ),
        ("LiTS", LITS_IMAGE_DIR, LITS_MASK_DIR, LITS_MAPPING),
        ("KiTS", KITS_IMAGE_DIR, KITS_MASK_DIR, KITS_MAPPING),
    ]

    for name, image_dir, mask_dir, mapping in dataset_specs:
        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            print(f"⚠️ {name} 경로가 없어 제외합니다.")
            continue

        train_base = AbdominalDataset(
            image_dir, mask_dir, img_size=IMG_SIZE,
            label_mapping=mapping, augment=True
        )
        val_base = AbdominalDataset(
            image_dir, mask_dir, img_size=IMG_SIZE,
            label_mapping=mapping, augment=False
        )

        train_indices, val_indices = split_indices(
            len(train_base), TRAIN_RATIO, SEED
        )
        train_subset = Subset(train_base, train_indices)
        val_subset = Subset(val_base, val_indices)

        train_subsets.append(train_subset)
        val_subsets.append(val_subset)
        dataset_summary.append((name, len(train_subset), len(val_subset)))

        val_loaders_by_name[name] = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

    if not train_subsets:
        raise RuntimeError("사용 가능한 학습 데이터셋이 없습니다. 경로를 확인하세요.")

    full_train_dataset = ConcatDataset(train_subsets)
    full_val_dataset = ConcatDataset(val_subsets)

    # 데이터셋별 총 샘플링 비율을 동일하게 만들어 LiTS 편향 완화
    train_sampler = make_balanced_dataset_sampler(train_subsets)

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        full_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("\n📁 데이터 구성")
    for name, train_count, val_count in dataset_summary:
        print(f"- {name}: Train {train_count}장 | Validation {val_count}장")
    print(f"- 전체 Train: {len(full_train_dataset)}장")
    print(f"- 전체 Validation: {len(full_val_dataset)}장")
    print("- 분할 방식: random_split과 동일한 torch.randperm 방식")
    print(f"- 분할 seed: {SEED} (모든 데이터셋 동일)")
    print("- 학습 샘플러: Synapse/LiTS/KiTS 균형 샘플링")

    # --------------------------------------------------------------------------
    # 4. 모델, Loss, Optimizer, Scheduler
    # --------------------------------------------------------------------------
    config_vit = CONFIGS["R50-ViT-B_16"]
    model = VisionTransformerMultiTask(
        config_vit,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    if RESUME_PATH is not None and os.path.isfile(RESUME_PATH):
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        cleaned = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
        model.load_state_dict(cleaned, strict=True)
        print(f"✅ 기존 가중치에서 이어서 학습: {RESUME_PATH}")

    # 배경 영향은 줄이고, 기존에 0%였던 장기 및 종양 클래스 가중치 강화
    class_weights = torch.tensor(
        [
            0.20,  # 0 배경
            3.00,  # 1 대동맥
            4.00,  # 2 담낭
            3.00,  # 3 좌측 신장
            3.00,  # 4 우측 신장
            1.00,  # 5 간
            4.00,  # 6 췌장
            3.00,  # 7 비장
            3.00,  # 8 위
            6.00,  # 9 간 종양
            6.00,  # 10 신장 종양
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    seg_criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        num_classes=NUM_CLASSES,
        ce_ratio=0.6,
        dice_ratio=0.4,
        ignore_index=255
    )
    cls_criterion = CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=INITIAL_LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    # 검증 foreground mIoU가 정체되면 학습률 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # --------------------------------------------------------------------------
    # 5. 학습
    # --------------------------------------------------------------------------
    best_fg_miou = -1.0
    epochs_without_improvement = 0

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8-sig") as history_file:
        writer = csv.writer(history_file)
        writer.writerow([
            "epoch", "lr", "train_total_loss", "train_seg_loss", "train_cls_loss",
            "val_seg_loss", "val_pixel_acc", "val_miou", "val_fg_miou",
            "val_tumor_acc"
        ])

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss_sum = 0.0
            seg_loss_sum = 0.0
            cls_loss_sum = 0.0
            ce_loss_sum = 0.0
            dice_loss_sum = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                cls_targets = (
                    ((targets == 9) | (targets == 10))
                    .view(targets.size(0), -1)
                    .any(dim=1)
                    .long()
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    seg_logits, cls_logits = model(inputs)
                    seg_loss, ce_loss, dice_loss = seg_criterion(seg_logits, targets)
                    cls_loss = cls_criterion(cls_logits, cls_targets)
                    total_loss = seg_loss + CLASSIFICATION_LOSS_WEIGHT * cls_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss_sum += total_loss.item()
                seg_loss_sum += seg_loss.item()
                cls_loss_sum += cls_loss.item()
                ce_loss_sum += ce_loss.item()
                dice_loss_sum += dice_loss.item()

                if batch_idx % 100 == 0:
                    print(
                        f"[Epoch {epoch:03d}/{EPOCHS}] "
                        f"Batch {batch_idx:05d}/{len(train_loader)} | "
                        f"Total={total_loss.item():.4f} | "
                        f"CE={ce_loss.item():.4f} | Dice={dice_loss.item():.4f}"
                    )

            num_train_batches = max(len(train_loader), 1)
            train_total_loss = total_loss_sum / num_train_batches
            train_seg_loss = seg_loss_sum / num_train_batches
            train_cls_loss = cls_loss_sum / num_train_batches

            # 전체 validation mIoU
            val_metrics = evaluate_model(
                model, val_loader, seg_criterion, cls_criterion,
                DEVICE, NUM_CLASSES
            )

            scheduler.step(val_metrics["foreground_miou"])
            current_lr = optimizer.param_groups[0]["lr"]

            print("\n" + "=" * 85)
            print(f"Epoch {epoch}/{EPOCHS} 완료 | LR={current_lr:.7f}")
            print(
                f"Train Total={train_total_loss:.4f} | "
                f"Seg={train_seg_loss:.4f} | Cls={train_cls_loss:.4f} | "
                f"CE={ce_loss_sum / num_train_batches:.4f} | "
                f"Dice={dice_loss_sum / num_train_batches:.4f}"
            )
            print_validation_result("전체 Validation", val_metrics)

            # 데이터셋별 결과를 따로 확인해야 어떤 클래스가 안 잡히는지 알 수 있음
            for dataset_name, dataset_loader in val_loaders_by_name.items():
                dataset_metrics = evaluate_model(
                    model, dataset_loader, seg_criterion, cls_criterion,
                    DEVICE, NUM_CLASSES
                )
                print_validation_result(dataset_name, dataset_metrics)

            print("\n[전체 Validation Class-wise IoU]")
            for class_idx, iou in enumerate(val_metrics["class_iou"]):
                iou_text = "N/A" if np.isnan(iou) else f"{iou * 100:.2f}%"
                print(f"- {class_idx:2d} {ORGAN_NAMES[class_idx]:<24}: {iou_text}")
            print("=" * 85 + "\n")

            writer.writerow([
                epoch,
                current_lr,
                train_total_loss,
                train_seg_loss,
                train_cls_loss,
                val_metrics["seg_loss"],
                val_metrics["pixel_acc"],
                val_metrics["mean_iou"],
                val_metrics["foreground_miou"],
                val_metrics["cls_accuracy"],
            ])
            history_file.flush()

            # 배경을 제외한 foreground mIoU 기준으로 최고 모델 저장
            if val_metrics["foreground_miou"] > best_fg_miou:
                best_fg_miou = val_metrics["foreground_miou"]
                epochs_without_improvement = 0

                torch.save(model.state_dict(), BEST_WEIGHT_PATH)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_foreground_miou": best_fg_miou,
                    "val_mean_iou": val_metrics["mean_iou"],
                    "class_iou": val_metrics["class_iou"],
                }, os.path.join(SAVE_DIR, "best_checkpoint.pth"))

                print(
                    f"💾 최고 모델 저장: {BEST_WEIGHT_PATH} | "
                    f"FG-mIoU={best_fg_miou * 100:.2f}%"
                )
            else:
                epochs_without_improvement += 1
                print(
                    f"성능 개선 없음: {epochs_without_improvement}/{PATIENCE} | "
                    f"현재 최고 FG-mIoU={best_fg_miou * 100:.2f}%"
                )

            # 매 epoch 최신 가중치 저장
            torch.save(model.state_dict(), FINAL_WEIGHT_PATH)

            if epochs_without_improvement >= PATIENCE:
                print(f"⏹️ Early stopping: {PATIENCE} epoch 동안 FG-mIoU 개선 없음")
                break

    # --------------------------------------------------------------------------
    # 6. 최종 저장
    # --------------------------------------------------------------------------
    torch.save(model.state_dict(), FINAL_WEIGHT_PATH)
    torch.save(model, FULL_MODEL_PATH)

    print("\n✅ 학습 종료")
    print(f"- 마지막 가중치: {FINAL_WEIGHT_PATH}")
    print(f"- 최고 mIoU 가중치: {BEST_WEIGHT_PATH}")
    print(f"- 전체 모델: {FULL_MODEL_PATH}")
    print(f"- 학습 기록: {HISTORY_PATH}")
    print(f"- 최고 Foreground mIoU: {best_fg_miou * 100:.2f}%")