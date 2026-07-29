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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch.optim as optim
import ml_collections

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
class AbdominalDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, img_size=224, label_mapping=None):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                                  glob.glob(os.path.join(image_dir, "*.jpg")) +
                                  glob.glob(os.path.join(image_dir, "*.jpeg")))

        # Test 데이터의 경우 정답지(mask_dir)가 없을 수 있음
        if mask_dir is not None and os.path.exists(mask_dir):
            self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")) +
                                     glob.glob(os.path.join(mask_dir, "*.jpg")) +
                                     glob.glob(os.path.join(mask_dir, "*.jpeg")))
            min_len = min(len(self.image_files), len(self.mask_files))
            self.image_files = self.image_files[:min_len]
            self.mask_files = self.mask_files[:min_len]
        else:
            self.mask_files = []

        self.img_size = img_size
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # 정답지 파일이 있으면 읽고, 없으면 모두 0(배경)인 빈 텐서 생성
        has_mask = len(self.mask_files) > 0

        if has_mask:
            mask_path = self.mask_files[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise FileNotFoundError(f"마스크를 읽을 수 없습니다: {mask_path}")

            if self.label_mapping is not None:
                mapped_mask = np.zeros_like(mask)
                for old_val, new_val in self.label_mapping.items():
                    mapped_mask[mask == old_val] = new_val
                mask = mapped_mask

            mask = cv2.resize(
                mask,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST
            )
            mask = torch.from_numpy(mask).long()
        else:
            mask = torch.zeros((self.img_size, self.img_size), dtype=torch.long)

        # has_mask를 함께 반환해야 라벨 없는 Synapse Test를 mIoU 계산에서 제외할 수 있음
        return image, mask, has_mask

# 데이터셋별 테스트 실행부
if __name__ == "__main__":
    print("🚀 TransUNet 데이터셋별 테스트 시작")

    # 1. 기본 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224
    NUM_CLASSES = 11
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    SPLIT_SEED = 42

    # 학습 중 매 epoch 저장된 가중치
    WEIGHT_PATH = r"./transunet_multitask_weights.pth"

    # 최상위 결과 폴더
    RESULT_ROOT = r"./transunet_test_results_by_dataset"
    os.makedirs(RESULT_ROOT, exist_ok=True)

    # 2. 데이터 경로
    PROCESSED_SYNAPSE_TEST = r"C:\Users\dongj\Desktop\Abdomen\Processed_2D\Test"

    LITS_IMAGE_DIR = r"C:\Users\dongj\Desktop\LITS\train_images\train_images"
    LITS_MASK_DIR = r"C:\Users\dongj\Desktop\LITS\train_masks\train_masks"

    KITS_IMAGE_DIR = r"C:\Users\dongj\Desktop\KITS23\NOT-AUGMENTED\DATASET_FINAL\JPEGImages"
    KITS_MASK_DIR = r"C:\Users\dongj\Desktop\KITS23\NOT-AUGMENTED\DATASET_FINAL\Annotations"

    LITS_MAPPING = {1: 5, 2: 9}
    KITS_MAPPING = {1: 3, 2: 10}

    # 데이터셋별 실제 평가 대상 클래스
    DATASET_CLASS_IDS = {
        "Synapse": list(range(0, 9)),
        "LiTS": [0, 5, 9],
        "KiTS": [0, 3, 10],
    }

    # 3. 유틸리티
    def make_test_subset(dataset, train_ratio=0.9, seed=42):
        """
        전체 데이터셋을 고정 seed로 9:1 분할하고 test subset을 반환합니다.

        주의:
        학습 코드에서도 동일한 random_split 방식과 동일한 seed를
        사용했을 때만 완전히 같은 test subset이 재현됩니다.
        """
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(seed)
        _, test_subset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )
        return test_subset

    def load_state_dict_safely(model, weight_path, device):
        """state_dict/checkpoint 형식을 모두 지원하여 가중치를 불러옵니다."""
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"가중치 파일을 찾을 수 없습니다: {weight_path}\n"
                "WEIGHT_PATH를 실제 파일 위치로 수정하세요."
            )

        try:
            checkpoint = torch.load(
                weight_path,
                map_location=device,
                weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(weight_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned_state_dict[cleaned_key] = value

        model.load_state_dict(cleaned_state_dict, strict=True)
        print(f"✅ 가중치 로드 완료: {weight_path}")


    def calculate_selected_miou(class_iou, selected_class_ids):
        """
        해당 데이터셋에서 실제 의미가 있는 클래스만 골라 mIoU를 계산합니다.
        NaN 클래스는 평균에서 제외합니다.
        """
        selected_values = np.asarray(
            [class_iou[class_id] for class_id in selected_class_ids],
            dtype=np.float64
        )
        valid_values = selected_values[~np.isnan(selected_values)]

        if len(valid_values) == 0:
            return float("nan")

        return float(np.mean(valid_values))

    def save_prediction_figure(
        image_tensor,
        true_mask,
        pred_mask,
        pred_class_label,
        has_mask,
        save_path,
        sample_title
    ):
        """원본, 정답, 예측을 하나의 이미지로 저장합니다."""
        custom_cmap = ListedColormap(ORGAN_COLORS_HEX)
        orig_img = image_tensor[0].cpu().numpy()

        unique_classes = np.unique(pred_mask)
        detected_organs = [
            ORGAN_NAMES[int(class_id)]
            for class_id in unique_classes
            if int(class_id) != 0 and int(class_id) in ORGAN_NAMES
        ]

        cls_text = "종양 발견됨" if pred_class_label == 1 else "정상 소견"
        seg_text = (
            ", ".join(detected_organs)
            if detected_organs
            else "감지된 장기 없음"
        )

        fig = plt.figure(figsize=(18, 6))
        plt.suptitle(sample_title, fontsize=16, fontweight="bold")

        plt.subplot(1, 3, 1)
        plt.title("원본 CT")
        plt.imshow(orig_img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("실제 정답" if has_mask else "실제 정답 없음")
        plt.imshow(
            true_mask,
            cmap=custom_cmap,
            vmin=0,
            vmax=NUM_CLASSES - 1
        )
        plt.axis("off")

        ax = plt.subplot(1, 3, 3)
        plt.title("TransUNet 예측")
        ax.imshow(
            pred_mask,
            cmap=custom_cmap,
            vmin=0,
            vmax=NUM_CLASSES - 1
        )
        ax.axis("off")

        patches = [
            mpatches.Patch(
                color=ORGAN_COLORS_HEX[class_id],
                label=f"{class_id}: {ORGAN_NAMES[class_id]}"
            )
            for class_id in range(NUM_CLASSES)
        ]
        ax.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            fontsize=9,
            title="클래스"
        )

        label_note = "" if has_mask else " / 정답 라벨 없음"
        plt.figtext(
            0.5,
            0.04,
            f"분류: {cls_text} | 분할: {seg_text}{label_note}",
            ha="center",
            fontsize=13,
            fontweight="bold"
        )

        plt.subplots_adjust(bottom=0.15, right=0.82)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    def test_one_dataset(
        dataset_name,
        dataset,
        model,
        seg_criterion,
        device,
        class_ids,
        has_ground_truth
    ):
        """
        하나의 데이터셋만 독립적으로 테스트합니다.

        저장 구조:
        RESULT_ROOT/
        └─ dataset_name/
           ├─ predictions/
           ├─ test_metrics.txt
           └─ confusion_matrix.npy
        """
        dataset_dir = os.path.join(RESULT_ROOT, dataset_name)
        prediction_dir = os.path.join(dataset_dir, "predictions")

        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(prediction_dir, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        total_hist = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.float64
        )
        total_seg_loss = 0.0
        total_cls_correct = 0
        total_cls_count = 0
        labeled_batch_count = 0
        labeled_image_count = 0
        saved_count = 0

        print("\n" + "=" * 75)
        print(f"📁 {dataset_name} 테스트 시작")
        print(f"- 이미지 수: {len(dataset)}장")
        print(f"- 정답 라벨: {'있음' if has_ground_truth else '없음'}")
        print(f"- 결과 폴더: {dataset_dir}")
        print("=" * 75)

        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets, has_mask) in enumerate(loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                has_mask = torch.as_tensor(has_mask, dtype=torch.bool)

                seg_logits, cls_logits = model(inputs)
                pred_masks = torch.argmax(seg_logits, dim=1)
                pred_classes = torch.argmax(cls_logits, dim=1)

                if has_mask.any():
                    labeled_indices = has_mask.to(device)

                    labeled_seg_logits = seg_logits[labeled_indices]
                    labeled_targets = targets[labeled_indices]
                    labeled_pred_masks = pred_masks[labeled_indices]
                    labeled_pred_classes = pred_classes[labeled_indices]

                    seg_loss = seg_criterion(
                        labeled_seg_logits,
                        labeled_targets
                    )
                    total_seg_loss += seg_loss.item()
                    labeled_batch_count += 1
                    labeled_image_count += labeled_targets.size(0)

                    true_classes = (
                        (
                            (labeled_targets == 9)
                            | (labeled_targets == 10)
                        )
                        .view(labeled_targets.size(0), -1)
                        .any(dim=1)
                        .long()
                    )

                    total_cls_correct += (
                        labeled_pred_classes == true_classes
                    ).sum().item()
                    total_cls_count += true_classes.numel()

                    targets_np = labeled_targets.cpu().numpy()
                    predictions_np = labeled_pred_masks.cpu().numpy()

                    for gt_mask, pred_mask in zip(
                        targets_np,
                        predictions_np
                    ):
                        total_hist += fast_hist(
                            gt_mask,
                            pred_mask,
                            NUM_CLASSES
                        )

                # 각 데이터셋 폴더에 예측 시각화 저장
                for sample_idx in range(inputs.size(0)):
                    sample_number = (
                        batch_idx * BATCH_SIZE + sample_idx
                    )
                    sample_has_mask = bool(
                        has_mask[sample_idx].item()
                    )

                    true_mask_np = (
                        targets[sample_idx].cpu().numpy()
                    )
                    pred_mask_np = (
                        pred_masks[sample_idx].cpu().numpy()
                    )
                    pred_class_label = int(
                        pred_classes[sample_idx].item()
                    )

                    save_path = os.path.join(
                        prediction_dir,
                        f"{dataset_name.lower()}_{sample_number:05d}.png"
                    )

                    save_prediction_figure(
                        image_tensor=inputs[sample_idx].cpu(),
                        true_mask=true_mask_np,
                        pred_mask=pred_mask_np,
                        pred_class_label=pred_class_label,
                        has_mask=sample_has_mask,
                        save_path=save_path,
                        sample_title=(
                            f"{dataset_name} Test Result "
                            f"#{sample_number}"
                        )
                    )
                    saved_count += 1

                if (batch_idx + 1) % 20 == 0:
                    processed_count = min(
                        (batch_idx + 1) * BATCH_SIZE,
                        len(dataset)
                    )
                    print(
                        f"[{dataset_name}] 진행: "
                        f"{processed_count}/{len(dataset)}"
                    )

        summary_path = os.path.join(
            dataset_dir,
            "test_summary.txt"
        )

        # 라벨 없는 Synapse Test는 예측 저장 결과만 기록
        if labeled_image_count == 0:
            with open(summary_path, "w", encoding="utf-8") as file:
                file.write(f"{dataset_name} Test Summary\n")
                file.write("=" * 50 + "\n")
                file.write(f"Weight: {WEIGHT_PATH}\n")
                file.write(f"Images: {len(dataset)}\n")
                file.write("Ground truth: 없음\n")
                file.write(
                    "mIoU/Loss는 정답 마스크가 없어 계산하지 않음\n"
                )
                file.write(f"Saved predictions: {saved_count}\n")

            print(f"\n📊 {dataset_name} 결과")
            print("- 정답 마스크가 없어 Loss/mIoU 계산 제외")
            print(f"- 저장된 예측 이미지: {saved_count}장")
            print(f"- 요약 파일: {summary_path}")

            return {
                "dataset": dataset_name,
                "num_images": len(dataset),
                "num_labeled_images": 0,
                "saved_count": saved_count,
            }

        pixel_acc, all_class_miou, class_iou = compute_metrics(
            total_hist
        )

        dataset_miou = calculate_selected_miou(
            class_iou,
            class_ids
        )

        foreground_ids = [
            class_id
            for class_id in class_ids
            if class_id != 0
        ]
        foreground_miou = calculate_selected_miou(
            class_iou,
            foreground_ids
        )

        avg_seg_loss = (
            total_seg_loss / labeled_batch_count
            if labeled_batch_count > 0
            else float("nan")
        )
        cls_accuracy = (
            total_cls_correct / total_cls_count
            if total_cls_count > 0
            else float("nan")
        )

        print("\n" + "-" * 75)
        print(f"📊 {dataset_name} 최종 테스트 결과")
        print("-" * 75)
        print(f"전체 이미지             : {len(dataset)}장")
        print(f"라벨 있는 이미지        : {labeled_image_count}장")
        print(f"저장된 결과 이미지      : {saved_count}장")
        print(f"평균 Segmentation Loss  : {avg_seg_loss:.6f}")
        print(f"Pixel Accuracy          : {pixel_acc * 100:.2f}%")
        print(
            f"{dataset_name} mIoU"
            f"          : {dataset_miou * 100:.2f}%"
        )
        print(
            f"{dataset_name} FG-mIoU"
            f"       : {foreground_miou * 100:.2f}%"
        )
        print(f"11-class 전체 mIoU      : {all_class_miou * 100:.2f}%")
        print(f"Tumor Classification Acc: {cls_accuracy * 100:.2f}%")

        print(f"\n[{dataset_name} 관련 클래스 IoU]")
        for class_id in class_ids:
            iou = class_iou[class_id]
            iou_text = (
                "N/A"
                if np.isnan(iou)
                else f"{iou * 100:.2f}%"
            )
            print(
                f"- {class_id:2d} "
                f"{ORGAN_NAMES[class_id]:<24}: {iou_text}"
            )

        metrics_path = os.path.join(
            dataset_dir,
            "test_metrics.txt"
        )

        with open(metrics_path, "w", encoding="utf-8") as file:
            file.write(f"{dataset_name} Test Metrics\n")
            file.write("=" * 60 + "\n")
            file.write(f"Weight: {WEIGHT_PATH}\n")
            file.write(f"Images: {len(dataset)}\n")
            file.write(
                f"Labeled images: {labeled_image_count}\n"
            )
            file.write(
                f"Segmentation Loss: {avg_seg_loss:.6f}\n"
            )
            file.write(
                f"Pixel Accuracy: {pixel_acc * 100:.2f}%\n"
            )
            file.write(
                f"{dataset_name} mIoU: "
                f"{dataset_miou * 100:.2f}%\n"
            )
            file.write(
                f"{dataset_name} Foreground mIoU: "
                f"{foreground_miou * 100:.2f}%\n"
            )
            file.write(
                f"11-class mIoU: "
                f"{all_class_miou * 100:.2f}%\n"
            )
            file.write(
                f"Tumor Classification Accuracy: "
                f"{cls_accuracy * 100:.2f}%\n\n"
            )

            file.write(
                f"{dataset_name} relevant class IoU\n"
            )
            for class_id in class_ids:
                iou = class_iou[class_id]
                iou_text = (
                    "N/A"
                    if np.isnan(iou)
                    else f"{iou * 100:.2f}%"
                )
                file.write(
                    f"{class_id}: "
                    f"{ORGAN_NAMES[class_id]} = {iou_text}\n"
                )

            file.write("\nAll 11 class IoU\n")
            for class_id, iou in enumerate(class_iou):
                iou_text = (
                    "N/A"
                    if np.isnan(iou)
                    else f"{iou * 100:.2f}%"
                )
                file.write(
                    f"{class_id}: "
                    f"{ORGAN_NAMES[class_id]} = {iou_text}\n"
                )

        np.save(
            os.path.join(dataset_dir, "confusion_matrix.npy"),
            total_hist
        )

        print(f"\n✅ 지표 저장: {metrics_path}")
        print(f"✅ 예측 저장: {prediction_dir}")

        return {
            "dataset": dataset_name,
            "num_images": len(dataset),
            "num_labeled_images": labeled_image_count,
            "seg_loss": avg_seg_loss,
            "pixel_acc": pixel_acc,
            "dataset_miou": dataset_miou,
            "foreground_miou": foreground_miou,
            "all_class_miou": all_class_miou,
            "cls_accuracy": cls_accuracy,
            "saved_count": saved_count,
        }

    # 4. 모델 생성 및 가중치 로드
    config_vit = CONFIGS["R50-ViT-B_16"]

    model = VisionTransformerMultiTask(
        config_vit,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    load_state_dict_safely(
        model,
        WEIGHT_PATH,
        DEVICE
    )
    model.eval()

    # 학습 코드와 동일하게 맞추려면 해당 학습 가중치로 수정
    class_weights = torch.tensor(
        [
            0.20,
            3.00,
            4.00,
            3.00,
            3.00,
            1.00,
            4.00,
            3.00,
            3.00,
            6.00,
            6.00,
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    seg_criterion = CrossEntropyLoss(
        weight=class_weights,
        ignore_index=255
    )

    # 5. 각 데이터셋 독립 구성
    datasets_to_test = []

    # Synapse: 현재 경로는 라벨 없는 test로 가정
    synapse_image_dir = os.path.join(
        PROCESSED_SYNAPSE_TEST,
        "images"
    )

    if os.path.isdir(synapse_image_dir):
        synapse_dataset = AbdominalDataset(
            synapse_image_dir,
            mask_dir=None,
            img_size=IMG_SIZE
        )

        if len(synapse_dataset) > 0:
            datasets_to_test.append(
                (
                    "Synapse",
                    synapse_dataset,
                    False
                )
            )
    else:
        print(
            f"⚠️ Synapse 이미지 폴더가 없습니다: "
            f"{synapse_image_dir}"
        )

    # LiTS: 전체 데이터에서 고정 10% test subset
    if (
        os.path.isdir(LITS_IMAGE_DIR)
        and os.path.isdir(LITS_MASK_DIR)
    ):
        lits_full_dataset = AbdominalDataset(
            LITS_IMAGE_DIR,
            LITS_MASK_DIR,
            img_size=IMG_SIZE,
            label_mapping=LITS_MAPPING
        )

        if len(lits_full_dataset) > 0:
            lits_test_dataset = make_test_subset(
                lits_full_dataset,
                train_ratio=0.9,
                seed=SPLIT_SEED
            )

            datasets_to_test.append(
                (
                    "LiTS",
                    lits_test_dataset,
                    True
                )
            )
    else:
        print("⚠️ LiTS 데이터 경로가 없어 제외합니다.")

    # KiTS: 전체 데이터에서 고정 10% test subset
    if (
        os.path.isdir(KITS_IMAGE_DIR)
        and os.path.isdir(KITS_MASK_DIR)
    ):
        kits_full_dataset = AbdominalDataset(
            KITS_IMAGE_DIR,
            KITS_MASK_DIR,
            img_size=IMG_SIZE,
            label_mapping=KITS_MAPPING
        )

        if len(kits_full_dataset) > 0:
            kits_test_dataset = make_test_subset(
                kits_full_dataset,
                train_ratio=0.9,
                seed=SPLIT_SEED
            )

            datasets_to_test.append(
                (
                    "KiTS",
                    kits_test_dataset,
                    True
                )
            )
    else:
        print("⚠️ KiTS 데이터 경로가 없어 제외합니다.")

    if not datasets_to_test:
        raise RuntimeError(
            "테스트할 수 있는 데이터셋이 없습니다. "
            "데이터 경로를 확인하세요."
        )

    print("\n📁 데이터셋별 테스트 구성")
    for dataset_name, dataset, has_ground_truth in datasets_to_test:
        print(
            f"- {dataset_name}: {len(dataset)}장 | "
            f"라벨 {'있음' if has_ground_truth else '없음'}"
        )

    # 6. 데이터셋별 독립 테스트
    all_results = []

    for dataset_name, dataset, has_ground_truth in datasets_to_test:
        result = test_one_dataset(
            dataset_name=dataset_name,
            dataset=dataset,
            model=model,
            seg_criterion=seg_criterion,
            device=DEVICE,
            class_ids=DATASET_CLASS_IDS[dataset_name],
            has_ground_truth=has_ground_truth
        )
        all_results.append(result)
    # 7. 전체 데이터셋 요약 저장
    overall_summary_path = os.path.join(
        RESULT_ROOT,
        "all_dataset_summary.txt"
    )

    with open(
        overall_summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("TransUNet Dataset-wise Test Summary\n")
        file.write("=" * 70 + "\n")
        file.write(f"Weight: {WEIGHT_PATH}\n")
        file.write(f"Device: {DEVICE}\n\n")

        for result in all_results:
            dataset_name = result["dataset"]
            file.write(f"[{dataset_name}]\n")
            file.write(
                f"Images: {result['num_images']}\n"
            )
            file.write(
                f"Labeled images: "
                f"{result['num_labeled_images']}\n"
            )
            file.write(
                f"Saved predictions: "
                f"{result['saved_count']}\n"
            )

            if result["num_labeled_images"] > 0:
                file.write(
                    f"Segmentation Loss: "
                    f"{result['seg_loss']:.6f}\n"
                )
                file.write(
                    f"Pixel Accuracy: "
                    f"{result['pixel_acc'] * 100:.2f}%\n"
                )
                file.write(
                    f"Dataset mIoU: "
                    f"{result['dataset_miou'] * 100:.2f}%\n"
                )
                file.write(
                    f"Foreground mIoU: "
                    f"{result['foreground_miou'] * 100:.2f}%\n"
                )
                file.write(
                    f"Tumor Classification Accuracy: "
                    f"{result['cls_accuracy'] * 100:.2f}%\n"
                )
            else:
                file.write(
                    "Metrics: 정답 라벨이 없어 계산 제외\n"
                )

            file.write("\n")

    print("\n" + "=" * 75)
    print("✅ 모든 데이터셋 테스트 완료")
    print(f"✅ 최상위 결과 폴더: {RESULT_ROOT}")
    print(f"✅ 전체 요약: {overall_summary_path}")
    print("=" * 75)