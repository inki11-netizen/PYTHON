import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1) 경로 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"/home/aivs/바탕화면/aa/model_save/CBAM_Iter_250000_acc_73.0.pth"
TARGET_IMAGE_PATH = r"/home/aivs/바탕화면/dataset/tinyImageNet/test/000210.png"

# --- (옵션) imshow가 안 되는 환경이면 파일로 저장하도록 변경 ---
USE_IMSHOW = True              # 서버/VM에서 imshow 안 뜨면 False로
SAVE_OVERLAY_PATH = "overlay.png"  # USE_IMSHOW=False일 때 저장 파일명


# --- 2) 시각화 함수 (JET + 투명도 40%) ---
def make_attention_overlay(raw_img_bgr_uint8, spa_map, alpha_heatmap=0.4):
    """
    raw_img_bgr_uint8: uint8 (H,W,3) BGR
    spa_map: torch.Tensor (B,1,h,w) or (1,1,h,w)
    """
    spa = spa_map.detach().float().cpu().numpy().squeeze()  # (h,w)
    spa = (spa - spa.min()) / (spa.max() - spa.min() + 1e-8)
    spa_resized = cv2.resize(spa, (raw_img_bgr_uint8.shape[1], raw_img_bgr_uint8.shape[0]))

    heatmap = cv2.applyColorMap((spa_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_img_bgr_uint8, 1.0 - alpha_heatmap, heatmap, alpha_heatmap, 0)
    return overlay


def show_or_save_overlay(raw_img_bgr_uint8, spa_map, title="CBAM Attention View"):
    overlay = make_attention_overlay(raw_img_bgr_uint8, spa_map, alpha_heatmap=0.4)

    if USE_IMSHOW:
        cv2.imshow(title, overlay)
        print("이미지 확인 중... 아무 키나 누르면 종료됩니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(SAVE_OVERLAY_PATH, overlay)
        print(f"✅ overlay 저장 완료: {SAVE_OVERLAY_PATH}")


# --- 3) 모델 정의 (체크포인트 구조와 동일한 Attention_module + 시각화용 수정) ---
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

        # ResNet50 bottleneck 블록을 layer2~layer17로 펼쳐둔 형태
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

        # CBAM 구성요소(채널 MLP는 Conv1x1, spatial은 Conv7x7)
        def make_cbam_components(ch):
            mlp = nn.Sequential(
                nn.Conv2d(ch, ch // 16, 1),
                nn.ReLU(True),
                nn.Conv2d(ch // 16, ch, 1)
            )
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

        # shortcut용 1x1 conv (다운샘플링/채널확장)
        self.conv2 = nn.Conv2d(64, 256, 1, 1, 0)
        self.bn_conv2 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 1, 2, 0)
        self.bn_conv5 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1024, 1, 2, 0)
        self.bn_conv9 = nn.BatchNorm2d(1024)
        self.conv16 = nn.Conv2d(1024, 2048, 1, 2, 0)
        self.bn_conv16 = nn.BatchNorm2d(2048)

        # pool / head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avgpool_final = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Linear(2048, 200)
        self.sigmoid = nn.Sigmoid()

    def cbam(self, feat, mlp, spatial, return_sa=False):
        """
        return_sa=True면 spatial attention map(sa)을 같이 반환
        """
        # Channel Attention
        ca = self.sigmoid(mlp(self.avg_pool(feat)) + mlp(self.max_pool(feat)))
        feat = feat * ca

        # Spatial Attention
        sa_in = torch.cat(
            [torch.mean(feat, 1, keepdim=True),
             torch.max(feat, 1, keepdim=True)[0]],
            dim=1
        )
        sa = self.sigmoid(spatial(sa_in))  # (B,1,H,W)
        out = feat * sa

        if return_sa:
            return out, sa
        return out

    def forward(self, x, raw_img=None, visualize_at="layer17"):
        """
        raw_img: 시각화용 BGR uint8 (H,W,3). None이면 시각화 안 함.
        visualize_at: 'layer2'~'layer17' 중 어디에서 spatial attention을 볼지
        """
        x = self.layer1(x)
        x = self.maxpool(x)

        # --- stage 2 ---
        sc = self.bn_conv2(self.conv2(x))
        if raw_img is not None and visualize_at == "layer2":
            out, sa = self.cbam(self.layer2(x), self.mlp2, self.spa2, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer2)")
        else:
            x = F.relu(self.cbam(self.layer2(x), self.mlp2, self.spa2) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer3":
            out, sa = self.cbam(self.layer3(x), self.mlp3, self.spa3, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer3)")
        else:
            x = F.relu(self.cbam(self.layer3(x), self.mlp3, self.spa3) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer4":
            out, sa = self.cbam(self.layer4(x), self.mlp4, self.spa4, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer4)")
        else:
            x = F.relu(self.cbam(self.layer4(x), self.mlp4, self.spa4) + sc)

        # --- stage 3 ---
        sc = self.bn_conv5(self.conv5(x))
        if raw_img is not None and visualize_at == "layer5":
            out, sa = self.cbam(self.layer5(x), self.mlp5, self.spa5, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer5)")
        else:
            x = F.relu(self.cbam(self.layer5(x), self.mlp5, self.spa5) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer6":
            out, sa = self.cbam(self.layer6(x), self.mlp6, self.spa6, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer6)")
        else:
            x = F.relu(self.cbam(self.layer6(x), self.mlp6, self.spa6) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer7":
            out, sa = self.cbam(self.layer7(x), self.mlp7, self.spa7, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer7)")
        else:
            x = F.relu(self.cbam(self.layer7(x), self.mlp7, self.spa7) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer8":
            out, sa = self.cbam(self.layer8(x), self.mlp8, self.spa8, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer8)")
        else:
            x = F.relu(self.cbam(self.layer8(x), self.mlp8, self.spa8) + sc)

        # --- stage 4 ---
        sc = self.bn_conv9(self.conv9(x))
        if raw_img is not None and visualize_at == "layer9":
            out, sa = self.cbam(self.layer9(x), self.mlp9, self.spa9, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer9)")
        else:
            x = F.relu(self.cbam(self.layer9(x), self.mlp9, self.spa9) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer10":
            out, sa = self.cbam(self.layer10(x), self.mlp10, self.spa10, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer10)")
        else:
            x = F.relu(self.cbam(self.layer10(x), self.mlp10, self.spa10) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer11":
            out, sa = self.cbam(self.layer11(x), self.mlp11, self.spa11, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer11)")
        else:
            x = F.relu(self.cbam(self.layer11(x), self.mlp11, self.spa11) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer12":
            out, sa = self.cbam(self.layer12(x), self.mlp12, self.spa12, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer12)")
        else:
            x = F.relu(self.cbam(self.layer12(x), self.mlp12, self.spa12) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer13":
            out, sa = self.cbam(self.layer13(x), self.mlp13, self.spa13, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer13)")
        else:
            x = F.relu(self.cbam(self.layer13(x), self.mlp13, self.spa13) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer14":
            out, sa = self.cbam(self.layer14(x), self.mlp14, self.spa14, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer14)")
        else:
            x = F.relu(self.cbam(self.layer14(x), self.mlp14, self.spa14) + sc)

        # --- stage 5 ---
        sc = self.bn_conv16(self.conv16(x))
        if raw_img is not None and visualize_at == "layer15":
            out, sa = self.cbam(self.layer15(x), self.mlp15, self.spa15, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer15)")
        else:
            x = F.relu(self.cbam(self.layer15(x), self.mlp15, self.spa15) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer16":
            out, sa = self.cbam(self.layer16(x), self.mlp16, self.spa16, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer16)")
        else:
            x = F.relu(self.cbam(self.layer16(x), self.mlp16, self.spa16) + sc)

        sc = x
        if raw_img is not None and visualize_at == "layer17":
            out, sa = self.cbam(self.layer17(x), self.mlp17, self.spa17, return_sa=True)
            x = F.relu(out + sc)
            show_or_save_overlay(raw_img, sa, title="CBAM Spatial Attention (layer17)")
        else:
            x = F.relu(self.cbam(self.layer17(x), self.mlp17, self.spa17) + sc)

        # head
        x = self.avgpool_final(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)
        return x


# --- 4) 메인 실행부 ---
if __name__ == "__main__":
    # 1) 모델 로드
    model = Attention_module().to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    # 체크포인트가 순수 state_dict라고 가정 (네 파일은 이 형태로 보임)
    state = torch.load(MODEL_PATH, map_location=DEVICE)

    # DataParallel로 저장했으면 module. 제거
    if isinstance(state, dict) and len(state) > 0:
        any_key = next(iter(state.keys()))
        if isinstance(any_key, str) and any_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    print("✅ 가중치 로드 성공")
    model.eval()

    # 2) 이미지 로드
    img_bgr = cv2.imread(TARGET_IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지 경로를 확인해 주세요: {TARGET_IMAGE_PATH}")

    # 시각화용 원본 (128x128)
    raw_for_show = cv2.resize(img_bgr, (128, 128))  # uint8 BGR

    # 전처리 (/255 * 2 - 1)  -> 학습 때 이걸 썼다면 그대로 OK
    img_input = raw_for_show.astype(np.float32) / 255.0 * 2.0 - 1.0
    input_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 3) 추론 + 시각화
    with torch.no_grad():
        logits = model(input_tensor, raw_img=raw_for_show, visualize_at="layer17")

    pred = int(torch.argmax(logits, dim=1).item())
    print(f"예측 클래스 id: {pred}")
