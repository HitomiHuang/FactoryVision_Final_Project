# Team05 · Stage 2 · Transfer Learning

以兩個不同的 Architecture Family（ResNet18、DenseNet121）各搭配兩種 Fine-tuning Strategy（Feature Extraction、Full Fine-tuning），完成 2 × 2 共 4 組實驗（E1–E4），並依 validation Macro F1 選出最終分類模型。

## 1. 環境

| 項目 | 版本 |
|---|---|
| Python | 團隊統一版本為 3.12.2；本 Stage 實際訓練與驗證使用 3.12.10 |
| torch / torchvision | 2.7.1 / 0.22.1 |
| numpy / scikit-learn / matplotlib / Pillow | 2.2.6 / 1.7.2 / 3.10.8 / 12.1.1 |
| 訓練裝置 | macOS，Apple M2 Max（PyTorch MPS） |

在本資料夾（`Team05_Stage2/`）建立虛擬環境並安裝套件：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip check
```

使用 NVIDIA GPU 時，可先從 PyTorch 官方 CUDA 索引安裝相同版本的 torch / torchvision，再安裝 `requirements.txt`：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

以下指令都在本資料夾執行，`--data_root` 指向課程資料的 `STUDENT_PUBLIC/`，請替換成實際路徑。

## 2. 最終模型

| 項目 | 內容 |
|---|---|
| 實驗編號 | E3 |
| 模型 | DenseNet121 + Feature Extraction |
| Checkpoint | `stage2_model.pth`（`model.state_dict()`，27.08 MB） |
| Pretrained source | torchvision `DenseNet121_Weights.IMAGENET1K_V1`（ImageNet-1K），<https://download.pytorch.org/models/densenet121-a639ec97.pth> |
| 最佳 epoch | 20 / 20 |
| Validation Macro F1 | 1.0000 |
| 設定檔 | `config.json` |

選模依據：E2、E3、E4 的 validation Macro F1 同為 1.0000。依團隊規則，同分時選參數量較少者，因此選 DenseNet121（6,960,006 個參數，ResNet18 為 11,179,590 個）。E3 與 E4 的總參數量相同，E3 只需訓練 6,150 個參數，CPU 推論時間也較短，因此最終選 E3。

## 3. 輸入與前處理

課程資料為 200 × 200 的 8-bit 灰階 PNG，訓練與驗證使用相同的前處理，沒有使用 data augmentation：

1. 灰階影像轉成 3 通道 RGB（三個通道數值相同）。
2. `Resize((224, 224))`，bilinear。
3. `ToTensor()`，像素縮放到 [0, 1]。
4. `Normalize`，使用 ImageNet pretrained 權重的官方值：mean = `[0.485, 0.456, 0.406]`，std = `[0.229, 0.224, 0.225]`。

## 4. 實驗設定

四組實驗使用相同的官方 train / validation 分割、前處理、評估流程與超參數；只有 learning rate 依策略不同。

| 共同設定 | 值 |
|---|---|
| Input size | 224 × 224 × 3 |
| Optimizer / Loss | Adam / CrossEntropyLoss |
| Epochs / Batch size | 20 / 32 |
| Weight decay | 0 |
| Seed | 42 |
| 選最佳 epoch | validation Macro F1 最高；同分時選 validation loss 最低的 epoch |

| Exp | Architecture | Strategy | Learning rate | Frozen layers | Trainable layers |
|---|---|---|---|---|---|
| E1 | ResNet18 | Feature Extraction | 0.001 | `conv1`、`bn1`、`layer1`–`layer4` | `fc`（新的 6 類 Linear） |
| E2 | ResNet18 | Full Fine-tuning | 0.0001 | 無 | 全部 |
| E3 | DenseNet121 | Feature Extraction | 0.001 | `features` | `classifier`（新的 6 類 Linear） |
| E4 | DenseNet121 | Full Fine-tuning | 0.0001 | 無 | 全部 |

- 兩個架構都把原本 1000 類的 ImageNet 分類層換成新的 6 類 Linear 層。
- Feature Extraction 凍結整個 backbone，訓練時整個網路維持 `eval()` 模式，BatchNorm 沿用 ImageNet 的 running mean / var，只更新新的分類層。
- Full Fine-tuning 從 ImageNet 權重開始訓練所有層。

## 5. 重現指令

### 5.1 資料讀取檢查

```bash
python src/check_classification.py --data_root path/to/STUDENT_PUBLIC --channels 3 --input_size 224
```

應看到 `train=1260 validation=270 batch=(32, 3, 224, 224) constant_accuracy=45/270`。

### 5.2 訓練 E1–E4

以下是產生 `results/` 內結果的完整指令。`--device mps` 為 Apple Silicon；NVIDIA GPU 改用 `--device cuda`，省略時為 `auto`（有 CUDA 用 CUDA，否則用 CPU）。

```bash
python src/train.py --data_root path/to/STUDENT_PUBLIC --exp E1 --architecture resnet18 --strategy feature_extraction --lr 1e-3 --epochs 20 --batch_size 32 --input_size 224 --weight_decay 0 --seed 42 --device mps
python src/train.py --data_root path/to/STUDENT_PUBLIC --exp E2 --architecture resnet18 --strategy full_finetune --lr 1e-4 --epochs 20 --batch_size 32 --input_size 224 --weight_decay 0 --seed 42 --device mps
python src/train.py --data_root path/to/STUDENT_PUBLIC --exp E3 --architecture densenet121 --strategy feature_extraction --lr 1e-3 --epochs 20 --batch_size 32 --input_size 224 --weight_decay 0 --seed 42 --device mps
python src/train.py --data_root path/to/STUDENT_PUBLIC --exp E4 --architecture densenet121 --strategy full_finetune --lr 1e-4 --epochs 20 --batch_size 32 --input_size 224 --weight_decay 0 --seed 42 --device mps
```

每組會輸出：

- `checkpoints/<exp>.pth`：最佳 epoch 的 `state_dict`
- `results/<exp>.json`：指標、混淆矩陣、參數量、checkpoint 大小、推論時間、每個 epoch 的 loss / accuracy / Macro F1、預測錯誤的檔名
- `results/<exp>_curves.png`：Training / Validation Curves
- `results/<exp>_cm.png`：Confusion Matrix

最終模型由 E3 複製而來：`checkpoints/E3.pth` → `stage2_model.pth`。

MPS 運算不保證每次結果完全相同，即使固定 seed 42，重跑的數字仍可能有些微差異。報告數字以 `results/*.json` 為準。

### 5.3 驗證最終模型

```bash
python src/evaluate.py --data_root path/to/STUDENT_PUBLIC
```

`evaluate.py` 讀取 `config.json`，以其中的 architecture、strategy 重建模型後載入 `stage2_model.pth`，在 validation 上輸出 Accuracy、Macro Precision、Macro Recall、Macro F1、參數量與混淆矩陣。預設 `--checkpoint stage2_model.pth --config config.json --device auto`。

## 6. Validation 結果

數字取自 `results/E1.json`–`results/E4.json`。

| Exp | Setting | Accuracy | Macro P | Macro R | Macro F1 | Params | Trainable Params | Size (MB) | Latency (ms) | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | ResNet18 + Feature Extraction | 0.9926 | 0.9927 | 0.9926 | 0.9926 | 11,179,590 | 3,078 | 42.71 | 8.73 | 20 |
| E2 | ResNet18 + Full Fine-tuning | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11,179,590 | 11,179,590 | 42.71 | 9.45 | 10 |
| E3 | DenseNet121 + Feature Extraction | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6,960,006 | 6,150 | 27.08 | 30.99 | 20 |
| E4 | DenseNet121 + Full Fine-tuning | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6,960,006 | 6,960,006 | 27.08 | 34.30 | 19 |

- Size 為 checkpoint 檔案大小（bytes ÷ 1024²）。
- Latency 的量法是 batch size 1、暖機 10 次、連續 100 次取平均，單位為 ms / 張，只計模型推論，不含讀檔。在 Apple M2 Max 的 **CPU** 上量測。共用的 `measure_latency` 只會為 CUDA 同步，MPS 的計時不準，所以用 MPS 訓練時改在 CPU 量。

## 7. 資料使用說明

- 只使用課程提供的 Classification Public Dataset，沿用官方 train（1,260 張）/ validation（270 張）分割。訓練只用 train，validation 只用於選 epoch 與比較模型。
- 官方分割中，train 的 `TRN_0504.png` 與 validation 的 `VAL_0101.png` 內容完全相同。本 Stage 保留原始分割，沒有移除或修改。
- E2、E3、E4 的 validation 指標都是 1.0000，270 張 validation 影像已無法區分這三組的分類表現，因此最終選模以參數量等部署成本為依據。
