# Stage 1 · Baseline + Improved CNN

負責人為葉文賢（A）。先跑黃怡禎（D）提供的 PCA + SVM，再從零訓練自建 CNN。

## 1. 安裝環境

在 Stage 根目錄開啟 PowerShell：解壓 ZIP 時為 `Team05_Stage1/`，在 team repo 中為 `stage1/`。使用 Python 3.12.2。以下先安裝 CPU 版 PyTorch，使用相容的 NVIDIA GPU 時，可把 PyTorch 安裝網址末尾的 `cpu` 改為團隊驗證過的 `cu118`。已有團隊虛擬環境時，也可讓 `$py` 指向該環境的 Python。[PyTorch 官方安裝指令](https://pytorch.org/get-started/previous-versions/#v271)

```powershell
python --version
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

最後一行會顯示套件版本和 CUDA 是否可用。以下指令都在同一個 PowerShell 視窗、同一個 Stage 資料夾執行。

## 2. 取得課程資料

從課程提供的下載來源取得 FactoryVision Public Dataset 並解壓。把 `$root` 設成 `STUDENT_PUBLIC` 資料夾的路徑（相對或絕對路徑皆可），程式不需修改。下例假設 `FactoryVision_DataSet/` 與 Stage 根目錄放在同一層。

```powershell
$root = "../FactoryVision_DataSet/STUDENT_PUBLIC"
```

這個 Stage 使用 `classification/{train,validation}/<類別>/`。沿用官方的 train 和 validation 分割，訓練只使用 train。分類資料的官方分割中有一對內容相同的影像（train 的 `TRN_0504.png`、validation 的 `VAL_0101.png`）。保留原始分割，並在報告中說明這項重疊。

## 3. 先跑資料檢查和 PCA + SVM

```powershell
& $py src/check_classification.py --data_root $root --input_size 128 --channels 1
& $py src/data.py --data_root $root --input_size 128 --channels 1
& $py src/pca_svm.py --data_root $root
```

- 第一支指令核對資料讀取、類別順序和混淆矩陣。使用課程資料應看到 `train=1260`、`validation=270`、`batch=(32, 1, 128, 128)`。
- 第二支指令只用 train 計算 normalization。128 × 128 灰階圖的 mean 為 `0.5070540018850411`，std 為 `0.2110269081061105`。若改用其他 CNN 輸入尺寸，請用該尺寸重算並填入最終 `config.json`。
- 第三支指令把結果寫到 `results/pca_svm.json`、`results/pca_svm_cm.png`，PCA checkpoint 寫到 `checkpoints/pca_svm.pkl`。此基準將影像縮至 64 × 64、像素除以 255 並攤平，只用 train 擬合 float64 完整 SVD（64 個主成分）和 RBF SVM（C=10、gamma=scale、seed=42）。實測 validation Accuracy 為 0.7630，Macro Precision 為 0.7920，Macro Recall 為 0.7630，Macro F1 為 0.7608；重跑時只有 latency 會隨電腦不同而改變。[實測 JSON](results/pca_svm.json) · [混淆矩陣](results/pca_svm_cm.png)

## 4. 自建 CNN 與 Ablation

### 4.1 模型與設定

| 項目 | 內容 |
|---|---|
| 模型 | `SimpleCNN`（`src/model.py`）：4 個 block，每個 Conv3×3 → ReLU → MaxPool2d(2)，通道 32 → 64 → 128 → 256；接 Global Average Pooling 和 Linear(256, 6)。參數 389,382 個 |
| Pretrained source | 無，from scratch（隨機初始化） |
| 輸入與前處理 | 128 × 128 灰階（1 channel）；Resize → ToTensor → Normalize。mean `0.5070540018850411`、std `0.2110269081061105`，只用 train 計算 |
| 訓練 | Adam、lr 0.001、weight_decay 0、batch 32、50 epochs、CrossEntropyLoss、seed 42 |
| 選 epoch | 每組取 validation Macro F1 最高的 epoch（平手取較早者） |
| 方法 A：Data Augmentation | 只套用在 train：RandomHorizontalFlip p=0.5、RandomVerticalFlip p=0.5、RandomRotation ±15°（同一組 pipeline，算一種方法） |
| 方法 B：Dropout | Linear 分類層前加 Dropout，p=0.5 |
| 訓練裝置 | CPU（Intel Core i5-12500，無 NVIDIA GPU；JSON 的 `latency_device` 記為 `Intel64 Family 6 Model 151`），Python 3.12.10、torch 2.7.1+cpu。Python 與團隊規定的 3.12.2 只差 patch 版本，套件版本相同 |

| Exp | 設定 | A | B |
|---|---|---|---|
| `E1` | Baseline CNN | ✗ | ✗ |
| `E2` | Baseline + Data Augmentation | ✓ | ✗ |
| `E3` | Baseline + Dropout | ✗ | ✓ |
| `E4` | Baseline + Data Augmentation + Dropout | ✓ | ✓ |

### 4.2 重現指令

從 Stage 根目錄執行。每組約 11 分鐘（CPU）。

```powershell
& $py src/train.py --data_root $root --exp E1
& $py src/train.py --data_root $root --exp E2
& $py src/train.py --data_root $root --exp E3
& $py src/train.py --data_root $root --exp E4
Copy-Item checkpoints/E3.pth stage1_model.pth
& $py src/evaluate.py --data_root $root
```

- `train.py` 輸出 `checkpoints/<exp>.pth`（不進 git）、`results/<exp>.json`、`results/<exp>_curves.png`、`results/<exp>_cm.png`。
- `evaluate.py` 讀 `config.json` 重建模型、載入 `stage1_model.pth`，印出 validation 指標、混淆矩陣和分錯的圖檔。

### 4.3 結果

Table 1：Stage 1 validation 結果（270 張）。數字取自 `results/*.json`；Latency 為 batch 1、暖機 10 次、100 次平均，同一台 CPU。

| Exp | Setting | Accuracy | Macro P | Macro R | Macro F1 | Params | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| `pca_svm` | PCA(64) + RBF SVM | 0.7630 | 0.7920 | 0.7630 | 0.7608 | — | 2.49 | 0.33 |
| `E1` | Baseline CNN | 0.9926 | 0.9928 | 0.9926 | 0.9926 | 389,382 | 1.49 | 2.70 |
| `E2` | + Data Augmentation | 0.9926 | 0.9929 | 0.9926 | 0.9926 | 389,382 | 1.49 | 2.67 |
| `E3` | + Dropout | 0.9926 | 0.9928 | 0.9926 | 0.9926 | 389,382 | 1.49 | 2.81 |
| `E4` | + Data Augmentation + Dropout | 0.9852 | 0.9860 | 0.9852 | 0.9850 | 389,382 | 1.49 | 2.97 |

- CNN 比 PCA + SVM 的 Macro F1 高約 0.23。傳統方法錯最多的是 `scratches`（45 張只對 22 張，14 張被判成 `inclusion`）、`inclusion`（對 27 張）和 `crazing`（對 33 張，12 張被判成 `rolled_in_scale`），見 `results/pca_svm_cm.png`。這些類別靠細長紋理或細裂紋區分，攤平像素 + PCA 抓不到這種局部形狀，CNN 的卷積層則可以。
- E1、E2、E3 在 validation 都只錯 2 張；E4 錯 4 張。270 張中 1 張約等於 0.0037，所以 E1～E3 的差異落在 1 張圖以內。

### 4.4 Final Improved CNN：E3（Baseline + Dropout）

- 依規定最終模型須為 Improved CNN，只從 E2～E4 選，標準是 validation Macro F1。
- E3 的 Macro F1 為 0.992591677985，E2 為 0.992588932806，E4 為 0.985040216098。E3 最高；三組參數量相同，不需用參數量決勝。
- E3 和 E2 在 4 位小數下相同，差在 E2 錯的 2 張都是 `pitted_surface`（該類 recall 較低），E3 的 2 張錯誤分散在 `patches` 和 `pitted_surface` 兩類，所以 macro 平均略高。
- E3 的 Macro F1 與 Baseline E1 相同（0.9926），也就是 Dropout **沒有讓 validation 分數再提升**，但也沒有變差。`stage1_model.pth` 即 `checkpoints/E3.pth`（best epoch 48），SHA-256 記錄在 `results/E3.json`。

### 4.5 錯誤案例

| 圖檔 | 真實 | 預測 | 哪些實驗分錯 | 觀察 |
|---|---|---|---|---|
| `patches/VAL_0097.png` | patches | crazing | E1、E3 | 只有幾個小而淡的暗斑，背景顆粒感重，跟 crazing 的細紋理相似；典型 patches 是大片暗區 |
| `pitted_surface/VAL_0144.png` | pitted_surface | inclusion | E3 | 對比很低、幾乎看不到凹坑，只剩直向條紋，和 inclusion 的長條狀紋理接近 |
| `pitted_surface/VAL_0175.png` | pitted_surface | rolled_in_scale | E1、E2、E4 | 對比低、上下亮度不均，凹坑不明顯 |

- 四組 CNN 共 10 次錯誤中有 8 次是 `pitted_surface`，是最難的類別；錯誤多發生在**對比低、瑕疵特徵不明顯**的圖。
- 每組完整的分錯清單在 `results/<exp>.json` 的 `misclassified` 欄位。

### 4.6 Overfitting / Generalization 分析

Table 2：最佳 epoch 時的 train 與 validation 數值（取自各 JSON 的 `history`）。

| Exp | Best epoch | Train acc | Val acc | Train loss | Val loss |
|---|---|---|---|---|---|
| `E1` | 42 | 0.9921 | 0.9926 | 0.0332 | 0.0254 |
| `E2` | 42 | 0.9659 | 0.9926 | 0.0854 | 0.0451 |
| `E3` | 48 | 0.9794 | 0.9926 | 0.0553 | 0.0238 |
| `E4` | 38 | 0.9722 | 0.9852 | 0.0864 | 0.0524 |

- **Baseline 沒有明顯 overfitting。** E1 的 train 和 validation 曲線在 50 epochs 內幾乎重疊（`results/E1_curves.png`），最佳 epoch 時 train acc 0.9921、val acc 0.9926。val loss 後期有起伏（例如 epoch 48 為 0.0770），但隨即回落到 0.0234（epoch 50），沒有持續上升的趨勢。模型沒有「死背」訓練資料。
- **這說明改善方法為什麼沒有加分。** Dropout 和 Data Augmentation 都是用來壓低 overfitting 的；Baseline 本身已經不 overfit，又已經接近 99%，兩種方法沒有可以修正的空間。
- **E2～E4 的 train acc 比 val acc 低**，是因為 train 的數字是在 augmentation 和 Dropout 開啟時算的（題目變難），validation 則是關閉狀態下算的。這是正常現象，不代表 underfitting。
- **E4（A + B 同時用）分數最低。** 兩種正則化疊加後，模型在 50 epochs 結束時仍最難擬合 train：最後一個 epoch 的 train loss E4 為 0.1211，高於 E1（0.0350）、E2（0.0609）、E3（0.0857）；validation 錯誤集中在 `pitted_surface`（4 張）。可能需要更多 epochs 才能發揮。
- **限制：** validation 只有 270 張，E1～E3 的差異只有 0～1 張圖，不足以證明哪個方法真的比較好；另外官方 split 中 `TRN_0504.png` 和 `VAL_0101.png` 內容相同（見第 2 節），保留原始分割未處理。
