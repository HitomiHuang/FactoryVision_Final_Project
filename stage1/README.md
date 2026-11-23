# Stage 1 · Baseline + Improved CNN

負責人為葉文賢（A）。先跑黃怡禎（D）提供的 PCA + SVM，再從零訓練自建 CNN。

## 1. 安裝環境

將專案下載或 clone 到本機，從專案根目錄開啟 PowerShell。使用 Python 3.12.2。以下先安裝 CPU 版 PyTorch，使用相容的 NVIDIA GPU 時，可把 PyTorch 安裝網址末尾的 `cpu` 改為團隊驗證過的 `cu118`。已有團隊虛擬環境時，也可讓 `$py` 指向該環境的 Python。[PyTorch 官方安裝指令](https://pytorch.org/get-started/previous-versions/#v271)

```powershell
cd stage1
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

從課程提供的下載來源取得 FactoryVision Public Dataset，解壓後把 `STUDENT_PUBLIC/` 放到專案根目錄的 `FactoryVision_DataSet/` 內。若已解壓到其他位置，把 `$root` 改成實際的 `STUDENT_PUBLIC` 目錄。

```powershell
$root = "../FactoryVision_DataSet/STUDENT_PUBLIC"
```

這個 Stage 使用 `classification/{train,validation}/<類別>/`。沿用官方的 train 和 validation 分割，訓練只使用 train。分類資料的官方分割中有一對內容相同的影像（train 的 `TRN_0504.png`、validation 的 `VAL_0101.png`）。保留原始分割，並在報告中說明這項重疊。[資料核對紀錄](../common/evidence/dataset_audit.json)

## 3. 先跑資料檢查和 PCA + SVM

```powershell
& $py src/check_classification.py --data_root $root --input_size 128 --channels 1
& $py src/data.py --data_root $root --input_size 128 --channels 1
& $py src/pca_svm.py --data_root $root --kernel rbf --gamma scale --out_dir ../data/stage1_pca_check --checkpoint ../data/stage1_pca_check/pca_svm.pkl
```

- 第一支指令核對資料讀取、類別順序和混淆矩陣。使用課程資料應看到 `train=1260`、`validation=270`、`batch=(32, 1, 128, 128)`。
- 第二支指令只用 train 計算 normalization。128 × 128 灰階圖的 mean 為 `0.5070540018850411`，std 為 `0.2110269081061105`。若改用其他 CNN 輸入尺寸，請用該尺寸重算並填入最終 `config.json`。
- 第三支指令把結果 JSON、混淆矩陣和 PCA checkpoint 寫到本機的 `data/stage1_pca_check/`。此基準將影像縮至 64 × 64、像素除以 255 並攤平，只用 train 擬合 float64 完整 SVD（64 個主成分）和 RBF SVM（C=10、gamma=scale、seed=42）。交接實測的 validation Accuracy 為 0.7630，Macro Precision 為 0.7920，Macro Recall 為 0.7630，Macro F1 為 0.7608。[實測 JSON](results/pca_svm.json) · [混淆矩陣](results/pca_svm_cm.png)

## 4. 開始自建 CNN

`src/model.py`、`src/train.py` 和 `src/evaluate.py` 目前是待實作的骨架。先讓 E1 跑完一個 epoch，確認可載回 checkpoint 並重跑 validation，再做四組正式實驗。

| 實驗 | 設定 |
|---|---|
| `pca_svm` | Traditional Baseline |
| `E1` | Baseline CNN |
| `E2` | Baseline + 改善方法 A |
| `E3` | Baseline + 改善方法 B |
| `E4` | Baseline + 改善方法 A + B |

- E1 從零訓練自建 CNN，前處理只用 Resize、Tensor Conversion 與 Normalization。E2–E4 選兩種不同的改善方法，記錄各自參數。訓練資料增強只套用在 train。
- 用 `get_dataloaders(..., channels=1)` 取得 train 和 validation loader。每個 batch 回傳 `images, labels, filenames`。訓練先呼叫 `set_seed(42)`，用 `evaluate(model, val_loader, device)["metrics"]["macro_f1"]` 取得選模指標。
- 每組保留最佳的 `state_dict`，例如 `checkpoints/E1.pth`，並在 `results/` 存實測 JSON、曲線和混淆矩陣。選 validation Macro F1 最高的模型，平手選參數較少者。
- 最終權重存為 `stage1_model.pth`。依實際模型填寫 `config.json`，補上可重跑的訓練和評估指令，並準備 `stage1_report.pdf`。[團隊規則](../AGENTS.md) · [交件範本](../common/templates/README.md)
