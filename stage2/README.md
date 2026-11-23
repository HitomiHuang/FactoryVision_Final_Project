# Stage 2 · Transfer Learning

負責人為李安爵（B）。先確認三通道資料讀取，再做兩種模型家族各搭配兩種 fine-tuning 策略。

## 1. 安裝環境

將專案下載或 clone 到本機，從專案根目錄開啟 PowerShell。使用 Python 3.12.2。以下先安裝 CPU 版 PyTorch，使用相容的 NVIDIA GPU 時，可把 PyTorch 安裝網址末尾的 `cpu` 改為團隊驗證過的 `cu118`。已有團隊虛擬環境時，也可讓 `$py` 指向該環境的 Python。[PyTorch 官方安裝指令](https://pytorch.org/get-started/previous-versions/#v271)

```powershell
cd stage2
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

## 3. 先跑三通道資料檢查

```powershell
& $py src/check_classification.py --data_root $root --channels 3 --input_size 224
```

使用課程資料應看到 `train=1260`、`validation=270`、`batch=(32, 3, 224, 224)`。檢查程式固定猜第一類，應答對 `45/270`，混淆矩陣總和為 270。

## 4. 開始 Transfer Learning

`src/model.py`、`src/train.py` 和 `src/evaluate.py` 目前是待實作的骨架。先選第一個模型家族與策略，讓 E1 跑完一個 epoch 並確認能載回 checkpoint，再完成四組正式實驗。

| 實驗 | Architecture Family | Fine-tuning Strategy |
|---|---|---|
| `E1` | 甲 | 1 |
| `E2` | 甲 | 2 |
| `E3` | 乙 | 1 |
| `E4` | 乙 | 2 |

- 兩個模型要來自不同家族。用公開 pretrained 權重自行訓練，記錄權重來源、模型所需的前處理與各策略的參數，例如 frozen / trainable layers。
- 用 `get_dataloaders(..., channels=3)` 讀取灰階圖。Normalization 使用所選 pretrained 權重的官方 mean 和 std。例如 ResNet18 的 ImageNet 權重可用下列方式取得。調整輸入尺寸時，記錄實際使用的 Resize 和其他前處理。[Torchvision 權重說明](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)

```python
from torchvision.models import ResNet18_Weights

preprocessing = ResNet18_Weights.IMAGENET1K_V1.transforms()
mean, std = preprocessing.mean, preprocessing.std
```

- 訓練先呼叫 `set_seed(42)`。每組保留最佳的 `state_dict`，例如 `checkpoints/E1.pth`，並在 `results/` 存實測 JSON、曲線、混淆矩陣、總參數與可訓練參數、模型大小及推論時間。用 `evaluate(model, val_loader, device)["metrics"]["macro_f1"]` 取得選模指標，平手選參數較少者。
- 最終權重存為 `stage2_model.pth`。依該模型填好 `config.json`，補上可重跑的訓練和評估指令，並準備 `stage2_report.pdf`。[團隊規則](../AGENTS.md) · [交件範本](../common/templates/README.md)
