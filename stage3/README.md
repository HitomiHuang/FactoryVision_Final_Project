# Stage 3 · Object Detection

負責人為謝允柔（C）。先把課程的 YOLO 標註整理成 Ultralytics 能讀取的目錄，再訓練並評估 detector。

## 1. 安裝環境

將專案下載或 clone 到本機，從專案根目錄開啟 PowerShell。使用 Python 3.12.2。以下先安裝 CPU 版 PyTorch，使用相容的 NVIDIA GPU 時，可把 PyTorch 安裝網址末尾的 `cpu` 改為團隊驗證過的 `cu118`。已有團隊虛擬環境時，也可讓 `$py` 指向該環境的 Python。[PyTorch 官方安裝指令](https://pytorch.org/get-started/previous-versions/#v271)

```powershell
cd stage3
python --version
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py -m pip check
New-Item -ItemType Directory -Force -Path "../data/ultralytics_config" | Out-Null
$env:YOLO_CONFIG_DIR = (Resolve-Path "../data/ultralytics_config").Path
& $py -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

`YOLO_CONFIG_DIR` 將 Ultralytics 的設定存到專案的 `data/`。最後一行會顯示套件版本和 CUDA 是否可用。以下指令都在同一個 PowerShell 視窗、同一個 Stage 資料夾執行。

## 2. 取得課程資料

從課程提供的下載來源取得 FactoryVision Public Dataset，解壓後把 `STUDENT_PUBLIC/` 放到專案根目錄的 `FactoryVision_DataSet/` 內。若已解壓到其他位置，把 `$root` 改成實際的 `STUDENT_PUBLIC` 目錄。

```powershell
$root = "../FactoryVision_DataSet/STUDENT_PUBLIC"
```

這個 Stage 使用 `detection/{train,validation}/`。沿用官方的 train 和 validation 分割，訓練只使用 train。偵測資料的官方分割中有一對內容相同的影像（train 的 `TRN_0504.png`、validation 的 `VAL_0101.png`），兩份 YOLO 標註的框座標略有不同。保留原始分割，並在報告中說明這項重疊。[資料核對紀錄](../common/evidence/dataset_audit.json)

## 3. 先整理 YOLO 資料

```powershell
& $py src/prepare_yolo.py --data_root $root --out_dir ../data/factoryvision_yolo
```

程式核對類別順序和影像、標註的配對，從原始的 `labels_yolo/` 複製標註到新資料夾的 `labels/`，並保留原始資料與切分。輸出位置為 `../data/factoryvision_yolo/`，內含 `data.yaml`、`images/{train,validation}` 和 `labels/{train,validation}`。再跑一次會驗證已整理的檔案。使用課程資料應看到 train 1,260 張、2,924 個框，validation 270 張、641 個框。[Ultralytics 資料格式](https://docs.ultralytics.com/datasets/detect/)

## 4. 開始訓練與選模

`src/model.py`、`src/train.py` 和 `src/evaluate.py` 目前是待實作的骨架。實驗由 `D1` 起編號。先把 `../data/factoryvision_yolo/data.yaml` 傳給 Ultralytics 的 `train(data=...)`，從公開 pretrained 權重自行訓練，並用 seed `42`。

團隊用 validation **mAP@0.5** 選最佳 epoch，平手選參數較少的模型。Ultralytics 8.3.239 預設用 **mAP@0.5:0.95** 選 `best.pt`。可設 `save_period=1` 保留每個 epoch 的權重，從 `results.csv` 找到 validation mAP@0.5 最高的 epoch，再把對應的權重複製為 `stage3_model.pt`。注意 CSV 從 epoch 1 開始，權重檔名從 `epoch0.pt` 開始。[該版本的選模指標](https://github.com/ultralytics/ultralytics/blob/v8.3.239/ultralytics/utils/metrics.py#L955-L958) · [權重儲存方式](https://github.com/ultralytics/ultralytics/blob/v8.3.239/ultralytics/engine/trainer.py#L617-L623) · [CSV 編號方式](https://github.com/ultralytics/ultralytics/blob/v8.3.239/ultralytics/engine/trainer.py#L757-L765)

## 5. 產生評估圖與失敗案例

用選定的六類 `stage3_model.pt`，將 detector、pretrained 權重來源、輸入尺寸、confidence threshold 和 NMS IoU 填進 `config.json`，再跑視覺化。若最終選的是其他實驗編號，將指令中的 `D1` 換成該編號。

```powershell
& $py src/visualize.py --data_root $root --exp D1
```

- `results/vis/val_01.png` 至 `val_10.png` 為 validation 對照圖。GT 框用綠色，Prediction 框用紅色並標示類別和 confidence。
- `results/failure/` 會從不同影像匯出最多五個 FP、FN 或定位不準（LOC）案例。交件至少要分析五個案例。若本次匯出不足，檢查實際預測再補足並記錄出處。
- `results/D1_visualization.json` 記錄案例和 confidence 掃描。可用 `--thresholds` 設定掃描門檻。此處固定 NMS，再以同類別、IoU ≥ 0.50 一對一配對，合計所有框的 TP / FP / FN 計算 micro Precision / Recall。LOC 指未配對且同類的框在 0.10 ≤ IoU < 0.50，計數時同時構成一個 FP 和一個 FN。
- 正式 `src/evaluate.py` 需另外回報 Precision、Recall、mAP@0.5 和實測推論時間。將結果存到 `results/`，完成可重跑的訓練與評估指令和 `stage3_report.pdf`，並在報告說明 IoU、confidence、NMS 及平均方式。[團隊規則](../AGENTS.md) · [交件範本](../common/templates/README.md)
