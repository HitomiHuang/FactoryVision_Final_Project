# Team05 · Stage 交件 README 範本

實驗完成後，Stage 1 由葉文賢（A）、Stage 2 由李安爵（B）、Stage 3 由謝允柔（C）使用此範本撰寫各自的交件 README。保留該 Stage 的段落並填入實際值。三份 `stageN.config.json` 已對應課程 PDF 第 28 頁，Stage 根目錄使用的檔名為 `config.json`。範本的 null 待最終模型確定後填寫，無 NMS 的 detector 依規定保留 `nms_iou_threshold: null`。

## 環境與安裝

Python 3.12.2。torch 2.7.1、torchvision 0.22.1。Stage 3 使用 Ultralytics 時為 8.3.239。從解壓後的 Stage 根目錄建立環境並安裝 `requirements.txt`。請記錄實際驗證的裝置與 wheel。[PyTorch 安裝選項](https://pytorch.org/get-started/previous-versions/#v271)

```powershell
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip check
$root = "../FactoryVision_DataSet/STUDENT_PUBLIC"
```

## 最終模型

| 項目 | 完成實驗後填寫 |
|---|---|
| 組別、Stage、實驗編號 | Team05；Stage 與實際選定的實驗編號 |
| 模型及選模依據 | 模型名稱、最佳 epoch、validation 的選模指標 |
| Checkpoint | Stage 1 用 `stage1_model.pth`，Stage 2 用 `stage2_model.pth`，Stage 3 用 `stage3_model.pt` |
| Pretrained source | Stage 1 填 from scratch，Stage 2、3 記錄實際公開權重的名稱及來源 |
| 輸入與前處理 | 尺寸、通道數、resize、normalization，Stage 3 另列 confidence 與 NMS IoU |
| 訓練設定 | 與該 checkpoint 相符的策略、改善方法及超參數 |

## 重現指令

以下是團隊約定的訓練與評估入口。將實際產生最終權重的完整參數寫進交件 README，並從乾淨環境重跑。下列指令中 E1 與 D1 分別示範分類與偵測的實驗編號。

```powershell
# Stage 1
& $py src/pca_svm.py --data_root $root
& $py src/train.py --data_root $root --exp E1
& $py src/evaluate.py --data_root $root --checkpoint stage1_model.pth

# Stage 2
& $py src/train.py --data_root $root --exp E1
& $py src/evaluate.py --data_root $root --checkpoint stage2_model.pth

# Stage 3, for the team's Ultralytics data layout
& $py src/prepare_yolo.py --data_root $root
& $py src/train.py --data_root $root --exp D1
& $py src/evaluate.py --data_root $root --checkpoint stage3_model.pt
& $py src/visualize.py --data_root $root --exp D1
```

## 驗證結果

列出實際 `results/<exp>.json`、圖檔與對應報告圖表。Stage 1、2 要說明 Macro F1 與混淆矩陣；Stage 3 要說明 mAP@0.5、推論時間與失敗案例。數字取自對應正式實驗的原始 JSON。
