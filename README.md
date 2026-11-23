# FactoryVision 期末專題

使用課程提供的鋼板表面瑕疵資料集，完成自建 CNN 分類、Transfer Learning 分類與 Object Detection，最後比較模型表現、大小與推論時間，提出產線部署決策。完整團隊規則見 [AGENTS.md](AGENTS.md)。

## 分工與必做實驗

| 目錄 | 負責人 | 實驗與交付重點 |
|---|---|---|
| [`stage1/`](stage1/) | 葉文賢（A） | PCA + SVM、自建 CNN。選兩種不同的改善方法，跑完 E1 Baseline、E2 +A、E3 +B、E4 +A+B。Flip 與 Rotation 合計只算一種 Data Augmentation。 |
| [`stage2/`](stage2/) | 李安爵（B） | 兩個不同的 Architecture Family × 兩種 Fine-tuning Strategy，四組都跑。記錄 pretrained 權重來源、策略設定、總參數量、可訓練參數量與模型大小。 |
| [`stage3/`](stage3/) | 謝允柔（C） | 至少訓練一個 Detector，繳交至少 10 張 Ground Truth vs Prediction 對照圖及至少 5 個 FP、FN 或定位不準案例，分析 Confidence Threshold 對 Precision 和 Recall 的影響。 |
| [`common/`](common/) | 黃怡禎（D） | 維護共用程式與 [交件範本](common/templates/)，提供 Stage 1 的 PCA + SVM。共用模組由 D 同步到 Stage 1、2 的 `src/`。 |
| [`final_report/`](final_report/) | 黃怡禎（D）彙整，A–C 提供各 Stage 內容 | 用正式提交的結果與 Final Model 做跨 Stage 失敗分析、部署比較與最終決策。七章順序見目錄內的 README。 |

三個 Stage 可並行開工。團隊先確認 iLearning 組別、截止日與可用 GPU。D 確認 `AGENTS.md` 第 6.1 節的統一環境版本，並依團隊時程在第 3 天交付共用介面給 A、B。

## 資料集與實驗規則

將課程資料放到本機的 `data/STUDENT_PUBLIC/`，或以 `--data_root` 指向現有的 `STUDENT_PUBLIC/` 根目錄。Stage 1、2 使用 `classification/{train,validation}/<類別>/`，Stage 3 使用 `detection/{train,validation}/`。兩個任務各有 1,260 張 train、270 張 validation。分類圖為 200 × 200 的 8-bit 灰階 PNG，Stage 2 需轉成 3 通道並使用預訓練模型的官方 Normalization。Stage 1 的 Normalization 只用 train 計算。

- 沿用課程的 train / validation split，seed 固定為 `42`。Validation 僅用於調參、選模、Early Stopping 與比較。訓練和 Data Augmentation 只使用 train。
- 類別順序固定為 `crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled_in_scale`、`scratches`。
- Stage 1 的 Baseline CNN 從零訓練，前處理限 Resize、Tensor Conversion、Normalization。Stage 2、3 若使用外部權重，限公開的 pretrained 權重，例如 ImageNet 或 COCO，並由組員自行訓練。
- Stage 3 若採用 Ultralytics，在本機 `data/` 下另建 `images/` 與 `labels/` 結構，從原始 `labels_yolo/` 取得標註。保留原始資料與 split，並將轉換程式與指令放進 Stage 3。

## 執行與評估

目前是專案骨架。各 Stage 的程式、依賴與模型就緒後，從各自的 Stage 根目錄執行。以下是 Stage 1 的指令格式。

```powershell
cd stage1
python -m pip install -r requirements.txt
python src/train.py --data_root ../data/STUDENT_PUBLIC --exp E1
python src/evaluate.py --data_root ../data/STUDENT_PUBLIC --checkpoint stage1_model.pth
```

| 任務 | 選最佳 epoch 與 Final Model 的依據 | 評估與報告 |
|---|---|---|
| Stage 1、2 分類 | Validation Macro F1 | Accuracy、Macro Precision、Macro Recall、Macro F1、Confusion Matrix、Training / Validation Curves |
| Stage 3 偵測 | Validation mAP@0.5 | Precision、Recall、mAP@0.5、推論時間、對照圖與失敗案例。說明 IoU 與 Confidence Threshold，採用 NMS 時也說明 NMS IoU。 |

主指標平手時選參數較少的模型。跨 Stage 的推論時間統一在同一台電腦、同一張 GPU 上測量，batch size 1，暖機 10 次後取 100 次的平均，單位為 ms / 張。計時涵蓋模型推論及 Stage 3 的 NMS（如有），讀檔時間排除。

實測數字寫入各 Stage 的 `results/<exp>.json`，與實驗圖一起提交 Git。結果、圖名與報告表格格式見 `AGENTS.md` 第 6 節。資料集、checkpoint、`runs/` 與 ZIP 留在本機。組員在各自的分支開發，透過 PR 合併到 `main`。

## 交件前核對

- **ZIP 與組別**。將 `XX` 換成 iLearning 的兩位數組別。每個 `TeamXX_StageN.zip` ≤ 1 GB，解壓後只有一層同名根目錄 `TeamXX_StageN/`。
- **必要檔案**。`stageN_report.pdf`、`config.json`、`requirements.txt`、`README.md`、`src/`（至少有 `model.py`、`train.py`、`evaluate.py`），以及對應的最終 checkpoint `stage1_model.pth`、`stage2_model.pth` 或 `stage3_model.pt`。
- **乾淨環境重跑**。用 Stage README 的指令安裝依賴並評估。確認 `config.json` 合法、設定與最終 checkpoint 一致，資料由 `--data_root` 傳入，checkpoint 使用 ZIP 內的相對路徑。Stage 3 的 `bbox_format` 為 `"xyxy_pixel"`，模型無需 NMS 時將 `nms_iou_threshold` 設為 `null`。
- **報告證據**。各 Stage 的圖與實測數字要寫入各自的 PDF，數字對照 `results/*.json`。D 用已正式提交的結果和 Final Model 彙整 `TeamXX_Final_Report.pdf`。

內容整理自課程 Stage 指南（`FactoryVision_Stage_Guide.html`）、四人分工計畫（`FactoryVision_Team_Plan.html`）及 [AGENTS.md](AGENTS.md)。
