# Team05 共用程式與開工交接

從 GitHub 下載專案後，先備妥課程提供的 `STUDENT_PUBLIC/`。在自己的 `stage1/`、`stage2/` 或 `stage3/` 資料夾開啟 PowerShell，裝好環境後，依下表的操作說明執行。[環境安裝說明](handoff/SETUP.md)

| 交給 | 現在可以執行 | 組員接續 |
|---|---|---|
| [Stage 1 · 葉文賢（A）](handoff/Stage1.md) | 計算 train mean / std、驗證灰階 loader、重跑 PCA + SVM，取得指標與錯分檔名 | 自建 CNN、E1–E4、選模與 Stage 1 報告 |
| [Stage 2 · 李安爵（B）](handoff/Stage2.md) | 以三通道讀取官方 split，共用分類評估、曲線、混淆矩陣、模型大小及推論時間 | 兩個模型家族 × 兩個策略，共四組實驗 |
| [Stage 3 · 謝允柔（C）](handoff/Stage3.md) | 檢查標註並建立 Ultralytics 資料目錄。取得六類 detector checkpoint 後，一個指令即可匯出對照圖、失敗案例與門檻分析 | 訓練 detector、依 validation mAP@0.5 選模，回報正式評估 |

第五組對應 `Team05`。環境採 Python 3.12.2，PyTorch 2.7.1 與 torchvision 0.22.1。在本機 NVIDIA GeForce RTX 2060（driver 496.76）及 CPU 均已跑過卷積、反向傳播和 NMS。各 Stage 的直接依賴列在各自的 `requirements.txt`。[安裝與驗證指令](handoff/SETUP.md) · [版本表](../AGENTS.md)

## 資料來源與核對

本機的課程資料位於 `FactoryVision_DataSet/STUDENT_PUBLIC/`。程式接受 `--data_root`，搬到其他電腦只需指定該電腦的資料位置。重跑下列指令可取得每類張數、影像格式、偵測框數量、缺少配對的檔名或跨 split 重複檔案的 SHA-256。

```powershell
python common/audit_data.py --data_root FactoryVision_DataSet/STUDENT_PUBLIC --output common/evidence/dataset_audit.json
```

兩個任務各有 train 1,260 張與 validation 270 張。分類圖都是 200 × 200 灰階 PNG，每類分別 210 與 45 張。偵測圖都是 200 × 200 RGB PNG，兩個 split 分別有 2,924 與 641 個 YOLO 標註框，圖片與 TXT、XML 的檔名可逐一配對。[逐檔核對結果](evidence/dataset_audit.json)

原始分割各有一對位元完全相同的 PNG，對應 train 的 `TRN_0504.png` 和 validation 的 `VAL_0101.png`。分類標籤均為 `patches`，兩份偵測 TXT 的三個框座標略有不同。實驗保留課程的固定 split，只對 train 執行 fit 與訓練用 augmentation。報告解讀 validation 時，請揭露這項已知重疊。[兩對檔案的雜湊與位置](evidence/dataset_audit.json)

## 已凍結的分類介面

| 呼叫 | 回傳或行為 |
|---|---|
| `get_dataloaders(...)` | 回傳 `train_loader, val_loader`。兩者每個 batch 都是 `images, labels, filenames`。檔名格式為 `類別/圖檔.png`。只有 train 打亂順序。 |
| `train_transform` | 可傳入完整的 PIL → tensor 流程，包含 resize、augmentation、轉 tensor 及 normalization。此設定只套用於 train。預設兩個 split 都只做 resize、轉 tensor、normalization。 |
| `compute_train_normalization(data_root, input_size, channels)` | Stage 1 以實際輸入尺寸計算 train 全部像素的 mean 與 population std，回傳兩份 list。 |
| `load_numpy(data_root, input_size)` | 回傳 `X_train, y_train, X_val, y_val`。影像為 float32、灰階、攤平且除以 255。 |
| `compute_metrics(y_true, y_pred)` | 回傳 `metrics`（含四項分類指標）及 `confusion_matrix`。列是真實類別，欄是預測類別，每一格為張數。 |
| `evaluate(model, loader, device)` | 同上，另回傳順序對齊的 `filenames, y_true, y_pred`。選模取 `result["metrics"]["macro_f1"]`。模型每張圖需輸出六個 logits。 |
| `plot_curves(history, path)`、`plot_confusion_matrix(cm, path)` | 產生 150 dpi 英文圖。輸出檔名帶實驗 ID，例如 `results/E1_curves.png`、`results/E1_cm.png`。 |
| `model_stats(model, ckpt_path)` | 回傳 `params, trainable_params, checkpoint_mb`。大小讀取實際檔案，以 bytes ÷ 1024² 計算。 |
| `measure_latency(model, input_size, device, runs)` | 回傳每張平均毫秒數。支援正方形邊長或 `(C, H, W)`。報告使用 batch 1、暖機 10 次、`runs=100`，CUDA 計時同步。適用於 A、B 的模型 forward。 |

黃怡禎（D）負責維護 `common/` 正本，並將共用程式同步到各 Stage 的 `src/`。Stage 1、2 共用 `data.py`、`seed.py`、`metrics.py`、`check_classification.py`，Stage 1 另有 `pca_svm.py`。Stage 3 使用 `prepare_yolo.py`、`visualize.py`、`seed.py`。後續更新 Stage 的 config 時保留組員已填的實驗設定。

## 結果與交付依據

- Stage 1 的 PCA + SVM 在官方 validation 的 Accuracy 為 0.7630，Macro Precision 為 0.7920，Macro Recall 為 0.7630，Macro F1 為 0.7608。以完整 float64 SVD 擬合 PCA，在兩個安裝環境和不同 CPU 執行緒設定下重訓，270 張預測、錯分檔名和混淆矩陣一致。[實驗 JSON](../stage1/results/pca_svm.json) · [混淆矩陣](../stage1/results/pca_svm_cm.png) · [交接 ZIP 解壓重訓紀錄](evidence/pca_reproduction.json)
- 固定猜第一類的 loader 檢查得到 `45/270 = 0.166667`，一通道和三通道的混淆矩陣總和皆為 270。這是評估程式的已知答案檢查。A、B 的正式模型由各自的 `train.py` 與 `evaluate.py` 實作。
- Stage 3 已用 Ultralytics 讀取全部影像與標註，並以未訓練的六類架構走過 270 張 validation 的圖檔輸出流程。C 的正式結果、mAP 與圖表須用自行訓練後選定的 checkpoint 產生。
- 課程提供的 `FactoryVision_Final_Project.pdf` 第 5、8、21–22、24–29 頁列出資料用途、實驗與交件要求。黃怡禎（D）的彙整分工見組內的 `FactoryVision_Team_Plan.html`，共用介面及格式見[團隊規則](../AGENTS.md)。

在 repo 根目錄執行 `python common/package_handoff.py` 可重建三份組內交接 ZIP。A 的套件同時包含這次實測的 PCA checkpoint，重新打包前需先執行 A 的 baseline 指令。正式 Stage ZIP 由各負責人完成模型與報告後製作。
