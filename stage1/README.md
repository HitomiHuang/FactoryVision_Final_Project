# Stage 1 · Baseline + Improved CNN

負責人為 A。使用課程 Classification 資料和固定 split，先跑 D 提供的 PCA + SVM，再從零訓練自建 CNN。Baseline 的前處理只有 Resize、Tensor Conversion 與 Normalization，mean / std 只從 train 計算。

| 實驗 | 設定 |
|---|---|
| `pca_svm` | Traditional Baseline |
| `E1` | Baseline CNN |
| `E2` | Baseline + 改善方法 A |
| `E3` | Baseline + 改善方法 B |
| `E4` | Baseline + 改善方法 A + B |

- `src/model.py`、`src/train.py`、`src/evaluate.py` 放模型、訓練和評估。由 D 同步 `data.py`、`seed.py`、`metrics.py`、`pca_svm.py`。
- `results/` 放實測 JSON、Training / Validation Curves 和 Confusion Matrix。以 validation Macro F1 選最佳 epoch 及模型，平手選參數較少者。
- 最終 checkpoint 為 `stage1_model.pth`。交件前依 [共用範本](../common/templates/) 補齊 `config.json`、`requirements.txt` 與可執行指令，並備妥 `stage1_report.pdf`。
