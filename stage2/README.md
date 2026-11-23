# Stage 2 · Transfer Learning

負責人為李安爵（B）。使用課程 Classification 資料、固定 split 與 3 通道輸入。記錄公開 pretrained 權重的來源、模型官方的 Normalization，以及各策略的訓練參數。

| 實驗 | Architecture Family | Fine-tuning Strategy |
|---|---|---|
| `E1` | 甲 | 1 |
| `E2` | 甲 | 2 |
| `E3` | 乙 | 1 |
| `E4` | 乙 | 2 |

- `src/model.py`、`src/train.py`、`src/evaluate.py` 放模型、訓練和評估。由黃怡禎（D）同步 `data.py`、`seed.py`、`metrics.py`。
- `results/` 放實測 JSON 和圖，記錄總參數量、可訓練參數量、checkpoint 大小與推論時間。以 validation Macro F1 選最佳 epoch 及模型，平手選參數較少者。
- 最終 checkpoint 為 `stage2_model.pth`。交件前依 [共用範本](../common/templates/) 補齊 `config.json`、`requirements.txt` 與可執行指令，並備妥 `stage2_report.pdf`。
