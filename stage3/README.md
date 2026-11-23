# Stage 3 · Object Detection

負責人為謝允柔（C）。使用課程 Detection 資料，實驗從 `D1` 起編號。若採用 Ultralytics，將原始 `labels_yolo/` 對應到 `data/` 內另建的 `labels/` 目錄，保留原始資料與 split，並將轉換程式與執行方式納入交件。

- `src/model.py`、`src/train.py`、`src/evaluate.py` 放 detector、訓練和評估。`src/visualize.py` 產生對照與失敗案例。
- `results/` 放實測 Precision、Recall、mAP@0.5 與推論時間。以 validation mAP@0.5 選最佳 epoch 及模型，平手選參數較少者。
- `results/vis/` 放至少 10 張 GT vs Prediction 對照圖；`results/failure/` 放至少 5 個 FP、FN 或定位不準的案例。
- 最終 checkpoint 為 `stage3_model.pt`。交件前依 [共用範本](../common/templates/) 補齊 `config.json`、`requirements.txt` 與可執行指令，並備妥 `stage3_report.pdf`。
