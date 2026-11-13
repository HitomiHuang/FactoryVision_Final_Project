# 交件範本

將對應的 `stageN.config.json` 複製到 Stage 根目錄並命名為 `config.json`。範本的 `null` 表示待填欄位，依最終 checkpoint 補齊。Stage 3 使用不需 NMS 的模型時，`nms_iou_threshold` 保留 `null`。

各 Stage 的 README 應寫明以下資訊。

- Python 與主要 framework 版本
- Final Model、checkpoint 檔名，以及適用時的 pretrained source
- 輸入尺寸、preprocessing，以及 Stage 3 的 confidence 和 NMS IoU thresholds
- 從 Stage 根目錄安裝依賴、訓練、評估的可執行指令
- Stage 3 的資料轉換與視覺化指令

用 `requirements.txt` 範本建立各 Stage 的依賴清單，版本採用 [AGENTS.md](../../AGENTS.md) 第 6.1 節確認後的版本。
