# 共用程式

D 維護這裡的正本。函式名稱、參數與輸出欄位依 [AGENTS.md](../AGENTS.md) 第 5 節與第 6.4 節。

| 模組 | 用途 | 完成後同步到 |
|---|---|---|
| `data.py` | 固定類別順序、分類 DataLoader、PCA + SVM 的 NumPy 資料 | Stage 1、2 |
| `seed.py` | 固定亂數種子 | Stage 1、2；Stage 3 依實作需要 |
| `metrics.py` | 分類指標、評估、畫圖、模型大小與推論時間 | Stage 1、2 |
| `pca_svm.py` | Traditional Baseline | Stage 1 |

各 Stage 複本放在 `src/`，用同層 import。需要更動時，由 D 先修改正本再同步。模組目前保留檔名與用途，完成實作後再同步。

[templates/](templates/) 放三份 config 欄位範本及共用的 README、requirements 範本。
