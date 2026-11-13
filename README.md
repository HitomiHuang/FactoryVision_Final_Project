# FactoryVision 期末專題

使用課程提供的鋼板表面瑕疵資料集，完成自建 CNN 分類、Transfer Learning 分類與 Object Detection。團隊規則與交件檢查見 [AGENTS.md](AGENTS.md)。

| 目錄 | 負責人 | 內容與交件 |
|---|---|---|
| [`common/`](common/) | D | 共用程式正本與 `templates/` |
| [`stage1/`](stage1/) | A | 自建 CNN、PCA + SVM 與 Ablation，交 `TeamXX_Stage1.zip` |
| [`stage2/`](stage2/) | B | Transfer Learning 的 2 × 2 實驗，交 `TeamXX_Stage2.zip` |
| [`stage3/`](stage3/) | C | Object Detection 與失敗案例，交 `TeamXX_Stage3.zip` |
| [`final_report/`](final_report/) | D | 跨 Stage 分析，交 `TeamXX_Final_Report.pdf` |
| `data/` | 本機 | 課程資料及 Stage 3 轉換後的資料，由組員自行準備 |

目前是專案骨架。各 Stage 的 `src/` 保留預定的模組位置，`results/` 用來收錄實際執行後的 JSON 和圖。正式的 `config.json`、`requirements.txt`、checkpoint 和報告由各負責人補齊。

## 開發流程

1. D 確認 [AGENTS.md](AGENTS.md) 第 6.1 節 的環境版本，完成共用程式後同步到 Stage 1、2 的 `src/`。各 Stage 從 [共用範本](common/templates/) 建立自己的交件檔。
2. 將課程資料放在 `data/STUDENT_PUBLIC/`，或用 `--data_root` 傳入現有 `STUDENT_PUBLIC/` 的相對路徑。沿用原始 train / validation split，seed 固定為 `42`。Validation 用於調參、選模、Early Stopping 和比較。
3. 訓練與評估程式完成後，從對應的 Stage 根目錄執行。實驗結果 JSON 和圖提交到該 Stage 的 `results/`，資料集、checkpoint、`runs/` 和 ZIP 留在本機。
4. 組員在各自的分支開發，透過 PR 合併至 `main`。交件前用乾淨環境照 Stage README 驗證，並核對報告數字與結果 JSON。
