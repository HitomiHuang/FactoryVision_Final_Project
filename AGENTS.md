# AGENTS.md — FactoryVision 團隊 repo

給 AI coding agent（Claude Code、Codex、Copilot、Cursor…）讀的工作規則。組員也要讀。
規則來源：課程 PDF `FactoryVision_Final_Project.pdf`（標 `p.N` 的是 PDF 頁碼）。標 💡 的是團隊約定，不是 PDF 規定。

## 0. 專案一句話

用鋼板表面瑕疵資料集（6 類）做三件事：Stage 1 自建 CNN 分類、Stage 2 Transfer Learning 分類、Stage 3 Object Detection。每個 Stage 交一個 ZIP，最後交 `TeamXX_Final_Report.pdf`。

## 1. 先確認你在改誰的資料夾

| 資料夾 | 負責人 | 最後變成 |
|---|---|---|
| `common/` | D | 共用程式正本 + `templates/` |
| `stage1/` | A | `TeamXX_Stage1.zip` |
| `stage2/` | B | `TeamXX_Stage2.zip` |
| `stage3/` | C | `TeamXX_Stage3.zip` |
| `final_report/` | D | `TeamXX_Final_Report.pdf` |

- 💡 **只改使用者負責的資料夾。** 不確定使用者是誰、負責哪個 Stage，先問。
- 💡 `stage1/src/` 和 `stage2/src/` 裡的 `data.py`、`seed.py`、`metrics.py`、`stage1/src/pca_svm.py` 是 `common/` 的**複本，不要直接改**。需要改，請使用者找 D 改 `common/`，再由 D 同步。
- 💡 `common/` 函式的名稱和參數已凍結（見第 5 節），不要自行改簽名。

## 2. 🚨 紅線（違反 → 整個專題最高 0 分，p.33）

| 不可以 | 說明 |
|---|---|
| Data leakage | Validation **只能**用於調參、選模型、Early Stopping、比較（p.5）。**不可**拿 validation 圖片訓練、不可把 validation 併進 train、不可用 validation 做 augmentation 來源。 |
| 用外部資料 | 只能用課程給的 FactoryVision Public Dataset；Stage 1–2 只能用 Classification 資料（p.5）。 |
| 用網路上已 fine-tune 好的 checkpoint | 禁止（p.5）。Stage 2/3 只能從公開的 **pretrained**（如 ImageNet / COCO）權重開始自己訓練。 |
| 捏造或修改數字 | 報告、README、`results/*.json` 的每個數字都必須是**真的跑出來的**。AI **絕對不可**編造、估計、「填個合理的值」。沒跑就寫 `TODO`。 |
| 抄襲 | report 和 code 都算（p.33）。不要整段貼網路上別組或公開 repo 的專題程式。 |

- p.33：允許用 AI 協助 coding、debugging、查 API、理解概念，但**提交內容由學生負責**。所以 AI 寫的 code，使用者要看得懂、講得出來。

## 3. 固定規格（不可改）

| 項目 | 值 | 來源 |
|---|---|---|
| 類別順序 `class_order` | `["crazing","inclusion","patches","pitted_surface","rolled_in_scale","scratches"]` | p.28 |
| Random seed | `42` | p.5 |
| Checkpoint 檔名 | `stage1_model.pth`、`stage2_model.pth`、`stage3_model.pt`（**Stage 3 是 `.pt`**） | p.25–27 |
| Stage 3 `bbox_format` | `"xyxy_pixel"`；不用 NMS 的模型 `nms_iou_threshold` 填 `null` | p.28 |
| 分類主要指標 | Macro F1；還要 Accuracy、Precision、Recall、Confusion Matrix。不得只用 Accuracy 說模型比較好 | p.6 |
| 偵測主要指標 | mAP@0.5；還要 Precision、Recall、Inference Time、Visualization | p.6 |
| 路徑 | **禁止絕對路徑**（`C:\`、`/home/...`、`/Users/...`）。資料用 `--data_root` 傳；checkpoint 用 ZIP 內相對路徑 | p.24 |
| `config.json` | 合法 JSON，欄位照 `common/templates/`，內容要和最終 checkpoint 一致 | p.28 |
| `requirements.txt` | 一行一個，建議 `package==version`；禁止本機路徑、`-e`、GitHub URL、私人 wheel | p.29 |
| `src/` | 至少 `model.py`、`train.py`、`evaluate.py`；主要 training logic 不可放在外部網址或雲端 notebook | p.29 |
| README 指令 | 不可要求「改 code 才能跑」 | p.29 |

## 4. 各 Stage 的規則

### Stage 1（A）

- Baseline CNN 必須 **from scratch**：不可用 pretrained weights，不可用 ResNet / VGG / MobileNet 等既有架構（p.8）。
- Baseline 只能用 Resize、Tensor Conversion、Normalization（p.8）。
- 至少 2 種**不同的** Improvement Method（p.9）。Flip + Rotation 只算 **1 種**（同一組 augmentation，p.34）。
- Ablation 必須做滿 E1 Baseline、E2 +A、E3 +B、E4 +A+B（p.10）。
- 每個方法要記錄參數，例如 Dropout p、weight_decay、augmentation 參數（p.9）。
- Traditional Baseline（PCA + SVM）在 `src/pca_svm.py`。
- 資料是 **8-bit 灰階** PNG 200×200；`get_dataloaders(channels=1 或 3)`。

### Stage 2（B）

- 至少 2 種**不同 Architecture Family**（ResNet18 和 ResNet50 算同一家，p.34）× 2 種 Strategy，**4 組全做**（p.14–15）。不可 A 只做 Strategy 1、B 只做 Strategy 2。
- 記錄 pretrained 來源（p.14）。Partial：列 frozen / trainable layers；Last-k：寫出 k；LoRA：target module、r、alpha、trainable params；L2-SP：係數（p.16）。
- 要報 Parameter、Trainable Parameter、Model Size（p.17）。
- 灰階圖要轉 3 通道：`channels=3`。

### Stage 3（C）

- 至少 1 個 detector（YOLO / Faster R-CNN / SSD / DETR，p.20）。
- ⚠️ 資料夾叫 `labels_yolo/`，Ultralytics 要的是 `labels/`。💡 **不要改原始資料集**；在 `data/` 下另外建一份整理好的結構（或用 symlink / 複製），並把轉換步驟寫進 `src/` 和 README。
- 要報 Precision、Recall、mAP@0.5、Inference Time（p.21）。
- 至少 10 張 GT vs Prediction 對照圖、至少 5 個 FP / FN / 定位不準案例（p.21–22），程式在 `src/visualize.py`。
- 報告要說明 IoU、Confidence Threshold；有用 NMS / Anchor / Object Query 也要說明（p.21）。

## 5. 共用程式約定 💡

| 函式（`from data import ...` 同層 import） | 用途 |
|---|---|
| `data.CLASS_ORDER` | 固定類別順序 |
| `data.get_dataloaders(data_root, input_size, mean, std, batch_size, num_workers, channels=3, train_transform=None, seed=42)` | train / validation DataLoader |
| `data.load_numpy(data_root, input_size)` | 給 PCA + SVM |
| `seed.set_seed(42)` | 固定亂數 |
| `metrics.compute_metrics(y_true, y_pred)` | Accuracy、Macro P/R/F1、Confusion Matrix |
| `metrics.evaluate(model, loader, device)` | 指標 + `y_true`、`y_pred`、檔名 |
| `metrics.plot_curves(history, path)` / `plot_confusion_matrix(cm, path)` | 畫圖 |
| `metrics.model_stats(model, ckpt_path)` | 參數量、checkpoint 大小 |
| `metrics.measure_latency(model, input_size, device, runs)` | 推論時間 |

- **import 一律同層**：`from data import get_dataloaders`。不要寫 `from common.data import ...`。
- **從 Stage 資料夾執行**：`cd stage1` 再 `python src/train.py --data_root ../data/STUDENT_PUBLIC`。這樣複製成 ZIP 後不用改 code。
- 推論時間量法（A、B、C 一致）：同一台電腦、同一張 GPU；batch size 1；暖機 10 次；跑 100 次取平均，單位 ms / 張；只算模型推論（Stage 3 含 NMS），不含讀檔。
- `compute_metrics` / `evaluate` 回傳的 key 名稱必須跟第 6.4 節的 JSON 欄位一樣。
- A、B 畫圖**一律用** `plot_curves`、`plot_confusion_matrix`，不要自己另寫一套，圖才會長得一樣。

## 6. 統一格式（大家做出來要一樣）💡

目的：三個 Stage 的程式、結果、圖、報告長得一樣，D 寫期末報告時可以直接比較。**沒寫到的地方，先照別人已經寫好的 Stage 做；還是不確定就問，不要自創。**

### 6.1 環境（D 第 1 天填好，全隊照用）

| 項目 | 全隊統一版本 |
|---|---|
| Python | `TODO`（例如 3.11.x） |
| `torch` / `torchvision` | `TODO` |
| `numpy` / `scikit-learn` / `matplotlib` / `Pillow` | `TODO` |
| `ultralytics`（只有 Stage 3） | `TODO` |

- 各 Stage 的 `requirements.txt` 用**同一組版本**，只列該 Stage 真的有用到的套件。
- 要加新套件：先在群組講，D 把版本加進上表，再加進 `requirements.txt`。
- 裝置：`--device` 預設 `auto`（有 CUDA 用 `cuda`，否則 `cpu`）。

### 6.2 指令列參數名稱（`train.py`、`evaluate.py` 都一樣）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--data_root` | 必填 | 指到 `STUDENT_PUBLIC/` |
| `--checkpoint` | `stageN_model.pth`（Stage 3 `.pt`） | 相對路徑 |
| `--exp` | `E1` | 實驗編號，決定輸出檔名 |
| `--epochs` / `--batch_size` / `--lr` / `--weight_decay` | 各 Stage 自訂 | |
| `--input_size` | 各 Stage 自訂 | 一個整數，正方形 |
| `--seed` | `42` | |
| `--device` | `auto` | |
| `--num_workers` | `0` | Windows 上最不會出錯 |
| `--out_dir` | `results` | |
| `--conf` / `--iou`（只有 Stage 3） | 自訂 | confidence / NMS IoU threshold |

- 不要用別的名字（例如 `--data`、`--lr_rate`、`--bs`）。
- 程式結構固定：`parse_args()` → `main()` → `if __name__ == "__main__": main()`。

### 6.3 實驗編號與存檔

| Stage | 實驗編號 | 意思 |
|---|---|---|
| 1 | `E1`–`E4` | Baseline / +A / +B / +A+B（p.10）；另加傳統方法 `pca_svm` |
| 2 | `E1`–`E4` | `E1` 架構甲+策略1、`E2` 架構甲+策略2、`E3` 架構乙+策略1、`E4` 架構乙+策略2 |
| 3 | `D1`、`D2`… | 每個 detector 設定一個編號 |
| 全部 | `E5` 以後 | 額外實驗，接著編 |

- 每個實驗的 checkpoint 存 `checkpoints/E1.pth`（不進 git）；選定的最終模型**複製**成 `stageN_model.pth`。
- Stage 1、2：存 `model.state_dict()`。`model.py` 要提供 `build_model(...)`，`evaluate.py` 讀 `config.json` → `build_model` → `load_state_dict`。
- Stage 3：用框架原生格式（例如 Ultralytics 的 `best.pt`），改名成 `stage3_model.pt`。
- **選最佳 epoch / 最終模型的標準**：Stage 1、2 用 validation **Macro F1** 最高；Stage 3 用 validation **mAP@0.5** 最高。平手時選參數量較小的。
- 不重新切資料，不做 k-fold；只用給定的 train / validation。
- Normalization：Stage 1 的 mean / std **只用 train 算**；Stage 2 用 pretrained 模型官方的值（例如 ImageNet）。數值寫進 `config.json`。

### 6.4 結果 JSON 格式（`results/<exp>.json`）

分類（Stage 1、2）：

```json
{
  "exp": "E1",
  "stage": 1,
  "description": "Baseline CNN",
  "settings": {"model_name": "...", "input_size": 128, "epochs": 50, "batch_size": 32, "lr": 0.001, "weight_decay": 0.0, "seed": 42},
  "best_epoch": 37,
  "metrics": {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0},
  "confusion_matrix": [[0, 0, 0, 0, 0, 0]],
  "params": 0,
  "trainable_params": 0,
  "checkpoint_mb": 0.0,
  "latency_ms": 0.0,
  "latency_device": "NVIDIA ...",
  "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
}
```

偵測（Stage 3）：`metrics` 改成 `{"precision", "recall", "map50"}`，另加 `"conf_threshold"`、`"nms_iou_threshold"`。

- 改善方法的參數（Dropout p、augmentation 參數、frozen layers、LoRA r…）全部放進 `settings`。
- 指標存 **0–1 的小數**，不要存百分比。
- 單位固定：大小 `MB`（bytes ÷ 1024²），時間 `ms / 張`。
- `confusion_matrix` 的列 = 真實類別、行 = 預測類別，順序照 `CLASS_ORDER`。

### 6.5 圖檔

| 圖 | 檔名 |
|---|---|
| Training / Validation Curves | `results/<exp>_curves.png` |
| Confusion Matrix | `results/<exp>_cm.png` |
| Stage 3 GT vs Prediction | `results/vis/val_01.png` … `val_10.png` |
| Stage 3 失敗案例 | `results/failure/fail_01_FN.png`（類型寫 `FP` / `FN` / `LOC`） |

- 圖上文字用**英文**；標題要有實驗編號，例如 `E2: Baseline + Dropout`。
- `dpi=150`；類別名稱照 `CLASS_ORDER` 順序。
- Confusion Matrix 顯示**張數**（不是百分比）。
- Stage 3 框的顏色：**GT 綠色、Prediction 紅色**，Prediction 框上標類別和 confidence。

### 6.6 程式風格

- PEP 8、縮排 4 個空格。函式和變數用 `snake_case`、class 用 `PascalCase`、常數用 `UPPER_CASE`。
- 註解和 docstring 用英文；每個函式至少一行 docstring。
- `src/` 只放 `.py`。notebook 可以自己用來試，但不要放進 `src/`，也不要讓 `src/` 依賴它。
- 訓練時每個 epoch 印一行，格式一樣：
  `[E1] epoch 3/50 train_loss=0.1234 train_acc=0.9512 val_loss=0.2345 val_acc=0.9111 val_f1=0.9087`

### 6.7 報告格式（`stageN_report.pdf`、期末報告）

- 內文中文，專有名詞與指標名稱用英文（例如 Macro F1、mAP@0.5）。
- 數字一律 **4 位小數**（例如 `0.9123`）；大小、時間 **2 位小數**。
- 分類結果表欄位固定這個順序：

  | Exp | Setting | Accuracy | Macro P | Macro R | Macro F1 | Params | Size (MB) | Latency (ms) |
  |---|---|---|---|---|---|---|---|---|

- 偵測結果表：`Exp | Detector | Input | Conf | NMS IoU | Precision | Recall | mAP@0.5 | Latency (ms)`。
- 圖表編號 `Figure 1`、`Table 1`，每張都要有說明文字。
- 報告裡每個數字都要能在 `results/*.json` 找到；**直接從 JSON 抄，不要手算或四捨五入兩次**。

## 7. Git 規則 💡

- **不要 commit**：`data/`、`*.pth`、`*.pt`、`*.zip`、`runs/`（已在 `.gitignore`）。模型檔用雲端硬碟分享。
- **要 commit**：`results/*.json` 和圖。
- 每人一個 branch，經 PR 合併進 `main`，不要直接 push `main`。PR review：C 看 A、A 看 B、B 看 C。
- Commit message 用 Conventional Commits，例如 `feat(stage1): add dropout experiment`。
- **AI 不可自行 commit、push、開 PR、刪 branch**，除非使用者明確要求。

## 8. AI 做事的方式

- **說「完成」前要實際跑過**，並貼出執行輸出。「看起來應該對」不算完成。
- 改動要小，只動跟需求有關的檔案。不要順手重構、改格式或改別人的檔案。
- 程式要讓 TA 看得懂（p.29 要求可讀、與報告相符）：變數命名清楚、註解用英文、不要過度抽象。
- 新增套件時，同步加進該 Stage 的 `requirements.txt` 並寫版本；不要升級既有套件版本。
- 訓練要設 `set_seed(42)`；超參數寫在 argparse 或 `config.json`，不要寫死在程式深處。
- 不確定某件事是 PDF 規定還是建議時，說清楚，不要當成規定。
- 寫新程式前，先看 `common/` 和其他 Stage 已經寫好的檔案，**照同樣的寫法**，不要另起一套。

## 9. 交件前檢查（每個 Stage）

- [ ] 沒有剩下 `TODO` 或 `TeamXX`：`grep -rn "TODO\|TeamXX" stageN`
- [ ] `python -m json.tool stageN/config.json` 通過，內容和最終 checkpoint 一致
- [ ] 在**乾淨環境**照 README 指令跑得起來：`pip install -r requirements.txt` → `python src/evaluate.py --data_root <path>`
- [ ] 沒有絕對路徑：`grep -rnE '[A-Za-z]:[\\/]|/home/|/Users/' stageN/src`（`https://` 會被抓到，屬誤判可忽略）
- [ ] ZIP 名稱和根資料夾都是 `TeamXX_StageN`，解壓後沒有多一層；≤ 1 GB（p.24）
- [ ] 報告的數字和 `results/*.json` 一致
- [ ] `requirements.txt` 的版本和第 6.1 節一樣
- [ ] 結果 JSON、圖檔名稱、報告表格欄位照第 6 節
