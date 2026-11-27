# Stage 1 實作計畫（Baseline + Improved CNN）

負責人：葉文賢（A）　分支：`feat/stage1`　狀態：**已實作，待使用者確認**

依據：`FactoryVision_Final_Project.md`（PDF 轉錄）Stage 1 章節、`AGENTS.md`。

> 本檔是工作計畫，不放進 `Team05_Stage1.zip`。

---

## 1. 模型與訓練設定

| 項目 | 做法 |
|---|---|
| 模型 | 自建 CNN：4 個 block，每個 Conv3×3 → ReLU → MaxPool（通道 32 → 64 → 128 → 256）→ Global Average Pooling → Linear 輸出 6 類 |
| 對應 PDF | 有 Convolution、Activation、Pooling、Classifier；from scratch，不用 pretrained、不用 ResNet / VGG / MobileNet |
| 輸入 | 128×128 灰階，`channels=1` |
| Normalization | 訓練開始時用 `compute_train_normalization` **只用 train** 計算 mean / std，寫進 JSON 和 `config.json` |
| 訓練 | Adam、lr `0.001`、weight_decay `0`、batch `32`、`50` epochs、seed `42`、CrossEntropyLoss |
| 選最佳 epoch | 每組內選 validation **Macro F1** 最高的 epoch；平手選較早的 |
| Validation 用途 | 只用來選 epoch、選模型、比較；**不拿來訓練、不做 augmentation** |

## 2. 改善方法

| 方法 | 內容 | 參數 |
|---|---|---|
| **A = Data Augmentation** | 只套用在 train。翻轉 + 旋轉屬於同一組 pipeline，算 1 種方法（PDF FAQ） | RandomHorizontalFlip p=0.5、RandomVerticalFlip p=0.5、RandomRotation ±15° |
| **B = Dropout** | 放在 Linear 分類層前 | p=0.5 |

兩種方法的參數都寫進 `results/<exp>.json` 的 `settings`。

## 3. 實驗

| Exp | 內容 | Aug | Dropout |
|---|---|---|---|
| `pca_svm` | Traditional Baseline，在這台電腦**重跑**，確認與交接值 Macro F1 = 0.7608 一致 | — | — |
| E1 | Baseline CNN | ✗ | ✗ |
| E2 | Baseline + A | ✓ | ✗ |
| E3 | Baseline + B | ✗ | ✓ |
| E4 | Baseline + A + B | ✓ | ✓ |

每組 CNN 產出：

| 檔案 | 進 git？ |
|---|---|
| `results/<exp>.json`（格式照 AGENTS 6.4） | ✓ |
| `results/<exp>_curves.png`、`results/<exp>_cm.png` | ✓ |
| `checkpoints/<exp>.pth` | ✗ |

## 4. 選出 Final Improved CNN

- **只從 E2～E4 挑**，選 validation Macro F1 最高的；平手選參數較少的。
- 原因：PDF 規定 `stage1_model.pth` 是 **Final Improved CNN**，不能交 Baseline。
- 如果 E1 反而比 E2～E4 都高，照實寫進分析，不改數字。
- 選定的 checkpoint 複製成 `stage1_model.pth`，並填好 `config.json`。

## 5. 會改的檔案（都在 `stage1/`）

| 檔案 | 內容 |
|---|---|
| `src/model.py` | `build_model(dropout)` |
| `src/train.py` | `--exp E1~E4` 決定 Aug / Dropout；訓練、每 epoch 印一行、存 checkpoint / JSON / 圖 |
| `src/evaluate.py` | 讀 `config.json` → `build_model` → 載回 `stage1_model.pth` → 印 Metrics、存 CM，**印出分錯的圖檔名稱** |
| `config.json` | 填最終模型的真實設定 |
| `README.md` | 見第 6 節 |

**不動**：`src/data.py`、`src/metrics.py`、`src/seed.py`、`src/pca_svm.py`、`src/check_classification.py`（`common/` 的複本）。

## 6. README 新增內容

| 段落 | 內容 |
|---|---|
| 環境與最終模型 | Python 版本、torch / torchvision 版本、Final Model 名稱、checkpoint 檔名、pretrained source = from scratch、輸入尺寸、前處理（照 `common/templates/README.md`） |
| 重現指令 | `pca_svm`、E1～E4 訓練、`evaluate.py`，不需改 code 就能跑 |
| 結果表 | 欄位：`Exp / Setting / Accuracy / Macro P / Macro R / Macro F1 / Params / Size (MB) / Latency (ms)`，`pca_svm` + E1～E4 共 5 列 |
| 選模理由 | 為什麼選這組；看混淆矩陣說明哪幾類錯誤變少 |
| 錯誤案例 | 挑幾張分錯的圖，說明可能原因（哪兩類容易搞混） |
| Overfitting / Generalization 分析 | 每組比較最佳 epoch 時 train 和 validation 的 accuracy / loss 差距；看加 Aug、Dropout 後差距有沒有縮小 |

**所有數字從 `results/*.json` 直接抄**，不手算、不編、不四捨五入兩次。小數 4 位，大小 / 時間 2 位。

## 7. 做事順序

1. 在 `stage1/.venv` 建環境（Python 3.12，CPU 版 torch 2.7.1）
2. 重跑 `pca_svm`，對數字
3. 寫 `model.py`、`train.py`、`evaluate.py`
4. E1 跑 1 個 epoch 測試：能存、能載回、`evaluate.py` 的數字和訓練時一樣
5. 正式跑 E1～E4
6. 從 E2～E4 選最好的 → `stage1_model.pth` → 填 `config.json`
7. 寫 README
8. 給使用者看 → 同意後才 commit、push

## 8. 驗收條件

- [x] `pca_svm` 重跑的 Macro F1 = 0.7608
- [x] E1～E4 各有 JSON、curves、CM，JSON 通過 `python -m json.tool`
- [x] `evaluate.py` 載入 `stage1_model.pth` 算出的 Macro F1 = 該組 JSON 的值
- [x] `config.json` 合法，10 個欄位沒有 `null`，內容和 `stage1_model.pth` 一致
- [x] `src/` 沒有絕對路徑
- [x] README 每個數字都能在 `results/*.json` 找到

## 10. 實作結果（2026-11-25）

| 項目 | 結果 |
|---|---|
| 最終模型 | **E3（Baseline + Dropout）**，best epoch 48，validation Macro F1 0.9926 |
| 發現 | Baseline E1 已達 0.9926 且沒有 overfitting，A、B 兩種方法沒有再提升；E4 最低（0.9850） |
| 與原計畫的差異 | ① `results/pca_svm.json` 換成本機重跑版本：指標完全相同，只有 latency 改成與 CNN 同一台 CPU 量測（0.33 ms）。② `evaluate.py` 改用 `utf-8-sig` 讀 `config.json`，相容 Windows PowerShell 寫出的 BOM。③ 錯誤清單沿用 `pca_svm.py` 的做法，存在 JSON 的 `misclassified` 欄位 |

## 9. 要知道的事

| 事情 | 影響 |
|---|---|
| 這台**沒有 NVIDIA 顯卡**（Intel UHD 770） | 只能用 CPU 訓練。Latency 是 CPU 數字，和隊上 RTX 2060 不同，之後可在那台重量 |
| Python 是 3.12.10，團隊規定 3.12.2 | 小版本差異，不影響結果 |
| 資料集位置 | `../../FactoryVision_DataSet/STUDENT_PUBLIC`（從 `stage1/` 算起） |
| 不在這次範圍 | `stage1_report.pdf`、ZIP 打包。**注意：PDF 規定結果要出現在報告裡才算分**，README 內容之後要搬進報告 |
