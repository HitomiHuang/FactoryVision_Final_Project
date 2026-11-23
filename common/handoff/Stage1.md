# Team05 · Stage 1 交接給葉文賢（A）

在 Stage 1 工作目錄開啟 PowerShell，依環境說明設定 `$py` 和 `$root`。下列三個指令已用課程資料跑過。[安裝環境](SETUP.md)

```powershell
& $py src/check_classification.py --data_root $root --channels 1
& $py src/data.py --data_root $root --input_size 128 --channels 1
& $py src/pca_svm.py --data_root $root --kernel rbf --gamma scale
```

第一個指令核對 loader、標籤順序、檔名與混淆矩陣。第二個只讀 train，在 128 × 128 下得到 mean `0.5070540018850411`、std `0.2110269081061105`。選用其他 CNN 輸入尺寸時用該尺寸重跑，將對應值寫入最終 config。

傳統基準將灰階圖以 bilinear 縮至 64 × 64，像素除以 255 並攤平。對 1,260 張 train 執行 float64 完整 SVD、保留 64 個主成分，再擬合 RBF SVM（C=10、gamma=scale、seed=42），在 270 張 validation 評估。PCA、SVM 都只對 train 呼叫 fit。[scikit-learn 官方資料切分說明](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage)

`--kernel rbf --gamma scale` 是本表的預設設定。`--kernel` 也支援 `linear`。使用 RBF 時，`--gamma` 可填 `scale`、`auto` 或非負數值。測試其他設定時，用獨立的 `--out_dir` 和 `--checkpoint` 保存該次結果。

| 實驗 | Accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|
| PCA + SVM | 0.7630 | 0.7920 | 0.7630 | 0.7608 |

`results/pca_svm.json` 保留未四捨五入的指標、設定、混淆矩陣與 64 筆錯分檔名，圖在 `results/pca_svm_cm.png`。序列化的模型在 `checkpoints/pca_svm.pkl`，其 SHA-256 記錄於實驗 JSON。SVM 的延遲在 CPU 量測，沒有 CNN 的 epoch 或可訓練權重數，對應欄位使用 JSON null。

接續工作是自建 CNN 的 E1–E4。每個 batch 可用 `for images, labels, filenames in train_loader` 接收。驗證用 `evaluate(model, val_loader, device)["metrics"]["macro_f1"]` 選最佳 epoch。正式 `stage1_model.pth` 是葉文賢選定的 Improved CNN，最後依該模型填好 `config.json` 並撰寫 Stage 1 報告。
