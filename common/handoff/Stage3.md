# Team05 · Stage 3 給 C 的交接

在 Stage 3 工作目錄開啟 PowerShell，依環境說明設定 `$py` 和 `$root`。此交接採用有 NMS 的 Ultralytics YOLO detector，可直接使用課程現成的 YOLO 標註。[安裝環境](SETUP.md) · [Ultralytics 偵測資料格式](https://docs.ultralytics.com/datasets/detect/)

```powershell
& $py src/prepare_yolo.py --data_root $root --out_dir ../data/factoryvision_yolo
```

輸出為 `../data/factoryvision_yolo/data.yaml` 和 `images/{train,validation}`、`labels/{train,validation}`。可將 YAML 路徑傳給 Ultralytics 的 `train(data=...)`。程式讀取原始資料、檢查類別順序及每個框，並把圖片和標註逐檔複製到新目錄。第二次執行會比對既有檔案內容。本次 Ultralytics 實讀 train 1,260 張、validation 270 張，標註共 2,924 與 641 個框。

訓練由 C 負責，從公開 pretrained 權重自行訓練。團隊選模採 validation **mAP@0.5**。目前鎖定的 Ultralytics 8.3.239 以 **mAP@0.5:0.95** 決定預設 `best.pt`。可在 `train(..., save_period=1)` 保留各 epoch 的權重，再依 validation mAP@0.5 最高的 epoch 選出對應檔案，複製為 `stage3_model.pt`。[該版本 fitness 的定義](https://github.com/ultralytics/ultralytics/blob/v8.3.239/ultralytics/utils/metrics.py#L955-L958) · [epoch 權重儲存邏輯](https://github.com/ultralytics/ultralytics/blob/v8.3.239/ultralytics/engine/trainer.py#L617-L623)

把選定的 detector、pretrained source、輸入尺寸、confidence 及 NMS IoU 填入 `config.json` 後執行下列指令。

```powershell
& $py src/visualize.py --data_root $root --exp D1
```

- `results/vis/val_01.png` 至 `val_10.png` 為等距抽取的 validation 對照，左圖以綠色畫 GT，右圖以紅色畫預測並標類別、confidence。
- `results/failure/` 會從不同圖片挑出最多五個實際 FP、FN 或 LOC 案例。符合條件的圖片不足時，如實輸出可取得的案例數。
- `results/D1_visualization.json` 記錄原始圖檔、選定案例的框與分數，以及全 validation 的 confidence 掃描。預設比較 0.10、0.25、0.50、0.75 和所選門檻，可用 `--thresholds` 指定。此處在固定 NMS 後，以類別相同、IoU ≥ 0.50、信心由高至低一對一配對，按所有框合計 TP / FP / FN 算 P/R。LOC 是團隊案例標記，指尚未配對且同類的框在 0.10 ≤ IoU < 0.50，計數時同時構成一個 FP 和一個 FN。
- C 的正式 `evaluate.py` 另需回報 Precision、Recall、mAP@0.5 與實際推論時間，並說明各自的門檻與平均方式。視覺化 JSON 的 P/R 是指定門檻的 micro P/R。額外實驗可用不同 `--out_dir` 保存圖表。[Ultralytics 預測輸出](https://docs.ultralytics.com/modes/predict/)
