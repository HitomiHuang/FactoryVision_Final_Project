# Team05 開工環境

使用 Python 3.12.2。以下 PowerShell 指令從各自的 Stage 根目錄或解壓後的交接資料夾執行。已有團隊環境時可直接使用該環境的 Python。

```powershell
python --version
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
```

這次在 Windows、NVIDIA GeForce RTX 2060、driver 496.76 上驗證的是 CUDA 11.8 wheel。交接 ZIP 的 `evidence/environment.json` 記錄實際載入的版本及 CPU、GPU 運算檢查。使用相容的 NVIDIA 機器可執行下列指令。若使用 CPU，將第一行的 `cu118` 換成 `cpu`。兩種 wheel 的套件版本相同。[PyTorch 官方版本配對與安裝指令](https://pytorch.org/get-started/previous-versions/#v271)

```powershell
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

在同一個 PowerShell 視窗，將 `$root` 設為課程資料解壓後的 `STUDENT_PUBLIC` 位置。如果資料放在專案根目錄下的 `FactoryVision_DataSet/`，從 `stage1/`、`stage2/` 或 `stage3/` 執行時可直接使用下面的相對路徑。

```powershell
$root = "../FactoryVision_DataSet/STUDENT_PUBLIC"
```

資料沿用官方 train / validation split。分類和偵測各有一對完全相同的 PNG，為 `TRN_0504.png` 與 `VAL_0101.png`。偵測的兩份標註座標略有不同。交接 ZIP 的 `evidence/dataset_audit.json` 附兩對檔案的 SHA-256 與分割統計。請保留官方切分，並在報告解讀 validation 結果時說明此限制。
