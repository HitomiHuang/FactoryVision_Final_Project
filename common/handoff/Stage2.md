# Team05 · Stage 2 交接給李安爵（B）

在 Stage 2 工作目錄開啟 PowerShell，依環境說明設定 `$py` 和 `$root`。先用下列指令驗證三通道讀檔與六類評估。[安裝環境](SETUP.md)

```powershell
& $py src/check_classification.py --data_root $root --channels 3 --input_size 224
```

本次在 Windows 另以 `--num_workers 2` 跑過，取得 `batch=(32, 3, 224, 224)`、`train=1260`、`validation=270`。固定猜第一類會答對 45 張，這是檢查標籤對應的已知答案。一般開發可採用預設 `--num_workers 0`。

在 `src/train.py` 可依下列方式接入。範例使用已安裝 torchvision 的 ResNet18 ImageNet 權重定義取得 mean / std。選用下一個架構時改用該架構的官方權重定義，並記錄來源、輸入尺寸與實際 preprocessing。[Torchvision 的 ResNet18 權重說明](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)

```python
from torchvision.models import ResNet18_Weights
from data import get_dataloaders
from metrics import evaluate
from seed import set_seed

set_seed(42)
preprocessing = ResNet18_Weights.IMAGENET1K_V1.transforms()
train_loader, val_loader = get_dataloaders(
    args.data_root, args.input_size, preprocessing.mean, preprocessing.std,
    args.batch_size, args.num_workers, channels=3,
)
# Each batch contains images, labels and filenames.
# After training one epoch, evaluate the six-logit classifier.
result = evaluate(model, val_loader, args.device)
val_f1 = result["metrics"]["macro_f1"]
```

接續工作是以兩個不同家族、兩種策略完成 E1–E4。共用 `metrics.py` 已提供曲線、混淆矩陣、總參數與可訓練參數、實際 checkpoint 大小及 batch 1 推論時間。最終選模後，再將與 `stage2_model.pth` 一致的架構、策略、pretrained source、尺寸及 normalization 寫進 `config.json`。
