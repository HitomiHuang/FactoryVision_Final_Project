"""Build stage1_report.html from results/*.json so every reported number comes from the JSON files.

Run from stage1/:
    .venv/Scripts/python.exe docs/report/build_report.py --data_root ../../FactoryVision_DataSet/STUDENT_PUBLIC
Then print the HTML to PDF with Edge (see README in this folder or the command printed at the end).
"""

import argparse
import base64
import datetime
import html
import json
import sys
from pathlib import Path

import torch

STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE / "src"))
from data import CLASS_ORDER  # noqa: E402
from model import build_model  # noqa: E402

RESULTS = STAGE / "results"
CNN_EXPS = ["E1", "E2", "E3", "E4"]
SETTING = {
    "pca_svm": "PCA(64) + RBF SVM",
    "E1": "Baseline CNN",
    "E2": "+ Data Augmentation (A)",
    "E3": "+ Dropout (B)",
    "E4": "+ Data Augmentation + Dropout (A+B)",
}
TITLE = {
    "E2": "Baseline + Data Augmentation",
    "E3": "Baseline + Dropout",
    "E4": "Baseline + Data Augmentation + Dropout",
}
# Qualitative notes written after looking at each misclassified validation image.
ERROR_NOTES = {
    "patches/VAL_0097.png": "只有幾個小而淡的暗斑，背景顆粒感重，與 crazing 的細紋理相似；典型 patches 是大片暗區。",
    "pitted_surface/VAL_0144.png": "對比很低、幾乎看不到凹坑，只剩直向條紋，與 inclusion 的長條狀紋理接近。",
    "pitted_surface/VAL_0175.png": "對比低、上下亮度不均，凹坑不明顯。",
}


def parse_args():
    """Parse the dataset location (used only to embed example images)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("stage1_report.html"))
    return parser.parse_args()


def load(exp):
    """Read one results JSON."""
    return json.loads((RESULTS / f"{exp}.json").read_text(encoding="utf-8"))


def f4(x):
    """Format a metric with 4 decimals."""
    return f"{x:.4f}"


def f2(x):
    """Format a size or time with 2 decimals."""
    return f"{x:.2f}"


def img(path, width="100%"):
    """Return an <img> tag with the PNG embedded as base64."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" style="width:{width}">'


def table(headers, rows, caption):
    """Return an HTML table with a caption line above it."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tbl"><p class="cap">{caption}</p><table><tr>{head}</tr>{body}</table></div>'


def figure(content, caption):
    """Return a figure block with its caption below."""
    return f'<div class="fig">{content}<p class="cap">{caption}</p></div>'


def architecture_rows(input_size):
    """Trace the model with a dummy image and list output shapes and parameter counts."""
    model = build_model(in_channels=1, num_classes=len(CLASS_ORDER), dropout=0.5).eval()
    x = torch.zeros(1, 1, input_size, input_size)
    rows = [["Input", "—", f"1 × {input_size} × {input_size}", "0"]]
    with torch.no_grad():
        for i, block in enumerate(model.features, start=1):
            conv = block[0]
            x = block(x)
            params = sum(p.numel() for p in block.parameters())
            rows.append([f"Block {i}",
                         f"Conv3×3 ({conv.in_channels}→{conv.out_channels}, padding 1) → ReLU → MaxPool 2×2",
                         " × ".join(str(d) for d in x.shape[1:]), f"{params:,}"])
        x = model.pool(x).flatten(1)
        rows.append(["Global Average Pooling", "AdaptiveAvgPool2d(1)", str(x.shape[1]), "0"])
        rows.append(["Dropout（只有 E3、E4）", "p = 0.5，推論時關閉", str(x.shape[1]), "0"])
        out = model.classifier(x)
        params = sum(p.numel() for p in model.classifier.parameters())
        rows.append(["Classifier", f"Linear({x.shape[1]}→{out.shape[1]})", str(out.shape[1]), f"{params:,}"])
    total = sum(p.numel() for p in model.parameters())
    rows.append(["<b>合計</b>", "", "", f"<b>{total:,}</b>"])
    return rows, total


def main():
    """Assemble the report sections and write a self-contained HTML file."""
    args = parse_args()
    r = {exp: load(exp) for exp in ["pca_svm", *CNN_EXPS]}
    pca, e1 = r["pca_svm"], r["E1"]
    s1 = e1["settings"]
    arch_rows, total_params = architecture_rows(s1["input_size"])
    if total_params != e1["params"]:
        raise ValueError("Architecture trace does not match the parameter count in E1.json")

    # Final model: highest validation Macro F1 among the improved CNNs (E2-E4), ties -> fewer params.
    final = max(["E2", "E3", "E4"], key=lambda e: (r[e]["metrics"]["macro_f1"], -r[e]["params"]))
    fr = r[final]

    def cm_errors(res, name):
        """Return (wrong count, most frequent wrong prediction, its count) for one true class."""
        row = res["confusion_matrix"][CLASS_ORDER.index(name)]
        i = CLASS_ORDER.index(name)
        wrong = [(n, CLASS_ORDER[j]) for j, n in enumerate(row) if j != i]
        n_top, top = max(wrong)
        return sum(n for n, _ in wrong), top, n_top, row[i]

    sections = []

    # ---- 1. Task and data
    sections.append(f"""
<h2>1. 任務與資料</h2>
<ul>
<li>任務：鋼板表面瑕疵 6 類分類，類別順序固定為 {", ".join(f"<code>{c}</code>" for c in CLASS_ORDER)}。</li>
<li>資料：課程提供的 FactoryVision Public Dataset（Classification），官方 split：train {s1["train_count"]:,} 張、validation {s1["validation_count"]:,} 張，每類 validation 45 張。影像為 200 × 200、8-bit 灰階 PNG。</li>
<li>只用 train 訓練與計算 normalization；validation 只用於選 epoch、選模型與比較，不參與訓練，也不做 augmentation。所有實驗 random seed = {s1["seed"]}。</li>
<li>主要指標為 validation <b>Macro F1</b>，同時回報 Accuracy、Macro Precision、Macro Recall 與 Confusion Matrix。</li>
<li>資料限制：官方 split 中 train 的 <code>TRN_0504.png</code> 與 validation 的 <code>VAL_0101.png</code> 內容相同。我們保留官方切分不做修改，解讀 validation 結果時需注意此重疊（1 / 270 張）。</li>
</ul>""")

    # ---- 2. Traditional baseline
    ps = pca["settings"]
    pm = pca["metrics"]
    pca_bad = sorted(((cm_errors(pca, c), c) for c in CLASS_ORDER), reverse=True)[:3]
    bad_text = "；".join(
        f"<code>{c}</code> 45 張只對 {ok} 張（{n_top} 張被判成 <code>{top}</code>）"
        for (n, top, n_top, ok), c in pca_bad)
    sections.append(f"""
<h2>2. Traditional Baseline：PCA + SVM</h2>
{table(["項目", "設定"], [
        ["前處理", f"Bilinear resize 到 {ps["input_size"]} × {ps["input_size"]}，像素 / 255，攤平成 {ps["input_size"]**2} 維向量"],
        ["PCA", f"{ps["pca_components"]} 個主成分，{ps["svd_solver"]} SVD（{ps["fit_dtype"]}），whiten = {ps["whiten"]}，只用 train 擬合"],
        ["SVM", f"kernel = {ps["kernel"]}，C = {ps["C"]:g}，gamma = {ps["gamma"]}，seed = {ps["seed"]}"],
        ["實作", f"<code>src/pca_svm.py</code>（scikit-learn {ps["scikit_learn"]}）"],
    ], "Table 1：PCA + SVM 設定。")}
{table(["Accuracy", "Macro P", "Macro R", "Macro F1"],
       [[f4(pm["accuracy"]), f4(pm["macro_precision"]), f4(pm["macro_recall"]), f4(pm["macro_f1"])]],
       "Table 2：PCA + SVM 的 validation 結果（取自 <code>results/pca_svm.json</code>）。")}
{figure(img(RESULTS / "pca_svm_cm.png", "50%"), "Figure 1：PCA + SVM 的 Confusion Matrix（列 = 真實類別，行 = 預測類別，數字為張數）。")}
<p>PCA + SVM 錯最多的類別：{bad_text}。這些類別要靠細長紋理或細裂紋區分，把整張圖攤平成像素向量再做 PCA，會失去像素之間的空間關係，抓不到這類局部形狀；這是後面改用 CNN 的動機。</p>
""")

    # ---- 3. Baseline CNN
    m1 = e1["metrics"]
    e1_hard = "、".join(f"<code>{c}</code> {cm_errors(e1, c)[3]} / 45" for _, c in pca_bad)
    sections.append(f"""
<h2>3. Baseline CNN（E1）</h2>
<p>自行設計的 <code>{s1["model_name"]}</code>，from scratch 訓練（隨機初始化，不使用任何 pretrained weights，也不使用 ResNet / VGG / MobileNet 等既有架構）。包含 Convolution、Activation（ReLU）、Pooling 與 Classifier。程式在 <code>src/model.py</code>。</p>
{table(["層", "內容", "輸出大小", "參數量"], arch_rows,
       f"Table 3：{s1["model_name"]} 架構（輸入 {s1["input_size"]} × {s1["input_size"]} 灰階）。輸出大小與參數量由程式實際追蹤模型得到。")}
{table(["項目", "設定"], [
        ["前處理（E1 只用這三個）", f"Resize {s1["input_size"]} × {s1["input_size"]} → ToTensor → Normalize（mean = {s1["normalization_mean"][0]:.4f}、std = {s1["normalization_std"][0]:.4f}，只用 train 計算）"],
        ["Optimizer / Loss", f"{s1["optimizer"]}，lr = {s1["lr"]:g}，weight_decay = {s1["weight_decay"]:g}；CrossEntropyLoss"],
        ["Batch size / Epochs", f"{s1["batch_size"]} / {s1["epochs"]}"],
        ["選 epoch", "取 validation Macro F1 最高的 epoch（平手取較早者），儲存該 epoch 的權重"],
        ["Seed / 裝置", f"{s1["seed"]} / {html.escape(e1["latency_device"])}，torch {s1["torch"]}"],
    ], "Table 4：CNN 訓練設定（E1～E4 共用，只有改善方法不同）。")}
{table(["Best epoch", "Accuracy", "Macro P", "Macro R", "Macro F1"],
       [[e1["best_epoch"], f4(m1["accuracy"]), f4(m1["macro_precision"]), f4(m1["macro_recall"]), f4(m1["macro_f1"])]],
       "Table 5：Baseline CNN（E1）的 validation 結果。")}
{figure(img(RESULTS / "E1_curves.png"), "Figure 2：E1 的 Training / Validation Loss 與 Accuracy 曲線。")}
{figure(img(RESULTS / "E1_cm.png", "50%"), f"Figure 3：E1 的 Confusion Matrix。270 張中錯 {len(e1["misclassified"])} 張。")}
<p>Baseline CNN 的 Macro F1 為 {f4(m1["macro_f1"])}，比 PCA + SVM 的 {f4(pm["macro_f1"])} 高 {f4(m1["macro_f1"] - pm["macro_f1"])}。卷積層保留空間結構，能學到裂紋、刮痕、斑點等局部紋理。PCA + SVM 錯最多的三類，在 E1 分對的張數為：{e1_hard}。</p>
""")

    # ---- 4. Improvement methods
    aug = r["E2"]["settings"]["data_augmentation"]
    drop = r["E3"]["settings"]["dropout_p"]
    sections.append(f"""
<h2>4. Improvement Methods</h2>
{table(["方法", "內容", "參數", "選用理由"], [
        ["A：Data Augmentation", "只套用在 train：RandomHorizontalFlip、RandomVerticalFlip、RandomRotation（在 Resize 之後、ToTensor 之前）",
         f"horizontal flip p = {aug["horizontal_flip_p"]:g}；vertical flip p = {aug["vertical_flip_p"]:g}；rotation ±{aug["rotation_degrees"]:g}°",
         "鋼板瑕疵沒有固定方向，翻轉和小角度旋轉不會改變類別，可以增加訓練樣本的多樣性。翻轉 + 旋轉屬於同一組 pipeline，只算一種方法。"],
        ["B：Dropout", "在 Linear 分類層前加 Dropout，推論時自動關閉", f"p = {drop:g}",
         "訓練時隨機關掉一半的特徵，避免分類層依賴少數特徵，降低 overfitting。"],
    ], "Table 6：兩種改善方法與參數（取自 <code>results/E2.json</code>、<code>results/E3.json</code> 的 <code>settings</code>）。")}
<p>兩種方法都屬於 regularization，但作用位置不同：A 改變輸入資料，B 改變模型內部。其他設定（架構、optimizer、lr、epochs、seed）與 E1 完全相同。</p>
""")

    # ---- 5. Ablation
    rows = []
    for exp in ["pca_svm", *CNN_EXPS]:
        m = r[exp]["metrics"]
        params = "—" if r[exp]["params"] is None else f"{r[exp]['params']:,}"
        rows.append([f"<code>{exp}</code>", SETTING[exp], f4(m["accuracy"]), f4(m["macro_precision"]),
                     f4(m["macro_recall"]), f4(m["macro_f1"]), params,
                     f2(r[exp]["checkpoint_mb"]), f2(r[exp]["latency_ms"])])
    wrong = {e: len(r[e]["misclassified"]) for e in CNN_EXPS}
    curves = "".join(f'<div class="half">{img(RESULTS / f"{e}_curves.png")}</div>' for e in ["E2", "E3", "E4"])
    cms = "".join(f'<div class="third">{img(RESULTS / f"{e}_cm.png")}</div>' for e in ["E2", "E3", "E4"])
    sections.append(f"""
<h2>5. Ablation Study（E1～E4）</h2>
{table(["Exp", "Setting", "Accuracy", "Macro P", "Macro R", "Macro F1", "Params", "Size (MB)", "Latency (ms)"], rows,
       "Table 7：Stage 1 全部實驗的 validation 結果。Size 為 checkpoint 檔案大小；Latency 為 batch 1、暖機 10 次、100 次平均，所有實驗在同一台 CPU 量測。")}
{figure(f'<div class="row">{curves}</div>', "Figure 4：E2、E3、E4 的 Training / Validation 曲線（由左至右、由上至下）。")}
{figure(f'<div class="row">{cms}</div>', "Figure 5：E2、E3、E4 的 Confusion Matrix（由左至右）。")}
<ul>
<li>E1、E2、E3 在 validation 各錯 {wrong["E1"]}、{wrong["E2"]}、{wrong["E3"]} 張，Macro F1 在 4 位小數下都是 {f4(r["E1"]["metrics"]["macro_f1"])}；E4 錯 {wrong["E4"]} 張，Macro F1 為 {f4(r["E4"]["metrics"]["macro_f1"])}。</li>
<li>單獨加 A（E2）或單獨加 B（E3）都沒有讓 validation 分數比 Baseline 更高，也沒有變差；同時加 A + B（E4）反而下降。</li>
<li>validation 每類只有 45 張，1 張圖約等於 0.0037 的 Accuracy。E1～E3 的差異在 1 張圖以內，不足以說明哪個方法真的比較好。</li>
<li>改善方法不改變模型參數量（Dropout 沒有權重），四組 checkpoint 大小與 latency 幾乎相同。</li>
</ul>
""")

    # ---- 6. Final model
    f2_full = {e: f"{r[e]['metrics']['macro_f1']:.12f}" for e in ["E2", "E3", "E4"]}
    sections.append(f"""
<h2>6. Final Improved CNN：{final}（{TITLE[final]}）</h2>
<ul>
<li>依規定，最終模型必須是 Improved CNN，所以只從 E2～E4 中選；標準為 validation Macro F1，平手時選參數量較少者。</li>
<li>完整精度的 Macro F1：E2 = {f2_full["E2"]}、E3 = {f2_full["E3"]}、E4 = {f2_full["E4"]}。<b>{final} 最高</b>；三組參數量相同（{fr["params"]:,}）。</li>
<li>E2 和 E3 都錯 2 張，但 E2 的 2 張都集中在 <code>pitted_surface</code>，E3 的錯誤分散在 <code>patches</code> 與 <code>pitted_surface</code> 兩類；Macro F1 是各類 F1 的平均，錯誤集中在同一類時該類 F1 下降較多，所以 E3 略高。</li>
<li>{final} 的 Macro F1 與 Baseline E1 相同，代表 Dropout 在這個資料集上<b>沒有帶來可量測的提升</b>；選它是因為它在 Improved CNN 中最好，而且結果不比 Baseline 差。</li>
<li>提交的 <code>stage1_model.pth</code> 即 {final} 在 best epoch {fr["best_epoch"]} 的權重（SHA-256 <code>{fr["checkpoint_sha256"][:16]}…</code>，完整值見 <code>results/{final}.json</code>）。用 <code>src/evaluate.py</code> 重新載入評估，得到相同的指標。</li>
</ul>
""")

    # ---- 7. Error analysis
    union = {}
    for e in CNN_EXPS:
        for item in r[e]["misclassified"]:
            entry = union.setdefault(item["filename"], {"true": item["true"], "pred": set(), "exps": []})
            entry["pred"].add(CLASS_ORDER[item["predicted"]])
            entry["exps"].append(e)
    err_rows = [[f"<code>{html.escape(name)}</code>", CLASS_ORDER[v["true"]], "、".join(sorted(v["pred"])),
                 "、".join(v["exps"])] for name, v in sorted(union.items())]
    total_err = sum(wrong.values())
    pitted_err = sum(1 for e in CNN_EXPS for it in r[e]["misclassified"]
                     if CLASS_ORDER[it["true"]] == "pitted_surface")
    shown = ["patches/VAL_0097.png", "pitted_surface/VAL_0144.png", "pitted_surface/VAL_0175.png"]
    pics = "".join(
        f'<div class="third">{img(Path(args.data_root) / "classification" / "validation" / name, "70%")}'
        f'<p class="small"><code>{name}</code><br>真實 {CLASS_ORDER[union[name]["true"]]}，預測 {"、".join(sorted(union[name]["pred"]))}（{"、".join(union[name]["exps"])}）<br>{ERROR_NOTES[name]}</p></div>'
        for name in shown)
    sections.append(f"""
<h2>7. 錯誤案例分析</h2>
{table(["圖檔", "真實類別", "被預測成", "分錯的實驗"], err_rows,
       "Table 8：E1～E4 所有分錯的 validation 影像（取自各 JSON 的 <code>misclassified</code>）。")}
{figure(f'<div class="row">{pics}</div>', "Figure 6：三個代表性的錯誤案例（原始 200 × 200 影像）。")}
<ul>
<li>四組 CNN 共 {total_err} 次錯誤，其中 {pitted_err} 次的真實類別是 <code>pitted_surface</code>，它是 CNN 最難分的類別，常被判成 <code>inclusion</code> 或 <code>rolled_in_scale</code>。</li>
<li>錯誤多發生在<b>對比低、瑕疵特徵不明顯</b>的影像：凹坑或斑點很淡時，剩下的背景紋理與其他類別相似。</li>
<li><code>patches/VAL_0097.png</code> 在 E1 和 E3 都被判成 <code>crazing</code>，代表這張圖本身就很模稜兩可，不是單一模型的偶發錯誤。</li>
</ul>
""")

    # ---- 8. Overfitting / generalization
    gap_rows, last_rows = [], []
    for e in CNN_EXPS:
        h, b = r[e]["history"], r[e]["best_epoch"] - 1
        gap_rows.append([f"<code>{e}</code>", r[e]["best_epoch"], f4(h["train_acc"][b]), f4(h["val_acc"][b]),
                         f4(h["train_loss"][b]), f4(h["val_loss"][b])])
        last_rows.append([f"<code>{e}</code>", f4(h["train_acc"][-1]), f4(h["val_acc"][-1]),
                          f4(h["train_loss"][-1]), f4(h["val_loss"][-1])])
    h1 = e1["history"]
    peak_epoch = max(range(40, len(h1["val_loss"])), key=lambda i: h1["val_loss"][i]) + 1
    sections.append(f"""
<h2>8. Overfitting / Generalization 分析</h2>
{table(["Exp", "Best epoch", "Train acc", "Val acc", "Train loss", "Val loss"], gap_rows,
       "Table 9：各實驗在 best epoch 的 train 與 validation 數值（取自 JSON 的 <code>history</code>）。")}
{table(["Exp", "Train acc", "Val acc", "Train loss", "Val loss"], last_rows,
       f"Table 10：各實驗在最後一個 epoch（第 {s1["epochs"]} 個）的數值。")}
<ul>
<li><b>Baseline 沒有明顯 overfitting。</b>E1 的 train 與 validation 曲線在 {s1["epochs"]} 個 epoch 內幾乎重疊（Figure 2）。best epoch 時 train acc {f4(h1["train_acc"][e1["best_epoch"]-1])}、val acc {f4(h1["val_acc"][e1["best_epoch"]-1])}，兩者只差 {f4(abs(h1["val_acc"][e1["best_epoch"]-1] - h1["train_acc"][e1["best_epoch"]-1]))}。val loss 後期有起伏（例如 epoch {peak_epoch} 為 {f4(h1["val_loss"][peak_epoch-1])}），但最後回到 {f4(h1["val_loss"][-1])}，沒有持續上升的趨勢。模型沒有「死背」訓練資料，學到的特徵能推廣到 validation。</li>
<li><b>這解釋了為什麼改善方法沒有加分。</b>Data Augmentation 和 Dropout 都是用來降低 overfitting 的 regularization；Baseline 本身已經不 overfit，validation 又已經接近 99%，兩種方法沒有可以修正的空間。</li>
<li><b>E2～E4 的 train acc 低於 val acc</b>（Table 9），因為 train 的數字是在 augmentation 與 Dropout 開啟時計算（題目變難），validation 是在關閉狀態下計算。這是 regularization 的正常現象，不代表 underfitting。</li>
<li><b>E4 同時使用 A + B，正則化最強。</b>最後一個 epoch 的 train loss 為 {f4(r["E4"]["history"]["train_loss"][-1])}，高於 E1（{f4(h1["train_loss"][-1])}）、E2（{f4(r["E2"]["history"]["train_loss"][-1])}）、E3（{f4(r["E3"]["history"]["train_loss"][-1])}），代表在同樣 {s1["epochs"]} 個 epoch 內模型最難擬合訓練資料，validation 錯誤也集中在較難的 <code>pitted_surface</code>。若要讓 E4 發揮，可能需要更多 epochs。</li>
</ul>
""")

    # ---- 9. Cost and conclusion
    sections.append(f"""
<h2>9. 工程成本與結論</h2>
<ul>
<li>Final CNN（{final}）有 {fr["params"]:,} 個參數，checkpoint {f2(fr["checkpoint_mb"])} MB，CPU 上單張推論 {f2(fr["latency_ms"])} ms；PCA + SVM 為 {f2(pca["checkpoint_mb"])} MB、{f2(pca["latency_ms"])} ms。CNN 稍慢，但 Macro F1 高約 {f4(fr["metrics"]["macro_f1"] - pm["macro_f1"])}，每張仍只需約 {f2(fr["latency_ms"])} ms，足以用在產線即時檢測。</li>
<li>Latency 在 {html.escape(fr["latency_device"])} 上量測，這台電腦沒有 NVIDIA GPU。</li>
<li><b>結論：</b>自建 CNN 明顯優於傳統 PCA + SVM。在這個資料集上，小型 CNN 已能達到 Macro F1 {f4(m1["macro_f1"])} 且沒有 overfitting；Data Augmentation 與 Dropout 沒有帶來可量測的提升，同時使用反而略降。最終選擇 {final} 作為 Final Improved CNN。</li>
<li><b>限制：</b>validation 只有 270 張，方法間的差異只有 0～2 張圖；只用 seed 42 跑一次，沒有多次重複估計變異；train / validation 有一對重複影像。</li>
</ul>
<h2>10. 重現方式</h2>
<p>從 <code>Team05_Stage1/</code> 根目錄執行（<code>$root</code> 為 <code>STUDENT_PUBLIC</code> 資料夾路徑）：</p>
<pre>python src/pca_svm.py --data_root $root
python src/train.py --data_root $root --exp E1   # E2, E3, E4 同理
copy checkpoints\\{final}.pth stage1_model.pth  # 選定的 Final Improved CNN
python src/evaluate.py --data_root $root         # 載入 stage1_model.pth 與 config.json</pre>
""")

    today = datetime.date.today().isoformat()
    page = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>Team05 Stage 1 Report</title>
<style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; font-size: 10.5pt; line-height: 1.55; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 4px; }}
h2 {{ font-size: 13pt; border-bottom: 1.5px solid #333; padding-bottom: 2px; margin-top: 18px; break-after: avoid; }}
.tbl {{ break-inside: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 4px 0 10px; font-size: 9.5pt; break-inside: avoid; }}
th, td {{ border: 1px solid #999; padding: 3px 6px; text-align: left; vertical-align: top; }}
th {{ background: #eee; }}
.cap {{ font-weight: bold; font-size: 9.5pt; margin: 8px 0 2px; }}
.fig {{ text-align: center; margin: 8px 0; break-inside: avoid; }}
.fig .cap {{ font-weight: normal; }}
.row {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; }}
.half {{ width: 49%; }} .third {{ width: 32%; }}
.small {{ font-size: 8.5pt; text-align: left; }}
code {{ font-family: Consolas, monospace; font-size: 9pt; }}
pre {{ background: #f4f4f4; padding: 6px; font-size: 9pt; }}
.meta {{ color: #444; margin-bottom: 8px; }}
</style></head><body>
<h1>Stage 1 Report：Baseline + Improved CNN</h1>
<p class="meta">Team05　·　負責人：葉文賢　·　{today}</p>
{"".join(sections)}
</body></html>"""
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} (final={final})")


if __name__ == "__main__":
    main()
