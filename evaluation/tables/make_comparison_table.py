"""Build the paper-style LaTeX tables: baseline vs quantization vs pruning of the fine-tuned ResNet-18.

Reads only result files already in the repo and writes, next to this script:
  comparison_table.tex  -- five tables, \\input-able:
      1 training / fine-tuning set-up        (tab:setup)
      2 what every model variant is          (tab:variants)
      3 main comparison after fine-tuning    (tab:compression-comparison)
      4 pruning before vs after fine-tuning  (tab:before-after)
      5 seed robustness                      (tab:seed-robustness)
  preview.tex           -- minimal document that \\inputs the tables (compile it to get the PDF)

Models run with several seeds (control, filter 10 %, unstructured 70 %: seeds 42/1/2; static PTQ: three
calibration sets) are shown as the mean over the three runs in tables 3 and 4.

Usage:  python3 evaluation/tables/make_comparison_table.py
        cd evaluation/tables && pdflatex -jobname=comparison_table preview.tex
Packages: booktabs, makecell, threeparttable, rotating, tabularx, amssymb, xcolor (table option).
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = {
    "Q": ROOT / "evaluation/results_quantization",
    "F": ROOT / "pruning/structured/filter/results/evaluation",
    "B": ROOT / "pruning/structured/whole_block/results/evaluation",
    "U": ROOT / "pruning/unstructured/results/evaluation",
    "QS": ROOT / "quantization/qat/results/evaluation_seeds",  # QAT seeds 42, 1, 2 (+ FP32), one session
}
PTQ_SEEDS = ROOT / "quantization/ptq/static/results/validation_results.csv"
STAGE1 = {"F": ROOT / "pruning/structured/filter/results/comparaison/stage1_cpu_validation.csv",
          "U": ROOT / "pruning/unstructured/results/report/stage1_cpu_validation.csv"}
BLOCK_RUNS = ROOT / "pruning/structured/whole_block/results"
SEEDS = (42, 1, 2)
QAT_RUNS = ["qat", "qat_seed1", "qat_seed2"]  # training seeds 42, 1, 2
BLOCK_ORDER = ["layer2.1", "layer1.0", "layer3.1", "layer4.1", "layer1.1"]


def read_csv(path):
    return list(csv.DictReader(open(path)))


def load(session, kind):
    return {r["model"]: r for r in read_csv(SRC[session] / "comparison" / kind / "table_validation.csv")}


DATA = {s: {k: load(s, k) for k in ("performance", "efficiency", "environment")} for s in SRC}


def get(session, model):
    row = {}
    for kind in ("performance", "efficiency", "environment"):
        row.update(DATA[session][kind][model])
    return row


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean_row(rows):
    """Field-wise mean of numeric fields (first row's value for text fields)."""
    out = dict(rows[0])
    for key in rows[0]:
        values = [num(r.get(key)) for r in rows]
        if all(v is not None for v in values):
            out[key] = statistics.mean(values)
    return out


def ptq_seed_rows():
    """Per-calibration-set performance of static PTQ (balanced accuracy = macro recall)."""
    rows = {int(float(r["seed"])): dict(r, balanced_accuracy=r["recall_macro"])
            for r in read_csv(PTQ_SEEDS) if r["model"].startswith("INT8_seed")}
    return [rows[s] for s in SEEDS]


PERF_KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "specificity_macro", "f1_macro", "auc_macro",
             "melanoma_recall"]


class Model:
    """One table row: a single run, or the mean of several seeds (self.runs holds every run)."""

    def __init__(self, session, base=None, multi=False, ptq=False, names=None):
        if ptq:  # efficiency measured on the saved (seed-42) model only; performance averaged over 3 calibrations
            seeds = ptq_seed_rows()
            self.runs = [dict(get(session, base), **{k: s[k] for k in PERF_KEYS}) for s in seeds]
            for run in self.runs[1:]:  # no paired-bootstrap p for calibration sets 1 and 2
                for k in PERF_KEYS:
                    run[f"{k}_paired_bootstrap_p"] = ""
        elif names:
            self.runs = [get(session, n) for n in names]
        elif multi:
            self.runs = [get(session, f"{base}_seed{s}") for s in SEEDS]
        else:
            self.runs = [get(session, base)]
        self.row = mean_row(self.runs)
        self.averaged = len(self.runs) > 1

    def significant(self, key):
        """p < 0.05 in every run, all changes in the same direction."""
        ps = [num(r.get(f"{key}_paired_bootstrap_p")) for r in self.runs]
        changes = [num(r.get(f"{key}_change_vs_fp32")) for r in self.runs]
        if any(p is None for p in ps) or any(c is None for c in changes):
            return False
        same_sign = all(c > 0 for c in changes) or all(c < 0 for c in changes)
        return all(p < 0.05 for p in ps) and same_sign


# ================================================================ formatting helpers
def pct(x):
    return f"{100 * float(x):.1f}"


def fmt(n):
    return lambda x: f"{float(x):.{n}f}"


def signed(x, digits=0):
    v = round(float(x), digits)
    if v == 0:
        return f"{0:.{digits}f}"
    return ("$+$" if v > 0 else "$-$") + f"{abs(v):.{digits}f}"


AVG = "$^{\\dagger}$"   # marks rows averaged over 3 runs
COLLAPSE = "$^{\\ddagger}$"


# ================================================================ Table 1: set-up
def training_minutes():
    def total(path, col):
        return sum(float(r[col]) for r in read_csv(path) if r.get("epoch") != "0" and r[col])

    base = json.load(open(ROOT / "resnet18_finetunning/results/strong_cosine25_resnet18/final_metrics.json"))
    ft = [total(p, "seconds") for p in sorted((ROOT / "pruning").glob("**/results/*/training_history.csv"))]
    qat = total(ROOT / "quantization/qat/results/training_history.csv", "epoch_seconds")
    return base["total_training_time_sec"] / 60, statistics.mean(ft) / 60, qat / 60, base["best_epoch"]


def table_setup():
    base_min, ft_min, qat_min, best_epoch = training_minutes()
    span = lambda text: (  # noqa: E731  -- X columns cannot be spanned, so give the span an explicit width
        "\\multicolumn{3}{>{\\raggedright\\arraybackslash}p{\\dimexpr\\linewidth-2.3cm-2\\tabcolsep\\relax}@{}}"
        f"{{{text}}}")
    rows = [
        ("\\textit{Data}", None),
        ("Dataset", span("HAM10000 dermoscopy images, 7 classes, 10{,}015 images; very imbalanced "
                         "(\\textit{nv} = 67\\%, \\textit{df} and \\textit{vasc} $\\approx$1\\% each).")),
        ("Split", span("By lesion (no lesion in two partitions), class-stratified, seed 42: "
                       "6{,}981 train / 1{,}532 validation / 1{,}502 test. All results are on the validation "
                       "split; the test split has not been used.")),
        ("Input", span("224$\\times$224 RGB, ImageNet mean/std normalisation (evaluation: resize only).")),
        ("\\textit{Training}", None),
        ("Starting point", "ImageNet weights (\\texttt{IMAGENET1K\\_V1}), new 7-class \\texttt{fc}",
         "FP32 baseline, after pruning (control: not pruned)",
         "FP32 baseline; quantizer ranges copied from static PTQ"),
        ("Epochs", "25", "10", "10"),
        ("Optimizer", span("Adam")),
        ("Learning rate", "$10^{-4}$, cosine decay to $\\approx$0", "$10^{-4}$, cosine decay to $\\approx$0",
         "$10^{-5}$ (ep.\\ 1--2), then cosine decay to $\\approx$0"),
        ("Batch size", span("32")),
        ("Loss", span("Cross-entropy with balanced class weights computed on the training split")),
        ("Augmentation", span("Rotation $\\pm$180$^\\circ$; random resized crop (scale 0.6--1.0, ratio 0.75--1.33); "
                              "horizontal and vertical flips ($p=0.5$); colour jitter (brightness and contrast 0.2, "
                              "saturation 0.15, hue 0.02)")),
        ("BatchNorm", "Trained", "Trained (block pruning: re-estimated on 2{,}000 training images first)",
         "Trained until ep.\\ 7, then frozen; quantizer ranges frozen from ep.\\ 8"),
        ("Checkpoint kept", f"Lowest validation loss (epoch {best_epoch})", "Highest validation macro F1",
         "Last epoch (no selection)"),
        ("Seeds", "42", "42 (+1, 2 for control, filter 10\\%, unstructured 70\\%)", "42, 1, 2"),
        ("Hardware", span("NVIDIA A40 (24\\,GB), PyTorch 2.12")),
        ("GPU time", f"{base_min:.1f} min", f"$\\approx${ft_min:.1f} min per run", f"{qat_min:.1f} min"),
    ]
    out = ["% ---------------------------------------------------------------- Table: set-up",
           "% Generated by evaluation/tables/make_comparison_table.py -- do not edit by hand.",
           "\\begin{table}[htbp]", "\\centering",
           "\\caption{Training and fine-tuning set-up. \\emph{Fine-tuning} is the recipe applied after pruning; "
           "the \\emph{control} uses exactly the same recipe without pruning (Table~\\ref{tab:variants}). "
           "Static PTQ and weight-only PTQ need no training.}",
           "\\label{tab:setup}", "\\small", "\\renewcommand{\\arraystretch}{1.18}",
           "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{2.3cm}LLL@{}}", "\\toprule",
           " & \\textbf{Baseline (FP32)} & \\textbf{Fine-tuning after pruning, and control} & \\textbf{QAT (INT8)} \\\\",
           "\\midrule"]
    for row in rows:
        if row[1] is None:
            out.append(f"\\multicolumn{{4}}{{@{{}}l}}{{{row[0]}}} \\\\[1pt]")
        elif len(row) == 2:
            out.append(f"{row[0]} & {row[1]} \\\\")
        else:
            out.append(" & ".join(row) + " \\\\")
        if row[0] == "Input":
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    return out


# ================================================================ Table 2: model variants
def table_variants():
    rows = [
        ("FP32 baseline", "ResNet-18 trained on HAM10000 (Table~\\ref{tab:setup}). The reference for every comparison.",
         "11.18\\,M parameters, 3.64 GFLOPs", "--"),
        ("FP32 control", "The baseline fine-tuned for \\textbf{10 more epochs} with exactly the pruning recipe, "
         "\\textbf{without} any compression.",
         "Not a compression method. Extra training alone changes the scores, so a pruned model should also be "
         "compared with the control: pruned $-$ control = effect of pruning only.", "10 ep."),
        ("Weight-only PTQ", "Conv and \\texttt{fc} weights stored as INT8, turned back into FP32 when loaded.",
         "Symmetric, scale $=\\max|w|/127$, one scale per tensor or per output channel. Only the file shrinks: "
         "compute stays FP32.", "No"),
        ("Static PTQ (W8A8)", "Weights \\emph{and} activations in INT8, integer kernels.",
         "x86 backend; weights per-channel symmetric, activations per-tensor affine; ranges calibrated on 1{,}000 "
         "class-stratified training images (3 different sets tried).", "No"),
        ("QAT (W8A8)", "Same INT8 format as static PTQ, but trained with simulated quantization.",
         "Starts from the static-PTQ ranges (Table~\\ref{tab:setup}).", "10 ep."),
        ("Unstructured pruning", "Individual weights set to zero; layer shapes unchanged.",
         "Global L1 magnitude over all conv and \\texttt{fc} weights; 30/50/70/90\\% zeroed. Mask kept during "
         "fine-tuning. Zeros are still stored and computed (dense kernels).", "10 ep."),
        ("Filter pruning", "Whole convolution filters removed: thinner layers.",
         "L1 norm of each filter; same ratio (10/20/30/40\\%) in every channel group (Torch-Pruning dependency "
         "graph); stem \\texttt{conv1} and \\texttt{fc} untouched.", "10 ep."),
        ("Block pruning", "Whole residual blocks replaced by identity: shallower network.",
         "Removed in a greedy order: " + ", ".join(f"\\texttt{{{b}}}" for b in BLOCK_ORDER) + " (1 to 5 blocks); "
         "BatchNorm re-estimated on 2{,}000 training images before fine-tuning.", "10 ep."),
    ]
    out = ["% ---------------------------------------------------------------- Table: variants",
           "\\begin{table}[htbp]", "\\centering",
           "\\caption{What each model in the result tables is. \\emph{Retrained}: epochs of training after "
           "compression (recipe in Table~\\ref{tab:setup}).}",
           "\\label{tab:variants}", "\\small", "\\renewcommand{\\arraystretch}{1.18}",
           "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{2.5cm}"
           ">{\\raggedright\\arraybackslash}p{4.1cm}L>{\\centering\\arraybackslash}p{1.45cm}@{}}",
           "\\toprule",
           "\\textbf{Model} & \\textbf{What changes} & \\textbf{Settings} & \\makecell{\\textbf{Re-}\\\\\\textbf{trained}} \\\\", "\\midrule"]
    groups = {"Weight-only PTQ": "Quantization", "Unstructured pruning": "Pruning"}
    for name, what, settings, retrained in rows:
        if name in groups:
            out.append(f"\\addlinespace[2pt]\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{groups[name]}}}}} \\\\")
        out.append(f"{name} & {what} & {settings} & {retrained} \\\\")
    out += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    return out


# ================================================================ Table 3: main comparison
PERF = [("accuracy", "Acc.", "max"), ("balanced_accuracy", "Bal.\\\\Acc.", "max"),
        ("precision_macro", "Prec.", "max"), ("specificity_macro", "Spec.", "max"),
        ("f1_macro", "F1", "max"), ("auc_macro", "AUC", "max"), ("melanoma_recall", "Mel.\\\\Rec.", "max")]
EFF = [("size_mb", "Size\\\\(MB)", "min", fmt(2)),
       ("size_ratio_fp32_over_model", "Compr.\\\\($\\times$)", "max", fmt(2)),
       ("params", "Params\\\\(M)", "min", lambda x: f"{float(x) / 1e6:.2f}"),
       ("gflops", "GFLOPs", "min", fmt(2)),
       ("latency_median_ms_1t", "Lat. 1T\\\\(ms)", "min", fmt(1)),
       ("speedup_vs_fp32_1t", "Speed-\\\\up 1T\\\\($\\times$)", "max", fmt(2)),
       ("latency_median_ms_4t", "Lat. 4T\\\\(ms)", "min", fmt(1)),
       ("throughput_img_s_4t", "Thr. 4T\\\\(img/s)", "max", fmt(0)),
       ("peak_memory_mb_1t", "Peak\\\\RAM\\\\(MB)", "min", fmt(1))]
ENV = [("energy_wh_per_1000", "Energy\\\\(Wh/1k)", "min", fmt(2)),
       ("energy_wh_per_1000_change_vs_fp32_pct", "$\\Delta$\\\\Energy\\\\(\\%)", "min", signed),
       ("co2e_g_per_1000", "CO$_2$e\\\\(g/1k)", "min", fmt(2))]
ARROW = {"max": "$\\uparrow$", "min": "$\\downarrow$"}
COLUMNS = ([(k, h, d, pct, True) for k, h, d in PERF]
           + [(k, h, d, f, False) for k, h, d, f in EFF]
           + [(k, h, d, f, False) for k, h, d, f in ENV])
NCOL = 1 + len(COLUMNS)

# (group title or None, [(label, Model, kind)]) ; kind: base | ref | method
GROUPS = [
    (None, [("FP32 baseline (ResNet-18)", Model("Q", "fp32"), "base"),
            ("FP32 control (baseline + 10 ep.)", Model("F", "control", multi=True), "ref")]),
    ("Quantization (INT8)", [
        ("Weight-only PTQ, per-tensor$^{a}$", Model("Q", "weight_only_per_tensor"), "method"),
        ("Weight-only PTQ, per-channel$^{a}$", Model("Q", "weight_only_per_channel"), "method"),
        ("Static PTQ (W8A8)", Model("Q", "ptq_static", ptq=True), "method"),
        ("QAT (W8A8)", Model("QS", names=QAT_RUNS), "method")]),
    ("Unstructured pruning (weights zeroed)$^{b}$, after fine-tuning", [
        (f"{s}\\% of weights zeroed", Model("U", f"sparse{s}", multi=True) if s == 70 else Model("U", f"sparse{s}_seed42"),
         "method") for s in (30, 50, 70, 90)]),
    ("Filter pruning (filters removed), after fine-tuning", [
        (f"{r}\\% of filters removed", Model("F", f"pruned{r}", multi=True) if r == 10 else Model("F", f"pruned{r}_seed42"),
         "method") for r in (10, 20, 30, 40)]),
    ("Block pruning (residual blocks removed), after fine-tuning", [
        (f"{k} block{'s' if k > 1 else ''} ({'+' if k > 1 else ''}\\texttt{{{BLOCK_ORDER[k - 1]}}})",
         Model("B", f"blocks{k}_seed42"), "method") for k in range(1, 6)]),
]


def table_main():
    methods = [m for _, items in GROUPS for _, m, kind in items if kind == "method"]
    best = {}
    for key, _, direction, f, _ in COLUMNS:
        values = [f(m.row[key]) for m in methods]
        nums = [float(v.replace("$", "")) for v in values]  # "$-$12" -> "-12"
        target = max(nums) if direction == "max" else min(nums)
        best[key] = {v for v, n in zip(values, nums) if n == target}

    def cell(model, kind, key, f, is_perf):
        text = f(model.row[key])
        if kind == "method" and text in best[key]:
            text = f"\\textbf{{{text}}}"
        if is_perf and kind != "base" and model.significant(key):
            text += "$^{\\ast}$"
        return text

    out = []
    w = out.append
    w("% ---------------------------------------------------------------- Table: main comparison")
    w("\\begin{sidewaystable}")
    w("\\centering")
    w("\\caption{Baseline vs.\\ quantization vs.\\ pruning (pruned models after fine-tuning). ResNet-18 on "
      "HAM10000, validation split ($n=1{,}532$); every model is evaluated the same way: CPU (Intel Xeon Gold 6348), "
      "BatchNorm fused, TorchScript. Each model is described in Table~\\ref{tab:variants}; the \\emph{control} is "
      "the baseline fine-tuned for 10 more epochs without compression. "
      "$\\uparrow$/$\\downarrow$: higher/lower is better. \\textbf{Bold}: best compressed model in the column. "
      f"$^{{\\ast}}$: significant change vs.\\ the baseline, up or down (paired bootstrap, $p<0.05$). "
      f"{AVG}: mean of 3 runs (seeds 42, 1, 2; static PTQ: 3 calibration sets); other rows: seed 42.}}")
    w("\\label{tab:compression-comparison}")
    w("\\scriptsize")
    w("\\setlength{\\tabcolsep}{2.2pt}")
    w("\\renewcommand{\\arraystretch}{1.12}")
    w("\\begin{threeparttable}")
    w("\\begin{tabular}{l " + "c" * len(PERF) + " " + "c" * len(EFF) + " " + "c" * len(ENV) + "}")
    w("\\toprule")
    a, b = 2, 1 + len(PERF)
    c, d = b + 1, b + len(EFF)
    e, g = d + 1, d + len(ENV)
    w(f" & \\multicolumn{{{len(PERF)}}}{{c}}{{\\textbf{{Performance}} (\\%, macro-averaged)}}"
      f" & \\multicolumn{{{len(EFF)}}}{{c}}{{\\textbf{{Efficiency}} (CPU inference)}}"
      f" & \\multicolumn{{{len(ENV)}}}{{c}}{{\\textbf{{Environment}} (estimated)}} \\\\")
    w(f"\\cmidrule(lr){{{a}-{b}}} \\cmidrule(lr){{{c}-{d}}} \\cmidrule(lr){{{e}-{g}}}")
    heads = ["\\makecell[l]{\\textbf{Model}}"]
    for key, h, direction, _, _ in COLUMNS:
        heads.append(f"\\makecell{{{h}\\,{ARROW[direction]}}}")
    w(" & ".join(heads) + " \\\\")
    w("\\midrule")
    for title, items in GROUPS:
        if title:
            w("\\addlinespace[2pt]")
            w(f"\\multicolumn{{{NCOL}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for lab, model, kind in items:
            label = ("" if kind == "base" else "\\quad ") + lab + (AVG if model.averaged else "")
            cells = [label] + [cell(model, kind, key, f, is_perf) for key, _, _, f, is_perf in COLUMNS]
            w(("\\rowcolor{gray!12} " if kind == "base" else "") + " & ".join(cells) + " \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")

    nz = {s: float(get("U", f"sparse{s}_seed42")["params_nonzero"]) / 1e6 for s in (30, 50, 70, 90)}
    gz = {s: float(get("U", f"sparse{s}_seed42")["state_dict_gzip_mb"]) for s in (30, 50, 70, 90)}
    gz_fp32 = float(get("U", "fp32")["state_dict_gzip_mb"])
    lat = sorted(float(DATA[s]["efficiency"]["fp32"]["latency_median_ms_1t"]) for s in SRC)
    w("\\begin{tablenotes}[flushleft]")
    w("\\scriptsize")
    w("\\item \\textit{How to read.} Bal.\\ Acc.\\ = macro recall; Mel.\\ Rec.\\ = melanoma recall. 95\\% CIs are "
      "about $\\pm$4 points on macro F1, so 1--2 point differences without $^{\\ast}$ are not meaningful. "
      f"For averaged rows ({AVG}), $^{{\\ast}}$ means significant in all 3 runs in the same direction; per-run "
      "spread is in Table~\\ref{tab:seed-robustness}. Static PTQ efficiency is measured on the seed-42 model "
      "(the 3 calibrations give the same architecture).")
    w("\\item \\textit{Speed.} Lat.: batch 1, median of 200 runs (30 warm-up), median of 3 interleaved repeats, at "
      "1 thread (1T) and 4 threads (4T). Thr.: batch 32. Peak RAM: memory increase while loading and warming up. "
      "GFLOPs are counted on the FP32 graph for every model (INT8 runs the same operations in INT8). "
      f"Models were timed in {len(SRC)} sessions and FP32 latency varied from {lat[0]:.1f} to {lat[-1]:.1f}\\,ms between them, "
      "so speed-up and $\\Delta$Energy are computed against the FP32 model timed in the same session: "
      "compare those rather than absolute latencies.")
    w("\\item \\textit{Environment.} Per 1{,}000 images at 1 thread. Energy is a CodeCarbon estimate (no hardware "
      "power counter; about 250\\,W per active thread assumed), so it follows run time; differences below about "
      "5\\% are noise. CO$_2$e: Tunisian grid, 564\\,g\\,CO$_2$e/kWh.")
    w("\\item[a] Weights are INT8 on disk only; they are converted back to FP32 when loaded, so speed and "
      "energy do not change (by design).")
    w("\\item[b] Zeros are still stored and computed, hence no gain in size or speed. Non-zero parameters: "
      + "/".join(f"{nz[s]:.2f}" for s in (30, 50, 70, 90)) + "\\,M; gzip-compressed file: "
      + "/".join(f"{gz[s]:.1f}" for s in (30, 50, 70, 90)) + f"\\,MB (FP32: {gz_fp32:.1f}\\,MB).")
    w("\\item[] Pruned models and the control keep the fine-tuning epoch with the best validation macro F1, so their "
      "scores are slightly optimistic. The test split has not been used.")
    w("\\end{tablenotes}")
    w("\\end{threeparttable}")
    w("\\end{sidewaystable}")
    w("")
    return out


# ================================================================ Table 4: before vs after fine-tuning
def before_rows():
    stage1 = {s: {r["run"]: r for r in read_csv(p)} for s, p in STAGE1.items()}
    blocks = {}
    for k in range(1, 6):
        for r in read_csv(BLOCK_RUNS / f"blocks{k}_seed42" / "metrics.csv"):
            blocks[(k, r["stage"])] = r
    return stage1, blocks


def table_before_after():
    stage1, blocks = before_rows()
    base = get("Q", "fp32")
    f1_base = float(base["f1_macro"])
    ref = "Fine-tuned (10 ep.)"
    # (model label, [(stage label, row, averaged)])
    entries = [("FP32 baseline", [("--", base, False)])]
    control = Model("F", "control", multi=True)
    entries.append(("FP32 control", [(ref, control.row, True)]))
    groups = [("Unstructured pruning", [
        (f"{s}\\% of weights zeroed",
         [("Pruned, not fine-tuned", stage1["U"][f"sparse{s}_seed42"], False),
          (ref, *(lambda m: (m.row, m.averaged))(Model("U", f"sparse{s}", multi=True) if s == 70
                                                 else Model("U", f"sparse{s}_seed42")))])
        for s in (30, 50, 70, 90)])]
    groups.append(("Filter pruning", [
        (f"{r}\\% of filters removed",
         [("Pruned, not fine-tuned", stage1["F"][f"pruned{r}_seed42"], False),
          (ref, *(lambda m: (m.row, m.averaged))(Model("F", f"pruned{r}", multi=True) if r == 10
                                                 else Model("F", f"pruned{r}_seed42")))])
        for r in (10, 20, 30, 40)]))
    groups.append(("Block pruning", [
        (f"{k} block{'s' if k > 1 else ''}",
         [("Pruned, not fine-tuned", blocks[(k, "raw")], False),
          ("+ BatchNorm re-estimated", blocks[(k, "bn_recalibrated")], False),
          (ref, get("B", f"blocks{k}_seed42"), False)])
        for k in range(1, 6)]))

    out = []
    w = out.append
    w("% ---------------------------------------------------------------- Table: before vs after fine-tuning")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{Pruning before and after fine-tuning (validation split, \\%). Efficiency does not change with "
      "fine-tuning (same architecture), see Table~\\ref{tab:compression-comparison}. "
      "Last column: macro F1 minus the FP32 baseline, in points. "
      f"{AVG}: mean of 3 fine-tuning seeds (42, 1, 2); pruning itself is deterministic, so the \\emph{{not "
      "fine-tuned} rows are the same for every seed.}")
    w("\\label{tab:before-after}")
    w("\\footnotesize")
    w("\\setlength{\\tabcolsep}{3.2pt}")
    w("\\begin{threeparttable}")
    w("\\begin{tabular}{@{}ll" + "c" * len(PERF_KEYS) + "c@{}}")
    w("\\toprule")
    heads = ["\\textbf{Model}", "\\textbf{Stage}", "Acc.", "\\makecell{Bal.\\\\Acc.}", "Prec.", "Spec.", "F1", "AUC",
             "\\makecell{Mel.\\\\Rec.}", "\\makecell{$\\Delta$F1 vs.\\\\baseline}"]
    w(" & ".join(heads) + " \\\\")
    w("\\midrule")

    def emit(label, stages):
        for i, (stage, row, averaged) in enumerate(stages):
            collapsed = float(row["balanced_accuracy"]) < 0.15
            stage_txt = stage + (AVG if averaged else "") + (COLLAPSE if collapsed else "")
            cells = [label if i == 0 else "", stage_txt] + [pct(row[k]) for k in PERF_KEYS]
            delta = 100 * (float(row["f1_macro"]) - f1_base)
            cells.append("--" if row is base else signed(delta, 1))
            if stage == ref:
                cells = cells[:2] + [f"\\textbf{{{c}}}" for c in cells[2:]]
            prefix = "\\rowcolor{gray!12} " if row is base else ""
            w(prefix + " & ".join(cells) + " \\\\")

    for label, stages in entries:
        emit(label, stages)
    for title, items in groups:
        w("\\addlinespace[3pt]")
        w(f"\\multicolumn{{{2 + len(PERF_KEYS) + 1}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for i, (label, stages) in enumerate(items):
            if i:
                w("\\addlinespace[2pt]")
            emit("\\quad " + label, stages)
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\begin{tablenotes}[flushleft]")
    w("\\footnotesize")
    w("\\item Bold rows: final models (after fine-tuning), as in Table~\\ref{tab:compression-comparison}.")
    w(f"\\item[{COLLAPSE}] Collapsed model: it predicts (almost) a single class for every image, so balanced "
      "accuracy $\\approx$ 1/7 = 14.3\\% and melanoma recall = 0. Accuracy stays near 66\\% only because "
      "\\textit{nv} is 66\\% of the images.")
    w("\\item Block-pruning rows before fine-tuning were evaluated on the training GPU (not fused); all other rows "
      "on CPU with the protocol of Table~\\ref{tab:compression-comparison}.")
    w("\\end{tablenotes}")
    w("\\end{threeparttable}")
    w("\\end{table}")
    w("")
    return out


# ================================================================ Table 5: seed robustness
SEED_PERF = [("accuracy", "Acc."), ("balanced_accuracy", "Bal.\\ Acc."), ("precision_macro", "Prec."),
             ("specificity_macro", "Spec."), ("f1_macro", "F1"), ("auc_macro", "AUC"),
             ("melanoma_recall", "Mel.\\ Rec.")]


def mean_sd(values):
    values = [100 * v for v in values]
    return f"{statistics.mean(values):.1f}\\,{{\\tiny$\\pm$\\,{statistics.stdev(values):.1f}}}"


def table_seeds():
    control = [get("F", f"control_seed{s}") for s in SEEDS]
    entries = [
        ("FP32 baseline (single model)", [get("Q", "fp32")], None),
        ("FP32 control (baseline + 10 ep.)", control, None),
        ("Static PTQ (W8A8), 3 calibration sets", ptq_seed_rows(), None),
        ("QAT (W8A8)", [get("QS", n) for n in QAT_RUNS], None),
        ("Unstructured pruning, 70\\% zeroed", [get("U", f"sparse70_seed{s}") for s in SEEDS], control),
        ("Filter pruning, 10\\% of filters", [get("F", f"pruned10_seed{s}") for s in SEEDS], control),
    ]
    out = []
    w = out.append
    w("% ---------------------------------------------------------------- Table: seed robustness")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{How much the results move with the random seed (seeds 42, 1, 2): mean\\,$\\pm$\\,standard "
      "deviation, in \\%. For static PTQ the seed picks the 1{,}000 calibration images; otherwise it sets "
      "fine-tuning. Last column: each pruned model minus the control trained with the same seed, i.e.\\ the effect "
      "of pruning alone.}")
    w("\\label{tab:seed-robustness}")
    w("\\footnotesize")
    w("\\setlength{\\tabcolsep}{2.8pt}")
    w("\\begin{tabular}{@{}l" + "c" * len(SEED_PERF) + "c@{}}")
    w("\\toprule")
    heads = (["\\textbf{Model}"] + [f"\\makecell{{{h}}}" for _, h in SEED_PERF]
             + ["\\makecell{$\\Delta$F1 vs.\\\\control}"])
    w(" & ".join(heads) + " \\\\")
    w("\\midrule")
    for label, rows, ctrl in entries:
        if len(rows) == 1:
            cells = [f"{100 * float(rows[0][k]):.1f}" for k, _ in SEED_PERF]
        else:
            cells = [mean_sd([float(r[k]) for r in rows]) for k, _ in SEED_PERF]
        if ctrl:
            delta = [100 * (float(r["f1_macro"]) - float(c["f1_macro"])) for r, c in zip(rows, ctrl)]
            dcell = f"{signed(statistics.mean(delta), 1)}\\,{{\\tiny$\\pm$\\,{statistics.stdev(delta):.1f}}}"
        else:
            dcell = "--"
        prefix = "\\rowcolor{gray!12} " if "baseline" in label else ""
        w(prefix + " & ".join([label, *cells, dcell]) + " \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}")
    return out


PREVIEW = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{booktabs,makecell,threeparttable,rotating,tabularx,amssymb}
\usepackage[table]{xcolor}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\begin{document}
\input{comparison_table}
\end{document}
"""


def main():
    lines = (["% Generated by evaluation/tables/make_comparison_table.py -- do not edit by hand.",
              "% Needs in the preamble: booktabs, makecell, threeparttable, rotating, tabularx, amssymb,",
              "% \\usepackage[table]{xcolor} and \\newcolumntype{L}{>{\\raggedright\\arraybackslash}X}", ""]
             + table_setup() + table_variants() + table_main() + table_before_after() + table_seeds())
    (HERE / "comparison_table.tex").write_text("\n".join(lines) + "\n")
    (HERE / "preview.tex").write_text(PREVIEW)
    print("wrote", HERE / "comparison_table.tex")


if __name__ == "__main__":
    main()
