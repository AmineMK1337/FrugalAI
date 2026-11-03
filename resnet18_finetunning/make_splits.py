"""
Fixed, lesion-aware, class-stratified train / validation / test split of HAM10000
==================================================================================
1. Verifies the dataset (image count, lesion count, official class counts, one .jpg
   per image_id, one dx per lesion_id) and stops if anything is off.
2. Splits LESIONS, not images, stratified by dx: 70% train, then the remaining 30%
   halved into 15% validation and 15% test (seed 42). Every image of a lesion lands
   in the same partition. HAM10000 has no patient identifier, so patient-level
   grouping is not possible.
3. Writes splits/{train,validation,test,ham10000_splits}.csv, split_summary.csv,
   split_config.json, integrity_report.txt and figures/. No image is copied,
   moved, resized or augmented.

Usage (from resnet18_finetunning/):
  python make_splits.py
  python make_splits.py --seed 42 --train 0.70 --val 0.15 --test 0.15
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_VERSION = "1.0"
HERE = Path(__file__).resolve().parent
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
OFFICIAL_COUNTS = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}
EXPECTED_IMAGES, EXPECTED_LESIONS = 10015, 7470
IMAGE_FOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
PARTITIONS = ["train", "validation", "test"]
METADATA_COLUMNS = ["age", "sex", "localization", "dx_type"]
OLD_TEST_SPLIT = HERE.parent / "skin-lesion-classification" / "results" / "test_split.csv"  # read only


def parse_args():
    p = argparse.ArgumentParser(description="Lesion-aware stratified split of HAM10000")
    p.add_argument("--metadata", default="../dataset/HAM10000_metadata", help="metadata CSV (relative to this folder)")
    p.add_argument("--image-dir", default="../dataset", help="folder containing HAM10000_images_part_1/2")
    p.add_argument("--out-dir", default="splits")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if abs(args.train + args.val + args.test - 1) > 1e-9:
        p.error("train + val + test must sum to 1")
    return args


def fail(msg):
    print(f"\nDATASET CHECK FAILED: {msg}\nNo split written.", file=sys.stderr)
    sys.exit(1)


def verify_dataset(meta, image_dir):
    """Section 1 checks. Stops on the first failure; returns image_id -> path and a log of the checks."""
    log = []
    if len(meta) != EXPECTED_IMAGES:
        fail(f"{len(meta)} metadata rows, expected {EXPECTED_IMAGES}")
    if meta["image_id"].duplicated().any():
        fail(f"duplicate image_id in metadata: {meta.loc[meta['image_id'].duplicated(), 'image_id'].tolist()[:5]}")
    log.append(f"metadata rows: {len(meta)} (expected {EXPECTED_IMAGES}), image_id unique")

    n_lesions = meta["lesion_id"].nunique()
    if n_lesions != EXPECTED_LESIONS:
        fail(f"{n_lesions} unique lesion_id, expected {EXPECTED_LESIONS}")
    log.append(f"unique lesion_id: {n_lesions} (expected {EXPECTED_LESIONS})")

    counts = meta["dx"].value_counts().to_dict()
    if counts != OFFICIAL_COUNTS:
        fail(f"class counts {counts} differ from the official {OFFICIAL_COUNTS}")
    log.append("class counts match the official ones: " + ", ".join(f"{c} {OFFICIAL_COUNTS[c]}" for c in OFFICIAL_COUNTS))

    dx_per_lesion = meta.groupby("lesion_id")["dx"].nunique()
    if (dx_per_lesion != 1).any():
        fail(f"{int((dx_per_lesion != 1).sum())} lesion_id have more than one dx, e.g. {dx_per_lesion[dx_per_lesion != 1].index[:5].tolist()}")
    log.append("every lesion_id has exactly one dx")

    files = {}
    for folder in IMAGE_FOLDERS:
        for f in (image_dir / folder).glob("*.jpg"):
            files.setdefault(f.stem, []).append(f)
    multiple = {k: v for k, v in files.items() if len(v) > 1}
    missing = sorted(set(meta["image_id"]) - set(files))
    extra = sorted(set(files) - set(meta["image_id"]))
    if missing:
        fail(f"{len(missing)} image_id without a .jpg, e.g. {missing[:5]}")
    if any(k in multiple for k in meta["image_id"]):
        fail(f"{len(multiple)} image_id with more than one .jpg, e.g. {list(multiple)[:5]}")
    log.append(f"every image_id has exactly one .jpg in {' / '.join(IMAGE_FOLDERS)} "
               f"({len(files)} .jpg files found, {len(extra)} not in the metadata)")
    return {k: v[0] for k, v in files.items()}, log


def split_lesions(meta, args):
    """70/15/15 of lesions, stratified by dx; the holdout is split again, stratified."""
    lesions = meta.groupby("lesion_id", sort=True)["dx"].first().reset_index()  # sorted: independent of row order
    train_l, rest_l = train_test_split(lesions, train_size=args.train, stratify=lesions["dx"], random_state=args.seed)
    val_share = args.val / (args.val + args.test)
    val_l, test_l = train_test_split(rest_l, train_size=val_share, stratify=rest_l["dx"], random_state=args.seed)
    assignment = pd.concat([train_l.assign(partition="train"), val_l.assign(partition="validation"),
                            test_l.assign(partition="test")])
    return dict(zip(assignment["lesion_id"], assignment["partition"]))


def old_split_partition(meta):
    """Re-create the old project's unstratified 80/10/10 lesion split (old src/ham10000/data.py, lesion_split),
    without importing the old code, and check it against the old results/test_split.csv."""
    lesion_ids = meta["lesion_id"].unique()  # the old code used the metadata order
    old_train, holdout = train_test_split(lesion_ids, train_size=0.8, random_state=42)
    old_val, old_test = train_test_split(holdout, test_size=0.5, random_state=42)
    part = {**{l: "train" for l in old_train}, **{l: "validation" for l in old_val}, **{l: "test" for l in old_test}}
    old_partition = meta["lesion_id"].map(part)
    verified = None
    if OLD_TEST_SPLIT.exists():
        saved = set(pd.read_csv(OLD_TEST_SPLIT)["image_id"])
        verified = saved == set(meta.loc[old_partition == "test", "image_id"])
    return old_partition, verified


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_fig(fig, out_dir, name):
    # No timestamp in the PDF, so repeated runs give byte-identical files
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "svg.hashsalt": "ham10000",
    })
    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    counts = pd.crosstab(df["dx"], df["partition"]).reindex(index=CLASSES, columns=PARTITIONS)
    x, w = np.arange(len(CLASSES)), 0.27

    # 1. Images per class and partition (log axis so the small classes stay visible)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, p in enumerate(PARTITIONS):
        bars = ax.bar(x + (k - 1) * w, counts[p], w * 0.92, color=colors[p], label=p)
        for b, v in zip(bars, counts[p]):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.08, str(v), ha="center", va="bottom", fontsize=6)
    ax.set_yscale("log")
    ax.set_ylim(top=counts.values.max() * 3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Images (log scale)")
    ax.set_title("Images per class and partition")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "images_per_class")

    # 2. Class proportions within each partition, next to the full dataset
    props = (counts / counts.sum() * 100)
    props["full dataset"] = df["dx"].value_counts(normalize=True).reindex(CLASSES) * 100
    cols = PARTITIONS + ["full dataset"]
    colors_all = {**colors, "full dataset": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w2 = 0.2
    for k, c in enumerate(cols):
        ax.bar(x + (k - 1.5) * w2, props[c], w2 * 0.92, color=colors_all[c], label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("% of the partition's images")
    ax.set_title("Class proportions per partition")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "class_proportions")

    # 3. Overall partition proportions, images and lesions
    img = df["partition"].value_counts(normalize=True).reindex(PARTITIONS) * 100
    les = df.drop_duplicates("lesion_id")["partition"].value_counts(normalize=True).reindex(PARTITIONS) * 100
    fig, ax = plt.subplots(figsize=(5, 3))
    xp = np.arange(len(PARTITIONS))
    for k, (name, s, hatch) in enumerate([("images", img, None), ("lesions", les, "//")]):
        bars = ax.bar(xp + (k - 0.5) * 0.36, s, 0.33, color=[colors[p] for p in PARTITIONS], hatch=hatch,
                      edgecolor="white", label=name)
        for b, v in zip(bars, s):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xp)
    ax.set_xticklabels(PARTITIONS)
    ax.set_ylim(0, 85)
    ax.set_ylabel("% of all images / lesions")
    ax.set_title("Partition sizes (plain: images, hatched: lesions)")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "partition_proportions")


def main():
    args = parse_args()
    metadata_path = (HERE / args.metadata).resolve()
    image_dir = (HERE / args.image_dir).resolve()
    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, newline="") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    meta = pd.read_csv(metadata_path, sep=sep)
    print(f"Metadata: {metadata_path} ({'tab' if sep == chr(9) else 'comma'}-separated), columns: {list(meta.columns)}")

    paths, dataset_log = verify_dataset(meta, image_dir)
    print("Dataset checks passed:\n  " + "\n  ".join(dataset_log))

    # --- split ---
    lesion_partition = split_lesions(meta, args)
    df = meta.copy()
    df["partition"] = df["lesion_id"].map(lesion_partition)
    df["image_path"] = [Path(os.path.relpath(paths[i], HERE)).as_posix() for i in df["image_id"]]  # relative to this folder
    df = df[["image_id", "image_path", "dx", "lesion_id", "partition"] + METADATA_COLUMNS]
    df["_order"] = df["partition"].map({p: i for i, p in enumerate(PARTITIONS)})
    df = df.sort_values(["_order", "image_id"]).drop(columns="_order").reset_index(drop=True)

    csv_files = {}
    for p in PARTITIONS:
        name = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}[p]
        df[df["partition"] == p].to_csv(out_dir / name, index=False, lineterminator="\n")
        csv_files[name] = out_dir / name
    df.to_csv(out_dir / "ham10000_splits.csv", index=False, lineterminator="\n")
    csv_files["ham10000_splits.csv"] = out_dir / "ham10000_splits.csv"

    # --- summary per class and partition ---
    rows = []
    for c in CLASSES + ["all classes"]:
        sub = df if c == "all classes" else df[df["dx"] == c]
        for p in PARTITIONS + ["total"]:
            part = sub if p == "total" else sub[sub["partition"] == p]
            rows.append({"class": c, "partition": p, "images": len(part), "lesions": part["lesion_id"].nunique(),
                         "pct_of_class_images": round(len(part) / len(sub) * 100, 2),
                         "pct_of_class_lesions": round(part["lesion_id"].nunique() / sub["lesion_id"].nunique() * 100, 2),
                         "pct_of_partition_images": round(len(part) / len(df if p == "total" else df[df["partition"] == p]) * 100, 2)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "split_summary.csv", index=False, lineterminator="\n")
    csv_files["split_summary.csv"] = out_dir / "split_summary.csv"

    # --- integrity checks ---
    report, ok_all = [], True

    def check(n, title, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        report.append(f"[{'PASS' if ok else 'FAIL'}] {n}. {title}\n      {detail}")

    img_parts = df.groupby("image_id")["partition"].nunique()
    check(1, "No image_id in more than one partition", (img_parts == 1).all() and not df["image_id"].duplicated().any(),
          f"{int((img_parts > 1).sum())} image_id in several partitions; {int(df['image_id'].duplicated().sum())} duplicate rows")
    les_parts = df.groupby("lesion_id")["partition"].nunique()
    check(2, "No lesion_id in more than one partition", (les_parts == 1).all(),
          f"{int((les_parts > 1).sum())} lesion_id in several partitions")
    missing = [p for p in df["image_path"] if not (HERE / p).is_file()]
    check(3, "All image paths exist", not missing, f"{len(missing)} missing of {len(df)}")
    bad_dx = sorted(set(df["dx"]) - set(CLASSES))
    check(4, "All dx values are valid codes", not bad_dx, f"invalid: {bad_dx}" if bad_dx else f"codes present: {sorted(set(df['dx']))}")
    counts = pd.crosstab(df["dx"], df["partition"]).reindex(index=CLASSES, columns=PARTITIONS).fillna(0).astype(int)
    absent = [(c, p) for c in CLASSES for p in PARTITIONS if counts.loc[c, p] == 0]
    check(5, "All 7 classes present in every partition", not absent,
          f"smallest cell: {counts.values.min()} images ({counts.stack().idxmin()})" if not absent else f"absent: {absent}")
    sizes = df["partition"].value_counts().reindex(PARTITIONS)
    check(6, f"Partition sizes add up to {EXPECTED_IMAGES}", sizes.sum() == EXPECTED_IMAGES,
          " + ".join(f"{p} {sizes[p]}" for p in PARTITIONS) + f" = {sizes.sum()}")
    les_sizes = df.drop_duplicates("lesion_id")["partition"].value_counts().reindex(PARTITIONS)
    check(7, "Actual image and lesion proportions per partition", True,
          "; ".join(f"{p}: images {sizes[p] / len(df):.2%}, lesions {les_sizes[p] / les_sizes.sum():.2%}" for p in PARTITIONS))
    full_prop = df["dx"].value_counts(normalize=True).reindex(CLASSES)
    diff = (counts / counts.sum()).sub(full_prop, axis=0).abs() * 100
    worst = diff.stack().idxmax()
    check(8, "Class proportions per partition within 1 pp of the full dataset", diff.values.max() < 1,
          f"max difference {diff.values.max():.3f} pp ({worst[0]} in {worst[1]})")

    old_partition, old_verified = old_split_partition(meta)
    new_test = df.loc[df["partition"] == "test", "image_id"]
    positions = pd.Index(meta["image_id"]).get_indexer(new_test)
    old_of_new_test = old_partition.iloc[positions].value_counts()
    detail = (f"old split re-created and {'matches' if old_verified else 'DOES NOT match'} {OLD_TEST_SPLIT.as_posix()}; "
              if old_verified is not None else "old test_split.csv not found, old split re-created only; ")
    detail += (f"of the {len(new_test)} new test images, {old_of_new_test.get('train', 0)} were in the old TRAIN set "
               f"({old_of_new_test.get('train', 0) / len(new_test):.1%}), {old_of_new_test.get('validation', 0)} in old validation, "
               f"{old_of_new_test.get('test', 0)} in old test")
    check(9, "Comparison with the old split (information)", old_verified is not False, detail)

    report_text = "\n".join([f"HAM10000 split integrity report (make_splits.py v{SCRIPT_VERSION}, seed {args.seed})",
                             "", "Dataset checks:"] + [f"  [PASS] {l}" for l in dataset_log] + ["", "Split checks:"]
                            + report + ["", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"])
    (out_dir / "integrity_report.txt").write_text(report_text + "\n", encoding="utf-8")
    print("\n" + report_text)

    make_figures(df, out_dir / "figures")

    config = {
        "script": "make_splits.py", "script_version": SCRIPT_VERSION,
        "seed": args.seed,
        "target_proportions_of_lesions": {"train": args.train, "validation": args.val, "test": args.test},
        "method": ("lesion-level split stratified by dx: sklearn train_test_split on the lesion table "
                   "(one row per lesion_id, sorted) with stratify=dx; first train vs rest, then rest into validation "
                   "vs test; every image follows its lesion. No patient-level grouping (HAM10000 has no patient id)."),
        "metadata_file": Path(args.metadata).as_posix(), "metadata_separator": "tab" if sep == "\t" else "comma",
        "image_dir": Path(args.image_dir).as_posix(),
        "dataset_counts": {"images": len(meta), "lesions": int(meta["lesion_id"].nunique()),
                           "classes": {c: OFFICIAL_COUNTS[c] for c in CLASSES}},
        "partition_counts": {p: {"images": int(sizes[p]), "lesions": int(les_sizes[p])} for p in PARTITIONS},
        "sha256": {name: sha256(path) for name, path in csv_files.items()},
        "all_checks_passed": bool(ok_all),
    }
    with open(out_dir / "split_config.json", "w", newline="\n") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_dir.as_posix()}: " + ", ".join(sorted(p.name for p in out_dir.iterdir())))
    if not ok_all:
        sys.exit(2)


if __name__ == "__main__":
    main()
