"""Block ranking for whole-block pruning (stage 1): which identity-shortcut BasicBlocks to remove, in which order.

Method: pruning/structured/block_scoring.py (greedy, class-weighted cross-entropy increase after BatchNorm
recalibration, on a stratified subset of 2,000 TRAIN images). The ranking is computed for each subset seed; the FIRST
seed's order is the one used for pruning, the others are a stability check. Every run starts from the same FP32
checkpoint (strict load), which is never modified; validation and test images are never loaded.

Writes pruning/structured/whole_block/results/selection/:
  subset_seed<s>.csv     image_id, lesion_id, dx, partition of the ranking subset (loader order)
  ranking_seed<s>.csv    one row per greedy step and candidate (scores, ranks, chosen)
  ranking_comparison.csv removal position and single-removal score of every candidate, per seed
  checks.json            subset per class, leakage check, exactness check, class weights, orders, hashes
  run.log

Usage: python pruning/structured/whole_block/rank_blocks.py [--subset-seeds 42 1]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import block_scoring as bs  # noqa: E402  (chdir to the fine-tuning project)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from ham10000.data import build_dataloaders, compute_class_weights  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

sp = bs.sp
OUT = HERE / "results" / "selection"
MODE = "identity"


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    logfile = OUT / "run.log"
    logfile.write_text("")
    log = lambda message: sp.log(message, logfile)  # noqa: E731
    if not torch.cuda.is_available():
        raise sp.CheckFailed("no GPU")
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    hashes = {str(p): sp.sha256(p) for p in sp.PROTECTED}
    config = load_config(sp.CONFIG)
    train_loader, _, _ = build_dataloaders(config)  # only for the train-split class weights; no image is loaded
    weights = compute_class_weights(train_loader, sp.NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = sp.load_fp32().to(device)
    names = bs.candidates(model, MODE)
    log(f"candidates ({MODE}): {names}; class weights {dict(zip(bs.CLASSES, weights.tolist()))}")

    checks = {"mode": MODE, "candidates": names, "subset_size": bs.SUBSET_SIZE, "min_per_class": bs.MIN_PER_CLASS,
              "class_weights": dict(zip(bs.CLASSES, weights.tolist())), "seeds": {}}
    orders, tables = {}, {}
    for seed in args.subset_seeds:
        subset, alloc = bs.stratified_subset(seed)
        subset.to_csv(OUT / f"subset_seed{seed}.csv", index=False)
        batches, leakage = bs.load_subset(config, subset, device)
        log(f"subset seed {seed}: {len(subset)} train images {dict((k, int(v)) for k, v in alloc.items())}; leakage check {leakage}")
        entry = {"images_per_class": {k: int(v) for k, v in alloc.items()}, "leakage_check": leakage}
        if seed == args.subset_seeds[0]:
            exact = bs.exactness_check(model, names, batches[0][0])
            log(f"exactness (nn.Identity vs bn2 weight and bias = 0, {len(batches[0][0])} real images): {exact}")
            if any(v["max_abs_logit_difference"] > 1e-5 or v["min_block_input"] < 0 for v in exact.values()):
                raise sp.CheckFailed(f"identity replacement is not exact: {exact}")
            entry["exactness_check"] = exact
        orders[seed], tables[seed] = bs.greedy_ranking(model, names, batches, criterion, MODE, log)
        tables[seed].to_csv(OUT / f"ranking_seed{seed}.csv", index=False)
        entry["removal_order"] = orders[seed]
        checks["seeds"][seed] = entry
        log(f"subset seed {seed}: removal order {orders[seed]}")
        del batches
        torch.cuda.empty_cache()

    rows = []
    for name in names:
        row = {"block": name}
        for seed in args.subset_seeds:
            single = tables[seed][(tables[seed]["step"] == 1) & (tables[seed]["candidate"] == name)].iloc[0]
            row.update({f"position_seed{seed}": orders[seed].index(name) + 1,
                        f"single_removal_loss_increase_seed{seed}": single["loss_increase"],
                        f"single_removal_f1_seed{seed}": single["f1_macro"],
                        f"single_removal_taylor_seed{seed}": single["taylor"]})
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(f"position_seed{args.subset_seeds[0]}")
    comparison.to_csv(OUT / "ranking_comparison.csv", index=False)
    first = args.subset_seeds[0]
    for seed in args.subset_seeds[1:]:
        tau = kendalltau(comparison[f"position_seed{first}"], comparison[f"position_seed{seed}"]).statistic
        checks[f"kendall_tau_seed{first}_vs_seed{seed}"] = float(tau)
        # per k: same blocks removed? same number of blocks removed per stage (same parameters and MACs)?
        per_k = [{"k": k, f"removed_seed{first}": orders[first][:k], f"removed_seed{seed}": orders[seed][:k],
                  "same_blocks": set(orders[first][:k]) == set(orders[seed][:k]),
                  "same_blocks_per_stage": sorted(n.split(".")[0] for n in orders[first][:k])
                  == sorted(n.split(".")[0] for n in orders[seed][:k])} for k in range(1, len(names) + 1)]
        checks[f"removed_sets_seed{first}_vs_seed{seed}"] = per_k
        log(f"removed blocks per k, seed {first} vs {seed}: same blocks for k = "
            f"{[r['k'] for r in per_k if r['same_blocks']]}, same blocks per stage for k = "
            f"{[r['k'] for r in per_k if r['same_blocks_per_stage']]}")
        log(f"order seed {first} vs seed {seed}: identical {orders[first] == orders[seed]}, Kendall tau {tau:.3f}")
    checks["order_used_for_pruning"] = orders[first]

    if {str(p): sp.sha256(p) for p in sp.PROTECTED} != hashes:
        raise sp.CheckFailed("the FP32 checkpoint, the split or the config changed during the run")
    checks["sha256_protected"] = hashes
    checks["torch"] = torch.__version__
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    log("protected files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-seeds", type=int, nargs="+", default=[42, 1])
    try:
        main(parser.parse_args())
    except sp.CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
