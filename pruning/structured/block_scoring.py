"""Shared block selection for block-level structured pruning of the fine-tuned ResNet-18.

Used by pruning/structured/whole_block (stage 1: whole identity-shortcut blocks) and, later,
pruning/structured/residual_branch (stage 2: the residual branch of any block, the shortcut kept).

Candidates: the 8 BasicBlocks, named "layer<i>.<j>".
  mode="identity" (stage 1): only the blocks whose shortcut is the identity (layer1.0, layer1.1, layer2.1, layer3.1,
      layer4.1). The block is replaced by nn.Identity. A BasicBlock computes relu(F(x) + x) and its input is
      non-negative (output of a ReLU or of the max pool), so the identity is exactly "drop F".
  mode="shortcut" (stage 2): all 8 blocks, the block replaced by its shortcut only. Not implemented yet.

Ranking subset: SUBSET_SIZE train images, stratified by diagnosis: proportional to the train split, at least
MIN_PER_CLASS per class (all images if a class has fewer), the largest class (nv) absorbs the difference. Drawn with a
fixed seed from the train rows of the split file, then shuffled (mixed classes in every BatchNorm batch). Evaluation
preprocessing (resize + normalize, no augmentation). The image ids are saved to CSV by the caller, so the subset is
reproducible, and the loader is checked against the split file: every image is train, and no lesion of the validation
or test split is in it.

BatchNorm recalibration: the running statistics of every BatchNorm are reset, momentum=None (cumulative average over
the batches), forward passes in train mode without gradients over the subset, then the momentum is restored (0.1, so a
later fine-tuning is unchanged). Always on copies: the full model (the reference) and every candidate removal are
recalibrated the same way, so a score measures the capacity lost, not stale statistics.

Score: increase of the class-weighted cross-entropy (the training loss; class weights from the FULL train split) on the
subset, candidate removed + recalibrated, versus full model + recalibrated. Greedy: the candidate with the smallest
increase is removed, then the remaining candidates are scored again with it removed, until none is left.
Logged, never used for selection: accuracy and macro F1 on the subset, loss and macro F1 without recalibration, and a
first-order Taylor score |dL/ds| at s = 1 for a scalar gate on the residual branch only, y = relu(s * F(x) + S(x))
(the gate multiplies the bn2 output; computed on the current recalibrated model of each step).
"""
import copy
import importlib.util
from pathlib import Path

STRUCTURED = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("filter_prune", STRUCTURED / "filter" / "prune.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)  # shared helpers of the structured runs (also chdir + sys.path to the fine-tuning project)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from ham10000.data import Ham10000Dataset, eval_transform  # noqa: E402

SPLITS = sp.PROJECT / "splits/ham10000_splits.csv"
BLOCKS = [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]
SUBSET_SIZE, MIN_PER_CLASS, BATCH = 2000, 50, 32
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # class_to_idx order (sorted names)


# ── Blocks ───────────────────────────────────────────────────────────────────

def get_block(model, name):
    layer, index = name.split(".")
    return getattr(model, layer)[int(index)]


def set_block(model, name, module):
    layer, index = name.split(".")
    getattr(model, layer)[int(index)] = module


def candidates(model, mode):
    """Prunable block names of an unpruned model."""
    if mode == "identity":
        return [n for n in BLOCKS if get_block(model, n).downsample is None]
    if mode == "shortcut":
        return list(BLOCKS)
    raise ValueError(mode)


def replace_block(model, name, mode):
    """Replace one block in place, with standard torch modules only (the saved whole model loads without imports)."""
    if mode == "identity":
        if get_block(model, name).downsample is not None:
            raise sp.CheckFailed(f"{name} has a downsample shortcut: it cannot be replaced by the identity")
        set_block(model, name, nn.Identity())
    elif mode == "shortcut":
        raise NotImplementedError("residual-branch pruning (stage 2)")
    else:
        raise ValueError(mode)
    return model


def pruned_copy(model, names, mode):
    model = copy.deepcopy(model)
    for name in names:
        replace_block(model, name, mode)
    return model


# ── Ranking subset ───────────────────────────────────────────────────────────

def stratified_subset(seed, size=SUBSET_SIZE, min_per_class=MIN_PER_CLASS):
    """Train rows of the split file: (subset in shuffled order, images per class)."""
    split = pd.read_csv(SPLITS)
    train = split[split["partition"] == "train"].sort_values("image_id").reset_index(drop=True)
    counts = train["dx"].value_counts().sort_index()
    alloc = np.minimum(counts, np.maximum(min_per_class, np.round(size * counts / counts.sum()))).astype(int)
    alloc[alloc.idxmax()] += size - alloc.sum()
    rng = np.random.RandomState(seed)
    rows = [train[train["dx"] == dx].iloc[np.sort(rng.choice(counts[dx], n, replace=False))]
            for dx, n in alloc.items()]
    subset = pd.concat(rows)
    subset = subset.iloc[rng.permutation(len(subset))].reset_index(drop=True)
    return subset[["image_id", "lesion_id", "dx", "partition"]], alloc


def check_train_only(dataset, indices):
    """The images actually loaded are train images, and none shares a lesion with validation or test."""
    split = pd.read_csv(SPLITS)
    rows = dataset.data.iloc[indices]
    partition = rows["image_id"].map(split.set_index("image_id")["partition"])
    held_out = set(split.loc[split["partition"] != "train", "lesion_id"])
    result = {"images": len(rows), "unique_images": int(rows["image_id"].nunique()),
              "not_in_train_partition": int((partition != "train").sum()),
              "sharing_a_validation_or_test_lesion": int(rows["lesion_id"].isin(held_out).sum())}
    if result["not_in_train_partition"] or result["sharing_a_validation_or_test_lesion"] or \
            result["unique_images"] != len(rows):
        raise sp.CheckFailed(f"ranking subset is not train-only: {result}")
    return result


def load_subset(config, subset, device):
    """(batches of (images, labels) on device, leakage check). Evaluation preprocessing, subset order kept."""
    dataset = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=eval_transform(config))
    if [c for c, _ in sorted(dataset.class_to_idx.items(), key=lambda kv: kv[1])] != CLASSES:
        raise sp.CheckFailed(f"unexpected class_to_idx {dataset.class_to_idx}")
    position = pd.Series(np.arange(len(dataset.data)), index=dataset.data["image_id"])
    indices = position.loc[subset["image_id"]].tolist()
    leakage = check_train_only(dataset, indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=BATCH, shuffle=False, num_workers=config.num_workers)
    batches = [(x.to(device), y.to(device)) for x, y in loader]
    labels = torch.cat([y for _, y in batches]).cpu().numpy()
    if not (labels == subset["dx"].map(CLASSES.index).to_numpy()).all():
        raise sp.CheckFailed("subset labels do not match the split file")
    return batches, leakage


# ── BatchNorm recalibration and scores ───────────────────────────────────────

@torch.no_grad()
def recalibrate_bn(model, batches):
    """In place: reset every BatchNorm, re-estimate with a cumulative average (train-mode passes), restore momentum."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for x, _ in batches:
        model(x)
    for m, momentum in zip(bns, momenta):
        m.momentum = momentum
    return model.eval()


@torch.no_grad()
def subset_scores(model, batches, criterion):
    model.eval()
    logits = torch.cat([model(x) for x, _ in batches])
    y = torch.cat([y for _, y in batches])
    pred = logits.argmax(1)
    return {"loss": float(criterion(logits, y)), "accuracy": float((pred == y).float().mean()),
            "f1_macro": float(f1_score(y.cpu(), pred.cpu(), average="macro", labels=list(range(len(CLASSES))),
                                       zero_division=0))}


def taylor_scores(model, names, batches, criterion):
    """|dL/ds| at s = 1, gate on the residual branch: y = relu(s * bn2(conv2(...)) + S(x)). L = the subset loss
    (same weighted mean as criterion). Logged only."""
    model = copy.deepcopy(model).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    gates = {n: torch.ones((), device=device, requires_grad=True) for n in names}
    hooks = [get_block(model, n).bn2.register_forward_hook(lambda module, inputs, out, g=gates[n]: out * g)
             for n in names]
    total_weight = sum(float(criterion.weight[y].sum()) for _, y in batches)
    for x, y in batches:
        (F.cross_entropy(model(x), y, weight=criterion.weight, reduction="sum") / total_weight).backward()
    for h in hooks:
        h.remove()
    return {n: float(g.grad.abs()) for n, g in gates.items()}


def greedy_ranking(model, names, batches, criterion, mode, log=print):
    """(removal order, one row per step and candidate). model is never modified."""
    reference = subset_scores(recalibrate_bn(copy.deepcopy(model), batches), batches, criterion)
    no_recal = subset_scores(model, batches, criterion)
    rows = [{"step": 0, "candidate": "none (full model)", "removed_before": "", "loss": reference["loss"],
             "loss_increase": 0.0, "accuracy": reference["accuracy"], "f1_macro": reference["f1_macro"],
             "loss_no_bn_recal": no_recal["loss"], "f1_macro_no_bn_recal": no_recal["f1_macro"]}]
    log(f"full model: loss {reference['loss']:.4f} (without recalibration {no_recal['loss']:.4f}), "
        f"F1 {reference['f1_macro']:.4f}")
    removed, remaining = [], list(names)
    while remaining:
        current = recalibrate_bn(pruned_copy(model, removed, mode), batches)
        taylor = taylor_scores(current, remaining, batches, criterion)
        step = []
        for name in remaining:
            pruned = pruned_copy(model, removed + [name], mode)
            raw = subset_scores(pruned, batches, criterion)
            recal = subset_scores(recalibrate_bn(pruned, batches), batches, criterion)
            step.append({"step": len(removed) + 1, "candidate": name, "removed_before": " ".join(removed),
                         "loss": recal["loss"], "loss_increase": recal["loss"] - reference["loss"],
                         "accuracy": recal["accuracy"], "f1_macro": recal["f1_macro"],
                         "loss_no_bn_recal": raw["loss"], "f1_macro_no_bn_recal": raw["f1_macro"],
                         "taylor": taylor[name]})
        step.sort(key=lambda r: r["loss_increase"])
        by_taylor = sorted(step, key=lambda r: r["taylor"])
        for rank, row in enumerate(step, 1):
            row.update(rank_by_loss=rank, rank_by_taylor=by_taylor.index(row) + 1, chosen=rank == 1)
        removed.append(step[0]["candidate"])
        remaining.remove(step[0]["candidate"])
        rows += step
        log(f"step {len(removed)}: " + ", ".join(f"{r['candidate']} +{r['loss_increase']:.4f} (F1 {r['f1_macro']:.3f})"
                                                 for r in step) + f" -> remove {removed[-1]}")
    return removed, pd.DataFrame(rows)


@torch.no_grad()
def exactness_check(model, names, x):
    """Per identity block: max |logits(block -> nn.Identity) - logits(original, bn2 weight and bias = 0)|, and the
    smallest value entering the block (must be >= 0 for relu(x) = x)."""
    model = copy.deepcopy(model).eval()
    smallest = {}
    hooks = [get_block(model, n).register_forward_pre_hook(
        lambda module, inputs, n=n: smallest.__setitem__(n, float(inputs[0].min()))) for n in names]
    model(x)
    for h in hooks:
        h.remove()
    result = {}
    for name in names:
        zeroed = copy.deepcopy(model)
        get_block(zeroed, name).bn2.weight.zero_()
        get_block(zeroed, name).bn2.bias.zero_()
        result[name] = {"max_abs_logit_difference": float((pruned_copy(model, [name], "identity")(x)
                                                           - zeroed(x)).abs().max()),
                        "min_block_input": smallest[name]}
    return result
