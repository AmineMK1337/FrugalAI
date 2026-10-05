# Model Evaluation Metrics

This document defines and explains all metrics used to evaluate compressed deep learning models
in the FrugalAI project. The primary task is **multiclass skin-lesion classification** on the
HAM10000 dataset using ResNet18, with compression techniques including FP16, INT8 Static
Post-Training Quantization (PTQ), and INT8 Quantization-Aware Training (QAT).

Metrics are organised into three categories:

1. **Predictive Performance** — classification quality on the medical task.
2. **Computational Efficiency** — resource requirements and compression gains.
3. **Environmental Efficiency** — energy consumption and estimated carbon footprint.

---

## 1. Predictive Performance

These metrics quantify how well a model performs the medical image classification task, and
whether compression degrades its diagnostic utility.

### 1.1 Multiclass Classification in HAM10000

HAM10000 contains **7 skin-lesion classes** (e.g., melanoma, basal cell carcinoma, nevus).
Binary metrics such as True Positives (TP), True Negatives (TN), False Positives (FP), and
False Negatives (FN) are extended to the multiclass setting using a **One-vs-Rest (OvR)**
strategy: for each class $c$, all other classes are treated as a single negative group. This
yields per-class confusion matrix values from which per-class metrics are derived.

Three aggregation strategies are used:

| Strategy | Description | When to use |
|---|---|---|
| **Per-class** | One metric value per class $c \in \{0,\ldots,K-1\}$ | Diagnosing which classes are most affected by compression |
| **Macro average** | Unweighted mean across all $K$ classes | Preferred when classes are imbalanced — each class counts equally |
| **Weighted average** | Mean weighted by class support (number of true samples per class) | Useful for reporting overall dataset-level performance |

> [!IMPORTANT]
> HAM10000 is **significantly class-imbalanced** (nevus dominates, dermatofibroma is rare).
> **Macro averaging is the primary aggregation** for all F1, Precision, Recall, Sensitivity,
> and Specificity metrics. A model that only learns the majority class can achieve high
> weighted-average F1 while catastrophically failing on rare but clinically critical classes.

---

### 1.2 Accuracy

**What it measures:** The proportion of all test samples that are correctly classified.

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} = \frac{\sum_{c} \text{TP}_c}{N}$$

where $N$ is the total number of test samples and the sum is taken over all classes under OvR.
In practice this reduces to the count of samples where the predicted label equals the true label.

**Units:** Dimensionless; typically expressed as a percentage (%) or as a value in $[0, 1]$.

**Interpretation:**
- A value of 1.0 (100%) indicates perfect classification.
- A value near the majority-class frequency (e.g., 67% for a 67%/33% split) may indicate a
  degenerate classifier that predicts only the dominant class.

**Relevance to medical classification:** Accuracy is a necessary but insufficient metric for
imbalanced medical datasets. A classifier that labels every image as "nevus" on HAM10000 would
achieve high accuracy while completely failing to detect melanoma. Always report alongside
Macro-F1 and per-class Recall.

**Comparing FP32 vs quantized models:** Accuracy degradation caused by quantization should be
small (typically < 1% for well-calibrated INT8 PTQ). A meaningful drop (e.g., > 2%) warrants
further analysis, particularly checking which classes are most affected.

---

### 1.3 Precision

**What it measures:** Of all samples predicted as class $c$, the fraction that truly belong to
class $c$. Low Precision means many false alarms (FP).

**Per-class formula (OvR for class $c$):**

$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$

**Macro-averaged Precision:**

$$\text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{c=0}^{K-1} \text{Precision}_c$$

**Units:** Dimensionless, in $[0, 1]$.

**Relevance to medical classification:** High Precision is important when a false positive
(e.g., flagging a benign lesion as malignant) leads to unnecessary interventions, patient
anxiety, or costly follow-up procedures.

**Comparing FP32 vs quantized models:** Precision can fluctuate between quantization variants
even when overall accuracy is stable. Report per-class Precision to identify whether rare
classes suffer disproportionate degradation.

---

### 1.4 Recall (Sensitivity)

**What it measures:** Of all samples that truly belong to class $c$, the fraction correctly
identified. Also known as the **True Positive Rate (TPR)**. Low Recall means many missed
detections (FN).

**Per-class formula (OvR for class $c$):**

$$\text{Recall}_c = \text{Sensitivity}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

**Macro-averaged Recall / Sensitivity:**

$$\text{Sensitivity}_{\text{macro}} = \frac{1}{K} \sum_{c=0}^{K-1} \text{Sensitivity}_c$$

**Units:** Dimensionless, in $[0, 1]$.

**Relevance to medical classification:** Recall/Sensitivity is the **most clinically critical
metric** in cancer screening. A missed melanoma (FN) is far more dangerous than a false alarm.
Quantization-induced Recall degradation on the melanoma class is therefore a serious concern,
even if overall Accuracy is preserved.

**Comparing FP32 vs quantized models:** Track Sensitivity per class. If the INT8 model shows
reduced Recall on malignant classes (e.g., melanoma) compared to FP32, this represents a
clinically meaningful regression regardless of aggregate metric stability.

---

### 1.5 Specificity

**What it measures:** Of all samples that do **not** belong to class $c$ (the negatives), the
fraction correctly identified as negative. Also known as the **True Negative Rate (TNR)**.

**Per-class formula (OvR for class $c$):**

$$\text{Specificity}_c = \frac{\text{TN}_c}{\text{TN}_c + \text{FP}_c}$$

**Macro-averaged Specificity:**

$$\text{Specificity}_{\text{macro}} = \frac{1}{K} \sum_{c=0}^{K-1} \text{Specificity}_c$$

**Units:** Dimensionless, in $[0, 1]$.

**Relevance to medical classification:** High Specificity ensures that the model does not
over-trigger on healthy tissue. In a multiclass context, it quantifies how well the model
avoids confusing one disease type with another. In the OvR scheme, Specificity measures how
rarely the model labels a non-$c$ sample as class $c$.

**Comparing FP32 vs quantized models:** Specificity is generally well-preserved under
quantization since false positives on well-represented majority classes tend to be stable.
Monitor per-class Specificity to detect unexpected regression.

---

### 1.6 F1-Score

**What it measures:** The harmonic mean of Precision and Recall for a given class. It balances
both error types (FP and FN) into a single score.

**Per-class formula:**

$$\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$

**Macro-F1 (primary aggregate metric for this project):**

$$\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{c=0}^{K-1} \text{F1}_c$$

**Weighted-F1:**

$$\text{F1}_{\text{weighted}} = \frac{1}{\sum_c n_c} \sum_{c=0}^{K-1} n_c \cdot \text{F1}_c$$

where $n_c$ is the number of true samples of class $c$.

**Units:** Dimensionless, in $[0, 1]$.

**Why Macro-F1 is the primary metric for HAM10000:**
Macro-F1 gives **equal weight to each class regardless of support**. In HAM10000:
- The nevus class (~67% of samples) dominates Weighted-F1.
- Rare but clinically important classes (e.g., dermatofibroma, vascular lesions) are
  underrepresented.

Using Macro-F1 prevents a model that ignores rare classes from appearing deceptively strong.
It is the standard reporting metric for imbalanced medical classification benchmarks.

**Comparing FP32 vs quantized models:** Macro-F1 is the **primary performance indicator** for
compression evaluation. Report:
- Absolute Macro-F1 for each model variant.
- Absolute drop: delta F1_macro = F1_macro(FP32) - F1_macro(compressed)
- Per-class F1 to identify which classes degrade most.

---

### 1.7 ROC-AUC

**What it measures:** The Area Under the Receiver Operating Characteristic (ROC) curve. The
ROC curve plots True Positive Rate (Sensitivity) against False Positive Rate
(1 - Specificity) as the classification threshold varies. AUC represents the
probability that the model ranks a randomly chosen positive sample higher than a randomly
chosen negative sample.

**Binary case:** AUC is computed from the probability score assigned to the positive class.

**Multiclass — Macro ROC-AUC (One-vs-Rest):**

For each class $c$, a ROC curve is computed treating $c$ as positive and all other classes as
negative, using the model's predicted probability $\hat{p}_c$. The Macro ROC-AUC is:

$$\text{AUC}_{\text{macro}} = \frac{1}{K} \sum_{c=0}^{K-1} \text{AUC}_c$$

This is the formulation used in `sklearn.metrics.roc_auc_score` with
`multi_class="ovr"` and `average="macro"`, which is adopted in this project.

**Units:** Dimensionless, in $[0.5, 1]$ for a useful classifier (0.5 = random, 1.0 = perfect).

**Relevance to medical classification:** ROC-AUC is threshold-independent, making it suitable
for evaluating classifier rankings before a decision threshold is chosen. It is particularly
useful in medical imaging where operating points can be adjusted post hoc.

**Comparing FP32 vs quantized models:** AUC is computed from predicted **probabilities**, not
just hard labels. Quantization can subtly shift probability distributions. A drop in Macro AUC
indicates that the model's probability calibration is affected, not just its argmax decisions.

> [!NOTE]
> Macro ROC-AUC requires that all $K$ classes appear in the test set. If a class has zero
> samples in the test split, the score cannot be computed. Ensure the test set contains
> examples from every class.

---

## 2. Computational Efficiency

These metrics quantify how much a model has been compressed and how many computational
resources it requires at inference time.

---

### 2.1 Number of Parameters

**What it measures:** The total count of learnable scalar weights in the model (all layers
combined).

**How it is calculated:**

$$\text{Params} = \sum_{\ell} \text{numel}(\theta_\ell)$$

where $\text{numel}(\theta_\ell)$ is the number of elements in the parameter tensor of layer
$\ell$.

In PyTorch: `sum(p.numel() for p in model.parameters())`.

**Units:** Count (dimensionless). Typically reported in **millions (M)** for large networks.
ResNet18 has approximately **11.2 M** parameters in its standard configuration.

**Why it matters for edge deployment:** Parameter count determines the minimum storage required
to represent the model (before considering numerical precision). It also influences the
theoretical computational complexity of each forward pass.

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:** The number of parameters is **unchanged**
by quantization — quantization changes the numerical precision of stored values, not the model
architecture. The same parameter count across all variants is expected and is not an indicator
of compression effectiveness.

---

### 2.2 Model Size

**What it measures:** The actual storage size of the serialized model on disk, including all
parameter tensors in their stored numerical format.

**How it is measured:** Serialize the model (e.g., `torch.save(model.state_dict(), path)`) and
measure the resulting file size in bytes.

**Units:** Bytes. Typically reported in **Megabytes (MB)**:

$$\text{Model Size (MB)} = \frac{\text{File size in bytes}}{1024^2}$$

**Why it matters for edge deployment:** Storage size determines whether the model can be
deployed on resource-constrained devices (embedded systems, mobile devices, medical edge
hardware). Smaller models also load faster into memory.

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:**

| Precision | Bits per parameter | Expected size (ResNet18, ~11M params) |
|---|---|---|
| FP32 | 32 bits (4 bytes) | ~45 MB |
| FP16 | 16 bits (2 bytes) | ~23 MB |
| INT8 PTQ | 8 bits (1 byte) | ~12 MB |
| INT8 QAT | 8 bits (1 byte) | ~12 MB |

> [!NOTE]
> Actual serialized sizes may differ from theoretical estimates due to format overhead, scale
> factors, zero-points, and metadata stored alongside INT8 weights.

---

### 2.3 Compression Ratio

**What it measures:** The factor by which the compressed model's storage size is reduced
relative to the FP32 baseline.

**Formula:**

$$\text{Compression Ratio} = \frac{\text{Size}_{\text{FP32}}}{\text{Size}_{\text{compressed}}}$$

**Units:** Dimensionless ratio (reported as $n\times$).

**Illustrative example:**
If the FP32 model is 45 MB and the INT8 model is 12 MB:

$$\text{Compression Ratio} = \frac{45\,\text{MB}}{12\,\text{MB}} \approx 3.75\times$$

A ratio of $3.75\times$ means the INT8 model is 3.75 times smaller on disk than the FP32 model.

**Size reduction percentage:**

$$\text{Size Reduction (\%)} = \left(1 - \frac{\text{Size}_{\text{compressed}}}{\text{Size}_{\text{FP32}}}\right) \times 100$$

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:**

| Model | Expected Compression Ratio (vs FP32) |
|---|---|
| FP32 | 1.0x (baseline) |
| FP16 | ~2.0x |
| INT8 PTQ | ~3.5–4.0x |
| INT8 QAT | ~3.5–4.0x |

Actual ratios depend on the serialization format and framework overhead.

---

### 2.4 FLOPs / GFLOPs

**What it measures:** The theoretical number of **Floating-Point Operations** (FLOPs) required
to process one input sample through the model. One multiply-accumulate (MAC) operation is
conventionally counted as **2 FLOPs** (one multiplication + one addition).

**How it is calculated:**
FLOPs are derived analytically from the model architecture and input shape. For a convolutional
layer with kernel $k \times k$, $C_{\text{in}}$ input channels, $C_{\text{out}}$ output
channels, and output spatial size $H_{\text{out}} \times W_{\text{out}}$:

$$\text{FLOPs}_{\text{conv}} = 2 \cdot k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}}$$

Total model FLOPs sum contributions from all layers. Typically reported in **GFLOPs**
(Giga-FLOPs, $10^9$ FLOPs). ResNet18 at $224\times224$ input requires approximately
**1.8 GFLOPs** per image.

In this project, GFLOPs are computed via `torchinfo`:
`GFLOPs ~= 2 x total_mult_adds / 1e9`.

**Units:** FLOPs (dimensionless count); reported as GFLOPs ($\times 10^9$ FLOPs).

**Why it matters for edge deployment:** GFLOPs quantify the theoretical computational burden
of the model, independent of hardware. Lower GFLOPs generally correlate with lower energy
consumption and shorter inference time, but this relationship is hardware-dependent.

**FLOPs vs Latency — an important distinction:**

> [!IMPORTANT]
> **FLOPs describe theoretical complexity; latency measures actual execution time.**
>
> Two models with identical FLOPs can have very different latencies on the same hardware due to:
> - Memory bandwidth constraints and cache behaviour.
> - Hardware-specific instruction parallelism (e.g., INT8 SIMD units vs FP32 units).
> - Framework and kernel optimisation (e.g., FBGEMM, OneDNN).
> - Batch size and input shape affecting vectorisation efficiency.
>
> Always report **both** GFLOPs and measured latency. Do not substitute one for the other.

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:** FLOPs are an architectural property and
remain the same across precisions for a fixed architecture. The **effective throughput**
changes because INT8 hardware can perform more operations per clock cycle. GFLOPs are
therefore reported for FP32 as a reference; precision-aware FLOPs (e.g., INT8 MACs) are
not separately computed in this project.

---

### 2.5 Latency / Inference Time

**What it measures:** The wall-clock time required to process one input image through the
model (model forward pass only, excluding data loading and preprocessing).

**How it is measured:** Time the model's forward pass over a fixed batch of pre-loaded images
(to exclude I/O), repeat the measurement multiple times, and report the mean and standard
deviation:

$$\text{Latency (ms/image)} = \frac{\text{Total wall-clock time (ms)}}{\text{Number of images processed}}$$

Best practice:
- Run several warm-up iterations before timing (to prime caches and JIT compilation).
- Average over multiple repetitions (e.g., 5 runs in this project).
- Report the batch size used, as batched inference amortises fixed per-batch overhead.

**Units:** Milliseconds per image (ms/image), or total seconds for a batch.

**Why it matters for edge deployment:** Latency directly determines real-time usability. In a
clinical setting, a dermatology screening tool must provide near-instant feedback. Edge devices
(e.g., point-of-care hardware) may have strict latency budgets (e.g., < 100 ms/image).

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:** INT8 models can achieve **2–4x lower
latency** than FP32 on hardware with native INT8 SIMD support (e.g., x86 CPUs via FBGEMM,
ARM CPUs via QNNPACK). FP16 speedup is significant on GPUs with half-precision tensor cores
but may be minimal on CPUs lacking native FP16 arithmetic.

> [!WARNING]
> All latency comparisons must be performed **under identical hardware and software conditions**
> (same CPU/GPU, same number of threads, same batch size, same framework version). Changing
> any of these invalidates the comparison.

---

### 2.6 Throughput

**What it measures:** The number of images the model can process per unit time, the inverse of
per-image latency.

**Formula:**

$$\text{Throughput (images/s)} = \frac{1000}{\text{Latency (ms/image)}}$$

or equivalently:

$$\text{Throughput (images/s)} = \frac{\text{Number of images processed}}{\text{Total wall-clock time (s)}}$$

**Units:** Images per second (img/s).

**Why it matters for edge deployment:** Throughput determines the capacity of a deployed
system. For batch screening scenarios (e.g., processing a queue of dermoscopy images), higher
throughput translates directly to shorter turnaround time and lower total cost of ownership.

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:** An INT8 model with 4x lower latency
delivers 4x higher throughput, enabling more images to be screened per unit time or
per unit energy cost.

---

### 2.7 Memory Footprint

**What it measures:** The amount of RAM (CPU) or VRAM (GPU) consumed at inference time. This
includes:
- **Model weights** loaded into memory (proportional to model size and precision).
- **Activation tensors** allocated during the forward pass.
- **Framework runtime overhead** (PyTorch buffers, quantization metadata, etc.).

**How it is measured:** Profile peak memory allocation during inference using
`torch.cuda.max_memory_allocated()` (GPU) or system memory profilers (CPU). For CPU
inference, tools such as `tracemalloc`, `psutil`, or dedicated profiling frameworks can be
used.

**Units:** Megabytes (MB) or Gigabytes (GB).

**Why it matters for edge deployment:** Edge devices have constrained RAM. A model that fits
on disk as 12 MB (INT8) must also fit its activations in memory at runtime. Activation
memory scales with batch size and can exceed model weight memory for large intermediate
feature maps.

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:** INT8 and FP16 models reduce weight memory
proportionally to their precision reduction. Activation memory reduction depends on whether
intermediate tensors are also stored in reduced precision (framework-dependent). Report
**peak runtime memory** rather than model file size to capture the full operational footprint.

---

## 3. Environmental Efficiency

These metrics quantify the environmental cost of running inference, including energy
consumption and estimated greenhouse gas emissions.

---

### 3.1 Energy Consumption per Inference

**What it measures:** The average electrical energy consumed by the hardware to process one
image through the model.

**Formula:**

$$\text{Energy per inference (kWh/image)} = \frac{E_{\text{total}} \text{ (kWh)}}{N}$$

where $E_{\text{total}}$ is the total electrical energy measured during the inference run and
$N$ is the number of images processed in that run.

**How it is measured:** In this project, energy is estimated using **CodeCarbon**'s
`OfflineEmissionsTracker`, which estimates power draw from hardware sensors (CPU/GPU power
counters or Intel RAPL) and integrates over the measurement duration. The tracker is started
immediately before the inference loop and stopped immediately after, with warm-up performed
*before* starting the tracker.

**Units:**
- Total energy: kilowatt-hours (kWh)
- Per-image energy: kWh/image (or equivalently mWh/image, J/image)

Unit conversions:
- $1\,\text{kWh} = 3.6 \times 10^6\,\text{J} = 3.6 \times 10^9\,\text{mJ}$
- $1\,\text{kWh/image} = 3.6 \times 10^6\,\text{J/image}$

**Comparing FP32, FP16, INT8 PTQ, and INT8 QAT:** Lower latency generally translates to lower
energy consumption per inference, since the hardware runs for a shorter time. INT8 models can
consume significantly less energy per image than FP32 models on compatible hardware.

---

### 3.2 Energy Consumption per N Images

**What it measures:** The cumulative energy required to process a fixed number of images,
useful for estimating operational cost at scale (e.g., a clinic processing 1,000 patients
per day).

**Formula:**

$$E_N \text{ (kWh)} = \text{Energy per inference (kWh/image)} \times N$$

**Units:** kWh for $N$ images.

**Why it matters:** While per-image energy differences may seem negligible in isolation,
they compound significantly at deployment scale. A model that consumes 25% less energy per
inference saves substantial operational energy when running millions of inferences.

---

### 3.3 CO₂e per Inference

**What it measures:** The estimated carbon dioxide equivalent (CO₂e) greenhouse gas emissions
associated with processing one image, derived from the electricity consumed and the carbon
intensity of the electricity grid.

**Formula:**

$$\text{CO}_2\text{e per image (g)} = \text{Energy per image (kWh/image)} \times \text{Carbon Intensity (gCO}_2\text{e/kWh)}$$

**Units and dimensional analysis:**

$$\frac{\text{kWh}}{\text{image}} \times \frac{\text{gCO}_2\text{e}}{\text{kWh}} = \frac{\text{gCO}_2\text{e}}{\text{image}}$$

**Carbon intensity:** The carbon intensity of electricity (gCO₂e/kWh) varies by country,
region, and time of day, depending on the mix of power sources (coal, natural gas, nuclear,
renewables). For reference:

| Region | Typical Carbon Intensity |
|---|---|
| France (nuclear-dominant) | ~60 gCO₂e/kWh |
| Tunisia (`TUN` in CodeCarbon) | Varies; CodeCarbon uses IEA/Our World in Data values |
| Germany (mixed) | ~400 gCO₂e/kWh |
| Poland (coal-dominant) | ~750 gCO₂e/kWh |

> [!IMPORTANT]
> **The carbon intensity source must always be documented when reporting CO₂e.**
>
> CO₂e estimates are only meaningful if the following are clearly stated:
> - The measurement tool (e.g., CodeCarbon v2.x).
> - The carbon intensity value or database used (e.g., IEA 2023).
> - The `country_iso_code` parameter (e.g., `TUN` for Tunisia in this project).
> - The measurement mode (offline vs online lookup).
>
> Without this information, reported CO₂e values cannot be reproduced or compared across studies.

**How CodeCarbon computes CO₂e:**

CodeCarbon measures hardware power draw (via Intel RAPL, NVML, or fallback estimates),
integrates over time to obtain energy in kWh, and multiplies by a country-specific carbon
intensity factor:

$$\text{CO}_2\text{e (kg)} = E_{\text{total}} \text{ (kWh)} \times \text{CI (kgCO}_2\text{e/kWh)}$$

The result is returned in **kilograms (kg)**; multiply by 1000 to convert to grams (g).

---

### 3.4 CO₂e per N Images

**What it measures:** Total estimated CO₂e emissions to process a batch of $N$ images.

**Formula:**

$$\text{CO}_2\text{e}_N \text{ (g)} = \text{CO}_2\text{e per image (g/image)} \times N$$

**Units:** gCO₂e for $N$ images.

**Why it matters:** Scale amplifies differences. A reduction of $10\,\mu\text{g CO}_2\text{e}$
per image becomes a reduction of $10\,\text{g CO}_2\text{e}$ per million images — comparable
to driving approximately 70 metres in an average petrol car.

---

## Experimental Comparison

### Models

The primary models evaluated in this project are:

| # | Model | Precision | Method |
|---|---|---|---|
| 1 | ResNet18 FP32 | 32-bit float | Full-precision baseline |
| 2 | ResNet18 FP16 | 16-bit float | FP16 inference (half precision) |
| 3 | ResNet18 INT8 PTQ | 8-bit integer | Static Post-Training Quantization |
| 4 | ResNet18 INT8 QAT | 8-bit integer | Quantization-Aware Training |

The **FP32 model is the reference baseline**. All compressed variants are compared against it.
No compressed model is compared against another compressed model without also referencing the
FP32 baseline.

### Comparison Table

The table below defines the columns of the experimental results. Values will be filled after
experiments are completed.

| Model | Precision | Method | Accuracy | Macro-F1 | Sensitivity | Specificity | ROC-AUC | Model Size (MB) | Compression Ratio | GFLOPs | Latency (ms/img) | Throughput (img/s) | Memory (MB) | Energy/Inference (kWh/img) | CO₂e/Inference (gCO₂e/img) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet18 FP32 | FP32 | Baseline | — | — | — | — | — | — | 1.0x | — | — | — | — | — |
| ResNet18 FP16 | FP16 | FP16 Inference | — | — | — | — | — | — | — | — | — | — | — | — |
| ResNet18 INT8 PTQ | INT8 | Static PTQ | — | — | — | — | — | — | — | — | — | — | — | — |
| ResNet18 INT8 QAT | INT8 | QAT | — | — | — | — | — | — | — | — | — | — | — | — |

---

## Important Experimental Principles

The following principles must be observed to ensure valid, reproducible, and comparable
results across model variants:

### Data and Evaluation Consistency

- **Use the same test set for all models.** The test set is fixed and never used for training,
  calibration, or hyperparameter selection. It is partitioned once and applied identically to
  FP32, FP16, INT8 PTQ, and INT8 QAT models.
- **Do not change the evaluation dataset between FP32 and quantized models.** Any difference
  in the test set would confound the comparison.
- **Report per-class results** (Precision, Recall, F1 per class) wherever they provide
  additional clinical insight, particularly for underrepresented HAM10000 classes.
- **Use Macro-F1 as the primary performance metric** because HAM10000 has significant class imbalance.
- **Keep preprocessing and input resolution consistent** across all model variants. Differences
  in normalization, resizing, or augmentation invalidate comparisons.

### Latency and Throughput Measurement

- **Measure latency under the same hardware and software conditions.** Use the same physical
  machine, CPU/GPU, number of threads, and PyTorch version for all timing experiments.
- **Clearly specify batch size** when measuring latency and throughput. Latency values are
  only comparable at the same batch size.
- **Perform warm-up iterations** before timing to exclude JIT compilation, cache cold-start,
  and lazy initialization artefacts.
- **Repeat measurements** (e.g., 5 repetitions) and report mean and standard deviation to
  quantify timing variability.

### Distinguishing Theoretical from Empirical Metrics

- **Distinguish theoretical metrics such as GFLOPs from hardware measurements such as latency
  and energy.** GFLOPs characterise the architecture; latency and energy characterise the
  system (hardware + software + model).
- GFLOPs are fixed for a given architecture and input size; do not use GFLOPs as a proxy for
  latency without empirical verification.

### Hardware and Software Documentation

- **Document the hardware used for inference:** CPU model, number of cores, CPU frequency,
  RAM capacity. If GPU is used, specify the GPU model and VRAM.
- **Document the software framework and relevant versions:** Python version, PyTorch version,
  Torchvision version, operating system, and quantization backend (e.g., FBGEMM, QNNPACK).
- **Document the measurement methodology for energy consumption:** tool used (CodeCarbon),
  version, measurement interval (`measure_power_secs`), and any known limitations (e.g.,
  fallback power estimates when hardware sensors are unavailable).
- **Document the carbon intensity source** for CO₂e calculations: country ISO code, database
  version, and the specific carbon intensity value applied.

---

## Frugal AI Interpretation

### The Fundamental Objective

The objective of this project is **not** to maximize classification accuracy. The goal is to
rigorously study the **trade-off** between three competing dimensions:

$$\text{Predictive Quality} \longleftrightarrow \text{Computational Efficiency} \longleftrightarrow \text{Environmental Efficiency}$$

### The Compression Trade-Off

Compression techniques reduce numerical precision or model complexity, with the following
expected effects:

```
Compression (FP16 / INT8 PTQ / INT8 QAT)
    |
    v
  Lower numerical precision / fewer computational resources
    |
    v
  Lower model size, latency, memory usage, energy, and CO2e
    |
    v
  Potential degradation in predictive performance
```

The direction and magnitude of performance degradation depend on the compression technique,
the calibration data quality (for PTQ), the fine-tuning strategy (for QAT), and the
characteristics of the dataset. QAT generally recovers more accuracy than PTQ because the
model is trained with quantization noise during fine-tuning.

### Defining a "Successful" Compression

A compressed model can be considered **advantageous** if it achieves:

1. **Substantial reductions** in one or more of: model size, GFLOPs, latency, memory
   footprint, energy per inference, or CO₂e per inference, **and**
2. **Acceptable predictive performance**, meaning the Macro-F1, Sensitivity, and ROC-AUC
   remain within a clinically acceptable degradation threshold relative to the FP32 baseline.

What constitutes "acceptable degradation" is application-dependent. For a screening tool
designed to flag suspicious lesions for expert review, a Macro-F1 drop of 1% or less with a
3x compression ratio might be considered a favourable trade-off. For a system intended to
support direct diagnostic decisions, stricter thresholds apply.

### The Goal: Best Trade-Off, Not Smallest Model

The experiment should **not** be interpreted as a competition to find the smallest or fastest
model. It should identify the compression variant that achieves the **best balance** between:

- Retaining diagnostic value (Macro-F1, Sensitivity, ROC-AUC).
- Reducing deployment cost (Model Size, Latency, Memory).
- Reducing environmental cost (Energy, CO₂e).

This frugal perspective is particularly relevant for medical AI deployed in resource-limited
settings (e.g., low-bandwidth telemedicine, battery-powered dermoscopy devices, or hospital
systems with energy-efficiency mandates).

---

## Metric Selection Summary

The table below summarises which metrics are mandatory for this project and which are
optional or supplementary.

| Metric | Category | Mandatory / Optional | Notes |
|---|---|---|---|
| Accuracy | Predictive | **Mandatory** | Overall classification correctness |
| Macro-F1 | Predictive | **Mandatory** | Primary metric for imbalanced HAM10000 |
| Macro Sensitivity | Predictive | **Mandatory** | Critical for detecting rare malignancies |
| Macro Specificity | Predictive | **Mandatory** | Avoids over-triggering on benign tissue |
| Macro ROC-AUC | Predictive | **Mandatory** | Threshold-independent ranking quality |
| Macro Precision | Predictive | **Mandatory** | Controls false alarm rate |
| Per-class F1 | Predictive | Optional | Recommended for clinical analysis |
| Per-class Sensitivity | Predictive | Optional | Recommended for melanoma class |
| Weighted-F1 | Predictive | Optional | Supplementary dataset-level view |
| Number of Parameters | Computational | **Mandatory** | Architecture characterisation |
| Model Size (MB) | Computational | **Mandatory** | Storage and deployment feasibility |
| Compression Ratio | Computational | **Mandatory** | Normalised compression effectiveness |
| GFLOPs | Computational | **Mandatory** | Theoretical computational complexity |
| Latency (ms/image) | Computational | **Mandatory** | Real-world inference speed |
| Throughput (img/s) | Computational | **Mandatory** | Operational capacity |
| Memory Footprint | Computational | Optional | Required if edge RAM is a constraint |
| Energy/Inference (kWh/img) | Environmental | **Mandatory** | Per-image energy cost |
| CO₂e/Inference (g/img) | Environmental | **Mandatory** | Per-image carbon footprint estimate |
| Energy per 1,000 images | Environmental | Optional | Useful for scale reporting |
| CO₂e per 1,000 images | Environmental | Optional | Useful for scale reporting |
