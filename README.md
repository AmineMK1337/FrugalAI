# FrugalAI

This project is a compact proof-of-concept for building a lightweight, energy-aware image-classification pipeline. The goal is to explore how a pretrained CNN can be trained and then compressed with quantization to reduce model size and inference cost while keeping an eye on energy consumption and carbon footprint.

## What was done

The workspace contains a small end-to-end experiment around ResNet18 on a synthetic image dataset:

- A dummy dataset generator creates train/validation/test folders with random RGB images.
- A baseline model is trained using a pretrained ResNet18 backbone.
- The trained model is evaluated and then compared against an INT8 quantized version.
- Energy and carbon estimates are measured with CodeCarbon.
- Results are exported as CSV files for comparison.

This is intended as a practical frugal-AI experiment rather than a production dataset or final production model.

## Project structure

- `make_dummy_dataset.py`  
  Generates a small synthetic dataset under `dataset/` so the pipeline can run without external data.

- `train_baseline.py`  
  Fine-tunes a pretrained `ResNet18` model on the generated data, saves the best checkpoint as `resnet18_baseline.pth`, and keeps the validation set for model selection.

- `quantization_experiment.py`  
  Runs the main compression benchmark. It compares FP32 vs INT8 quantized inference across:
  - classification metrics
  - model size and compression ratio
  - latency / throughput
  - estimated energy and CO2 emissions
  - confusion matrices

- `test_codecarbon.py`  
  Verifies the CodeCarbon tracker on a synthetic inference run to estimate energy use and carbon output.

- `requirements.txt`  
  Lists the Python dependencies used by the project.

- `dataset/`  
  Contains generated train/validation/test folders used by the training and evaluation scripts.

## Model and experiment flow

1. Generate a synthetic dataset.
2. Train a ResNet18 baseline.
3. Run post-training quantization (INT8) on the model.
4. Compare FP32 and quantized performance.
5. Save CSV reports and confusion matrices.

The experiment is designed to answer questions such as:

- How much smaller is the quantized model?
- Does accuracy drop significantly?
- What is the latency improvement?
- How much energy and carbon can be saved?

## Notes about realism

The generated dataset is intentionally random noise and not a meaningful real-world classification dataset. That is by design for a lightweight pipeline test.

The scripts are therefore useful as a laboratory setup to validate the workflow, benchmarking logic, and computational cost analysis. They are not a substitute for a properly labeled dataset and real-world validation.

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Typical workflow

```bash
python make_dummy_dataset.py
python train_baseline.py
python quantization_experiment.py
python test_codecarbon.py
```

## Outputs

Depending on execution, the project can generate files such as:

- `resnet18_baseline.pth`
- `results_quantization.csv`
- `confusion_matrix_fp32.csv`
- `confusion_matrix_int8.csv`
- `codecarbon_emissions.csv`

## Summary

This repository demonstrates a small frugal-AI workflow: train a strong baseline, compress it with quantization, and measure the tradeoff between model efficiency and predictive performance. The project is useful as a teaching tool, benchmarking starter, and quick experimentation environment for low-resource AI deployment.
