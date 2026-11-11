# Static Quantization-Aware Training (QAT)

Ce dossier implémente la **quantification statique consciente de l'entraînement (QAT)** en INT8 pour le ResNet-18 fine-tuné sur HAM10000, selon la même structure et les mêmes principes que le dossier `quantization/ptq/weight_only/`.

## Différence clé avec le PTQ (Post-Training Quantization)

- **PTQ (weight-only)** : Compresse uniquement les poids en INT8 après coup, sans ré-entraînement. L'inférence reste en FP32.
- **Static QAT** :
  - Fusionne les modules (`Conv + BN + ReLU`) via le ResNet-18 quantifiable officiel de torchvision.
  - Insère des observateurs de fausse quantification (*fake-quantization*) à la fois pour les **poids (INT8)** et les **activations (INT8)**.
  - Ré-entraîne le modèle pendant 1 ou plusieurs époques pour adapter les poids à la perte de précision due à la quantification.
  - Convertit le modèle en un vrai graphe INT8 prêt pour l'inférence accélérée sur CPU (`fbgemm`).

## Structure du dossier

```text
quantization/static_quantization_aware_training/
├── quantize.py        # Helpers: création QAT, fusion, conversion, inspection des poids
├── static_qat.py      # Script principal: entraînement QAT + évaluation validation + export CSV
├── plot_results.py    # Génération des graphiques (PNG 300 dpi & PDF)
├── README.md          # Documentation
└── results/           # Résultats générés automatiquement
    ├── resnet18_qat_int8.pt      # Modèle quantifié INT8
    ├── file_sizes.csv            # Comparaison taille mémoire FP32 vs INT8
    ├── validation_results.csv    # Métriques (Accuracy, Macro F1, Recall, etc.)
    ├── weight_errors.csv         # Erreurs relatives de poids couche par couche
    └── plots/                    # Graphiques générés par plot_results.py
```

## Utilisation

Depuis la racine du projet (`FrugalAI`) :

```powershell
# 1. Lancer le Static QAT (par défaut: 2 époques de calibration, ou 1 époque avec --epochs 1)
python quantization/static_quantization_aware_training/static_qat.py --epochs 1

# 2. Générer les graphiques comparatifs
python quantization/static_quantization_aware_training/plot_results.py
```

