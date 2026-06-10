# PyTorch MLP

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn
- MLflow
- Matplotlib
- pytest

A PyTorch reimplementation of the same multilayer perceptron family as the [NumPy Gradient Descent MLP](https://github.com/doorukb/Multilayer-perceptron). The project uses autograd for training, logs experiments to MLflow, runs a fixed hyperparameter grid with strict train/validation/test discipline, registers the sweep winner in the Model Registry, and loads that model by registry URI for inference-only promotion. Analytical gradients are checked against the NumPy reference in `experiments/01_parity_check.py`.

## What it does

The network learns from the same synthetic surface as the NumPy project: Z = X^2 - Y^2 + 1.2 + noise, where X and Y are drawn uniformly from [-1, 1] and noise is Gaussian with mean 0 and standard deviation 0.5. Inputs are 2D (X, Y).

For the hyperparameter sweep, the continuous target Z is binarized into class labels: label = 1 when Z > 1.2, else 0. Selection uses validation AUC; F1 is reported alongside. The baseline run (`experiments/02_baseline_run.py`) keeps the original regression formulation (MSE on Z) as a sanity check on the shared surface.

Gradient parity (`experiments/01_parity_check.py`) compares PyTorch autograd to the NumPy backprop implementation on architecture [2, 4, 1] for sigmoid, tanh, and ReLU. Clone the NumPy repo as a sibling `Multilayer-Perceptron` directory or set `NUMPY_MLP_SRC` to its `src/` path.

## Relationship to the NumPy MLP

[Multilayer-perceptron](https://github.com/doorukb/Multilayer-perceptron) implements forward pass, backpropagation, MSE loss, and gradient descent entirely in NumPy. This repo reuses the same surface generator and split logic, validates that PyTorch gradients match the hand-derived NumPy gradients, then extends the workflow with MLflow tracking, artifact logging, a hyperparameter sweep, and model registry promotion. The NumPy README links here under Roadmap.

## RESULTS

Gradient parity (experiment 01, architecture [2, 4, 1], seed 42, atol 1e-4):

    sigmoid   PASS
    tanh      PASS
    ReLU      PASS

Run locally: `python experiments/01_parity_check.py`

Baseline tracked run (experiment 02, regression MSE, architecture [2, 5, 1], lr=0.001, 20 epochs, seed 42):

    final train MSE:  0.4665
    final val MSE:    0.5162
    test MSE:         0.3721

Run locally: `python experiments/02_baseline_run.py`

Hyperparameter sweep (experiment 03, 3 architectures x 3 learning rates x 2 dropout values, 20 epochs, seed 42):

Data split: 800 train, 100 validation, 100 test (from n=1000). Labels: Z > 1.2. The test set was not accessed until after the best configuration was selected by highest validation AUC.

    arch              lr       dropout    val_auc    best_val_auc    val_f1     best_val_f1
    [2, 5, 5, 2]      0.01     0.0        0.5500     0.5500          0.6250     0.6250
    [2, 5, 5, 2]      0.01     0.1        0.5488     0.5496          0.6316     0.6667
    [2, 5, 5, 2]      0.001    0.0        0.5484     0.5488          0.4186     0.4337
    [2, 5, 5, 2]      0.001    0.1        0.5484     0.5488          0.4318     0.4337
    [2, 5, 5, 2]      0.0001   0.0        0.5488     0.5488          0.2857     0.2899
    [2, 5, 5, 2]      0.0001   0.1        0.5488     0.5488          0.2857     0.2899
    [2, 10, 2]        0.001    0.0        0.4914     0.4962          0.6395     0.6395
    [2, 10, 2]        0.001    0.1        0.4914     0.4962          0.6395     0.6395
    [2, 10, 2]        0.0001   0.0        0.4962     0.4962          0.6395     0.6395
    [2, 10, 2]        0.0001   0.1        0.4962     0.4962          0.6395     0.6395
    [2, 10, 2]        0.01     0.0        0.4685     0.4930          0.5505     0.6818
    [2, 10, 2]        0.01     0.1        0.4689     0.4930          0.5505     0.6866
    [2, 5, 2]         0.01     0.0        0.4617     0.4629          0.4950     0.5441
    [2, 5, 2]         0.01     0.1        0.4580     0.4621          0.4854     0.5441
    [2, 5, 2]         0.001    0.0        0.4597     0.4597          0.5082     0.6395
    [2, 5, 2]         0.001    0.1        0.4589     0.4589          0.5082     0.6395
    [2, 5, 2]         0.0001   0.0        0.4568     0.4580          0.6207     0.6395
    [2, 5, 2]         0.0001   0.1        0.4568     0.4580          0.6207     0.6395

Selected configuration (highest validation AUC):

    architecture:  [2, 5, 5, 2]
    learning rate: 0.01
    dropout:       0.0
    best val AUC:  0.5500
    test AUC:      0.5147
    test F1:       0.6560

The two-layer hidden network [2, 5, 5, 2] at lr=0.01 edges out shallower architectures on validation AUC. AUC values cluster near 0.5 on this noisy binarized surface, which is expected when the label boundary cuts through high-variance regions of Z.

Run locally: `python experiments/03_hyperparam_sweep.py`

Model promotion (experiment 04):

Loads the sweep winner from the MLflow Model Registry at `models:/torchmlp-mlp/{version}` (not a local path or `runs:/` URI), prints the source run ID for lineage, and evaluates on the held-out test split without retraining.

    registry URI:  models:/torchmlp-mlp/latest
    test AUC:      0.5147
    test F1:       0.6560

Run locally: `python experiments/04_model_promotion.py`

## View in MLflow

From the repo root:

    $env:MLFLOW_ALLOW_FILE_STORE = "true"
    python experiments/02_baseline_run.py
    python experiments/03_hyperparam_sweep.py
    python experiments/04_model_promotion.py
    mlflow ui

Open http://127.0.0.1:5000 in a browser.

Experiments:

    torchmlp-surface           baseline regression run (experiment 02)
    torchmlp-hyperparam-sweep  18 grid trials plus sweep-winner run (experiment 03)

Model Registry tab: registered model name `torchmlp-mlp`. Each sweep-winner retrain creates a new version.

Optional: pin a specific registry version before promotion:

    $env:MLFLOW_MODEL_VERSION = "2"
    python experiments/04_model_promotion.py

MLflow 3+ blocks the default file store unless you set `MLFLOW_ALLOW_FILE_STORE=true` or point tracking at SQLite, for example `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`.

Per-run artifacts include hyperparameters, per-epoch train/val metrics, a learning-curve PNG, and the serialized PyTorch model.

## Installation

- Python 3.10 or later is required.

- Clone the repository and install dependencies:

    git clone https://github.com/doorukb/pytorch-multilayer-perceptron.git
    cd pytorch-multilayer-perceptron
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

- Dependencies (requirements.txt):

    torch>=2.0
    numpy>=1.24
    scikit-learn>=1.3
    mlflow>=2.0
    matplotlib>=3.7
    pytest>=7.0

- Gradient parity against the NumPy reference (optional):

    git clone https://github.com/doorukb/Multilayer-perceptron.git ../Multilayer-Perceptron
    python experiments/01_parity_check.py

- Running experiments from repo root:

    python experiments/02_baseline_run.py
    python experiments/03_hyperparam_sweep.py
    python experiments/04_model_promotion.py

## Testing

To run all tests from the project root:

    pytest tests/ -v
