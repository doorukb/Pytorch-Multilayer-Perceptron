from __future__ import annotations
import os
import _path_setup
import mlflow
import mlflow.pytorch
import torch.nn as nn
from _mlflow_setup import configure_tracking
from mlflow.tracking import MlflowClient
from torchmlp.data import create_surface_split_dataloaders
from torchmlp.tracking import REGISTERED_MODEL_NAME
from torchmlp.trainer import evaluate, resolve_device

BASE_SEED = 42
DEFAULT_N_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_VERSION = "latest"

# load the sweep winner from the MLflow Model Registry and evaluate on the test set

# prerequisite : run experiments/03_hyperparam_sweep.py first so torchmlp-mlp is registered

# this script is inference-only
# it loads the model by registry URI : models:/torchmlp-mlp/{version} not by local path or runs

# test split matches training : BASE_SEED=42, n_samples=1000, batch_size=32
# headline metric : test MSE (regression)

# override the registry version with MLFLOW_MODEL_VERSION (default: latest)

# run from repo root : python experiments/04_model_promotion.py

# MLflow 3+ file store : set MLFLOW_ALLOW_FILE_STORE=true before running, or use MLFLOW_TRACKING_URI=sqlite:///mlflow.db

def model_uri(name: str, version: str) -> str:
    return f"models:/{name}/{version}"

def resolve_model_version(name: str, version: str = DEFAULT_MODEL_VERSION) -> str:
    if version != "latest":
        return version

    versions = MlflowClient().search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for model {name!r}. Run experiments/03_hyperparam_sweep.py first.")

    latest = max(versions, key=lambda model_version: int(model_version.version))
    return str(latest.version)

def load_registered_model(version: str = DEFAULT_MODEL_VERSION) -> nn.Module:
    configure_tracking()
    resolved_version = resolve_model_version(REGISTERED_MODEL_NAME, version)
    uri = model_uri(REGISTERED_MODEL_NAME, resolved_version)
    return mlflow.pytorch.load_model(uri)

def evaluate_registered_model(model: nn.Module, *, seed: int = BASE_SEED) -> dict[str, float]:
    _, _, test_loader = create_surface_split_dataloaders(n=DEFAULT_N_SAMPLES, batch_size=DEFAULT_BATCH_SIZE, seed=seed)
    device = resolve_device(None)
    model.to(device)
    criterion = nn.MSELoss()
    return evaluate(model, test_loader, criterion, device, task="regression")

def main() -> None:
    version = os.environ.get("MLFLOW_MODEL_VERSION", DEFAULT_MODEL_VERSION)

    configure_tracking()
    resolved_version = resolve_model_version(REGISTERED_MODEL_NAME, version)
    uri = model_uri(REGISTERED_MODEL_NAME, resolved_version)
    source_run_id = MlflowClient().get_model_version(REGISTERED_MODEL_NAME, resolved_version).run_id

    print("Model promotion — load registered model and evaluate on test set")
    print(f"Registry model:  {REGISTERED_MODEL_NAME}")
    print(f"Version:         {resolved_version}")
    print(f"Registry URI:    {uri}")
    print(f"Source run ID:   {source_run_id}")
    print(f"Test split seed: {BASE_SEED}")
    print()

    model = mlflow.pytorch.load_model(uri)
    test_metrics = evaluate_registered_model(model)
    test_loss = test_metrics["loss"]

    print(f"Test loss (MSE): {test_loss:.6f}")
    print()
    print("Model loaded from MLflow Model Registry (not retrained from scratch).")

if __name__ == "__main__":
    main()