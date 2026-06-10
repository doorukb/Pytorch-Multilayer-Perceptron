import importlib
import mlflow
import torch
from torchmlp.model import MLP
from torchmlp.tracking import REGISTERED_MODEL_NAME, log_pytorch_model

promotion = importlib.import_module("04_model_promotion")

def test_model_uri_uses_registry_scheme():
    assert promotion.model_uri(REGISTERED_MODEL_NAME, "latest") == "models:/torchmlp-mlp/latest"

def test_resolve_model_version_passes_through_explicit_version():
    assert promotion.resolve_model_version(REGISTERED_MODEL_NAME, "2") == "2"

def test_load_registered_model_round_trip(tmp_path, monkeypatch):
    db_path = (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_experiment("test-promotion")

    model = MLP([2, 5, 2])
    with mlflow.start_run():
        log_pytorch_model(model, registered_model_name=REGISTERED_MODEL_NAME)

    loaded = promotion.load_registered_model("1")
    output = loaded(torch.randn(4, 2))
    assert output.shape == (4, 2)

def test_resolve_model_version_latest_requires_registry(tmp_path, monkeypatch):
    db_path = (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")

    try:
        promotion.resolve_model_version("missing-model", "latest")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "No registered versions found" in str(exc)