import pytest
from torchmlp.config import TrainConfig

def test_layer_sizes_property():
    config = TrainConfig(hidden_sizes=[5, 10])
    assert config.layer_sizes == [2, 5, 10, 1]

def test_to_dict_serializable():
    config = TrainConfig(learning_rate=0.01, epochs=50, dropout=0.1)
    params = config.to_dict()
    assert params["learning_rate"] == 0.01
    assert params["epochs"] == 50
    assert params["dropout"] == 0.1
    assert "hidden_sizes" in params
    assert "seed" in params

def test_validate_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout must be"):
        TrainConfig(dropout=1.5).validate()

def test_validate_classification_requires_two_outputs():
    with pytest.raises(ValueError, match="output_dim"):
        TrainConfig(task="classification", output_dim=1).validate()

def test_to_mlflow_params_serializes_lists_and_device():
    config = TrainConfig(hidden_sizes=[5, 10], device=None)
    params = config.to_mlflow_params()

    assert params["hidden_sizes"] == "5,10"
    assert params["layer_sizes"] == "2,5,10,1"
    assert params["device"] == "auto"
    assert params["registered_model_name"] == "torchmlp-mlp"
    assert all(isinstance(value, str) for value in params.values())

def test_to_mlflow_params_serializes_none_registered_model_name():
    config = TrainConfig(registered_model_name=None)
    assert config.to_mlflow_params()["registered_model_name"] == "none"