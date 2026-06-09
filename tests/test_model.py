import math
import pytest
import torch
from torchmlp.model import MLP

def test_mlp_linear_shapes():
    model = MLP([2, 5, 10, 1])
    expected = [(2, 5), (5, 10), (10, 1)]
    assert len(model.linears) == len(expected)
    for linear, (in_dim, out_dim) in zip(model.linears, expected, strict=True):
        assert linear.in_features == in_dim
        assert linear.out_features == out_dim

def test_mlp_forward_output_shape():
    model = MLP([2, 5, 1])
    x = torch.randn(10, 2)
    out = model(x)
    assert out.shape == (10, 1)

def test_mlp_parameters_registered():
    layer_sizes = [2, 5, 10, 1]
    model = MLP(layer_sizes)
    n_layers = len(layer_sizes) - 1
    assert len(list(model.parameters())) == 2 * n_layers
    state = model.state_dict()
    for i in range(n_layers):
        assert f"linears.{i}.weight" in state
        assert f"linears.{i}.bias" in state

def test_mlp_xavier_init_stats():
    torch.manual_seed(0)
    fan_in, out_dim = 100, 50
    model = MLP([fan_in, out_dim])
    weight = model.linears[0].weight.detach()
    bias = model.linears[0].bias.detach()
    expected_std = math.sqrt(1.0 / fan_in)
    assert 0.8 * expected_std < weight.std().item() < 1.2 * expected_std
    assert abs(weight.mean().item()) < 0.05
    assert torch.allclose(bias, torch.zeros_like(bias))

def test_mlp_output_layer_is_linear():
    torch.manual_seed(0)
    model = MLP([2, 4, 1])
    x = torch.tensor([[10.0, -10.0], [-5.0, 5.0]])

    with torch.no_grad():
        hidden = model.activation(model.linears[0](x))
        out = model.linears[1](hidden)

    assert torch.any(out < 0) or torch.any(out > 1)

def test_mlp_activation_changes_hidden():
    torch.manual_seed(42)
    x = torch.randn(8, 2)
    model_sigmoid = MLP([2, 6, 1], activation="sigmoid")
    torch.manual_seed(42)
    model_relu = MLP([2, 6, 1], activation="relu")

    with torch.no_grad():
        hidden_sigmoid = model_sigmoid.activation(model_sigmoid.linears[0](x))
        hidden_relu = model_relu.activation(model_relu.linears[0](x))

    assert not torch.allclose(hidden_sigmoid, hidden_relu)
    assert torch.all(hidden_relu >= 0)
    assert torch.any(hidden_sigmoid != hidden_relu)


@pytest.mark.parametrize("layer_sizes", [[2], []])
def test_invalid_layer_sizes_raises(layer_sizes):
    with pytest.raises(ValueError, match="layer_sizes must contain at least"):
        MLP(layer_sizes)