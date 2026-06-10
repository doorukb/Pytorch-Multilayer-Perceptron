import importlib

from torchmlp.config import TrainConfig
from torchmlp.tracking import REGISTERED_MODEL_NAME

sweep = importlib.import_module("03_hyperparam_sweep")
def test_build_grid_size():
    assert len(sweep.build_grid()) == 18


def test_grid_configs_skip_registry():
    assert all(config.registered_model_name is None for config in sweep.build_grid())


def test_winner_config_enables_registry():
    config = sweep.build_grid()[0]
    winner = sweep.winner_config(config)
    assert winner.registered_model_name == REGISTERED_MODEL_NAME
    assert config.registered_model_name is None

def test_select_best_picks_lowest_val_loss():
    configs = sweep.build_grid()
    results = [
        sweep.TrialResult(index=0, config=configs[0], history={}, best_val_loss=0.5, run_id="a"),
        sweep.TrialResult(index=1, config=configs[1], history={}, best_val_loss=0.2, run_id="b"),
        sweep.TrialResult(index=2, config=configs[2], history={}, best_val_loss=0.3, run_id="c"),
    ]
    best = sweep.select_best(results)
    assert best.run_id == "b"
    assert best.best_val_loss == 0.2

def test_select_best_breaks_ties_by_index():
    config = TrainConfig()
    results = [
        sweep.TrialResult(index=1, config=config, history={}, best_val_loss=0.2, run_id="later"),
        sweep.TrialResult(index=0, config=config, history={}, best_val_loss=0.2, run_id="earlier"),
    ]
    best = sweep.select_best(results)
    assert best.run_id == "earlier"

def test_run_name_for_is_readable():
    config = TrainConfig(hidden_sizes=[5, 5], learning_rate=1e-3, dropout=0.1)
    name = sweep.run_name_for(config)
    assert name == "arch=5-5_lr=0.001_do=0.1"