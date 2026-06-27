from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import optuna
import torch
import yaml

from utils.points_generator import generate_collocation_points, generate_interior_data
from utils.stats import relative_error_analytic
from utils.training import train_inverse_pinn_mixed
from main import MLP_u, MLP_k, get_activation, init_weights, set_seed

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yml"
RESULTS_DIR = ROOT / "results_optuna"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _as_list(value: Any, fallback: List[Any]) -> List[Any]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return value
    return [value]


def _as_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected int-like value, got {value!r}") from exc


def _as_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float-like value, got {value!r}") from exc


def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "cpu")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def suggest_hyperparams(trial: optuna.Trial, opt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    hidden_layers_min = _as_int(opt_cfg.get("hidden_layers_min"), 2)
    hidden_layers_max = _as_int(opt_cfg.get("hidden_layers_max"), 8)
    hidden_layers = trial.suggest_int("hidden_layers", hidden_layers_min, hidden_layers_max)

    hidden_units_choices = _as_list(opt_cfg.get("hidden_units"), [16, 32, 64])
    hidden_units = trial.suggest_categorical("hidden_units", hidden_units_choices)

    activations = _as_list(opt_cfg.get("activations"), ["tanh", "relu"])
    activation = trial.suggest_categorical("activation", activations)

    lr_adam = trial.suggest_float(
        "lr_adam",
        _as_float(opt_cfg.get("lr_adam_min"), 1.0e-5),
        _as_float(opt_cfg.get("lr_adam_max"), 1.0e-3),
        log=True,
    )
    lr_lbfgs = trial.suggest_float(
        "lr_lbfgs",
        _as_float(opt_cfg.get("lr_lbfgs_min"), 1.0e-2),
        _as_float(opt_cfg.get("lr_lbfgs_max"), 1.0),
        log=True,
    )
    lambda_data = trial.suggest_float(
        "lambda_data",
        _as_float(opt_cfg.get("lambda_data_min"), 1.0),
        _as_float(opt_cfg.get("lambda_data_max"), 1.0e3),
        log=True,
    )
    lambda_pde = trial.suggest_float(
        "lambda_pde",
        _as_float(opt_cfg.get("lambda_pde_min"), 1.0e-1),
        _as_float(opt_cfg.get("lambda_pde_max"), 1.0e1),
        log=True,
    )
    lambda_l1 = trial.suggest_float(
        "lambda_l1",
        _as_float(opt_cfg.get("lambda_l1_min"), 1.0e-9),
        _as_float(opt_cfg.get("lambda_l1_max"), 1.0e-6),
        log=True,
    )
    lambda_l2 = trial.suggest_float(
        "lambda_l2",
        _as_float(opt_cfg.get("lambda_l2_min"), 1.0e-8),
        _as_float(opt_cfg.get("lambda_l2_max"), 1.0e-4),
        log=True,
    )

    return {
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
        "activation": activation,
        "lr_adam": lr_adam,
        "lr_lbfgs": lr_lbfgs,
        "lambda_data": lambda_data,
        "lambda_pde": lambda_pde,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
    }


def objective(trial: optuna.Trial) -> float:
    cfg = load_config()
    opt_cfg = cfg.get("optuna", {})

    device = _device_from_cfg(cfg)
    base_seed = _as_int(cfg.get("seed"), 0)

    # Make data deterministic across trials.
    set_seed(base_seed)
    n_interior = _as_int(opt_cfg.get("N_interior"), _as_int(cfg["data"]["N_interior"], 1000))
    n_boundary = _as_int(opt_cfg.get("N_boundary"), _as_int(cfg["data"]["N_boundary"], 200))
    n_data = _as_int(opt_cfg.get("N_data"), _as_int(cfg["data"]["N_data"], 1000))

    X_int, X_bnd = generate_collocation_points(
        N_interior=n_interior,
        N_boundary=n_boundary,
        device=str(device),
    )
    X_data, u_data, k_data = generate_interior_data(
        N_data=n_data,
        device=str(device),
    )

    # Re-seed for model initialization to allow variation across trials.
    set_seed(base_seed + trial.number)

    params = suggest_hyperparams(trial, opt_cfg)
    activation = get_activation(params["activation"])

    model_u = MLP_u(2, 1, params["hidden_layers"], params["hidden_units"], activation).to(device)
    model_k = MLP_k(2, 1, params["hidden_layers"], params["hidden_units"], activation).to(device)
    model_u.apply(init_weights)
    model_k.apply(init_weights)

    adam_epochs = _as_int(opt_cfg.get("adam_epochs"), _as_int(cfg["training"]["adam_epochs"], 1000))
    lbfgs_iterations = _as_int(opt_cfg.get("lbfgs_iterations"), _as_int(cfg["training"]["lbfgs_iterations"], 200))
    plot_every = _as_int(opt_cfg.get("plot_every"), max(adam_epochs, 1))
    plot_every_lbfgs = _as_int(opt_cfg.get("plot_every_lbfgs"), max(lbfgs_iterations, 1))

    train_inverse_pinn_mixed(
        model_u,
        model_k,
        X_int,
        X_bnd,
        X_data,
        u_data,
        k_data,
        adam_epochs=adam_epochs,
        lbfgs_iterations=lbfgs_iterations,
        lr_adam=params["lr_adam"],
        lr_lbfgs=params["lr_lbfgs"],
        lambda_bc=_as_float(cfg["training"].get("lambda_bc"), 1.0),
        lambda_pde=params["lambda_pde"],
        lambda_data=params["lambda_data"],
        lambda_l1=params["lambda_l1"],
        lambda_l2=params["lambda_l2"],
        enable_plots=False,
        plot_every=plot_every,
        plot_every_lbfgs=plot_every_lbfgs,
    )

    eval_points = _as_int(opt_cfg.get("eval_points"), 60)
    error = relative_error_analytic(model_u, model_k, n_points=eval_points, device=str(device))

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(error)


def main() -> None:
    cfg = load_config()
    opt_cfg = cfg.get("optuna", {})

    sampler_seed = _as_int(opt_cfg.get("sampler_seed"), _as_int(cfg.get("seed"), 0))
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=_as_int(opt_cfg.get("n_startup_trials"), 5),
        n_warmup_steps=_as_int(opt_cfg.get("n_warmup_steps"), 0),
    )

    study_name = opt_cfg.get("study_name", "pinn_optuna")
    storage = opt_cfg.get("storage")
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    n_trials = _as_int(opt_cfg.get("n_trials"), 20)
    timeout = opt_cfg.get("timeout")

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
    }
    with (RESULTS_DIR / "best_params.json").open("w") as f:
        json.dump(best_summary, f, indent=2)

    print("Best objective value:", study.best_value)
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
