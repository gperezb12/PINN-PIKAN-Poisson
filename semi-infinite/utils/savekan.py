# ==== utils_kan_io.py ====
import os
import json
import torch
from typing import Dict, Any

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)

def save_kan_checkpoint(model_u: torch.nn.Module, model_k: torch.nn.Module, cfg_u: Dict[str, Any], cfg_k: Dict[str, Any],
                        losses: Dict[str, list], meta: Dict[str, Any], out_dir: str) -> str:
    """Save KAN model checkpoint with configuration and training history.
    
    Args:
        model_u: Trained model for solution u
        model_k: Trained model for diffusion coefficient k
        cfg_u: Configuration dict for model_u
        cfg_k: Configuration dict for model_k
        losses: Dictionary of loss histories
        meta: Metadata dictionary
        out_dir: Output directory path
    
    Returns:
        Path to saved checkpoint file
    """
    ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, "checkpoint_kan.pt")
    torch.save({
        "model_u_state": model_u.state_dict(),
        "model_k_state": model_k.state_dict(),
        "cfg_u": cfg_u,                # e.g., {"width":[2,4,4,1], "grid":..., "k":...}
        "cfg_k": cfg_k,
        "losses": losses,              # {"adam": [...], "lbfgs": [...]}
        "meta": meta                   # {"training_time": "...", "seed": 32, ...}
    }, ckpt_path)
    # (opcional) exporta pérdidas y meta en JSONs legibles
    with open(os.path.join(out_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return ckpt_path

def load_kan_checkpoint(kan_cls: type, ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    """Load KAN models from checkpoint.
    
    Args:
        kan_cls: KAN class constructor
        ckpt_path: Path to checkpoint file
        device: Device to load models on
    
    Returns:
        Tuple of (model_u, model_k, extras) where extras contains losses and metadata
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_u = ckpt["cfg_u"]
    cfg_k = ckpt["cfg_k"]

    # Importante: crea los modelos con EXACTA config usada al entrenar
    model_u = kan_cls(**cfg_u).to(device)
    model_k = kan_cls(**cfg_k).to(device)

    model_u.load_state_dict(ckpt["model_u_state"], strict=True)
    model_k.load_state_dict(ckpt["model_k_state"], strict=True)
    model_u.eval()
    model_k.eval()

    extras = {
        "losses": ckpt.get("losses", {}),
        "meta": ckpt.get("meta", {})
    }
    return model_u, model_k, extras
