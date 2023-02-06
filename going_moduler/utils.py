
"""
File containing various utility functions for Pytorch model training.
"""

import torch
from pathlib import Path
import os


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    """Saves a Pytorch model to a training directory
    """
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok = True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with .pt or .pth"
        
    model_save_path = os.path.join(target_dir_path, model_name)
    
    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj = model.state_dict(), f = model_save_path)
