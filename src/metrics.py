import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

class ErrorTracker:
    """
    Running accumulator for physical error metrics across batches.
    """
    def __init__(self):
        self.mae_sum = 0.0
        self.mse_sum = 0.0
        self.mbe_sum = 0.0
        self.total_points = 0
        
        self.active_mae_sum = 0.0
        self.active_points = 0
        
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Updates running counters. Expects tensors transformed back to real units.
        """
        self.total_points += y_true.numel()
        
        diff = y_pred - y_true
        self.mae_sum += torch.abs(diff).sum().item()
        self.mse_sum += torch.pow(diff, 2).sum().item()
        self.mbe_sum += diff.sum().item()
        
        mask = y_true > 0.0
        if mask.sum() > 0:
            self.active_points += mask.sum().item()
            self.active_mae_sum += torch.abs(diff[mask]).sum().item()
            
    def compute(self) -> dict:
        """
        Computes and returns the final aggregated metrics.
        """
        return {
            "real_mae": self.mae_sum / self.total_points if self.total_points > 0 else 0.0,
            "real_rmse": np.sqrt(self.mse_sum / self.total_points) if self.total_points > 0 else 0.0,
            "real_mbe": self.mbe_sum / self.total_points if self.total_points > 0 else 0.0,
            "real_active_mae": self.active_mae_sum / self.active_points if self.active_points > 0 else 0.0
        }

def calculate_metrics(raw_metrics: dict, nominal_output: float) -> dict:
    """
    Enriches raw physical metrics with percentage bounds based on nominal capacity.
    """
    metrics = raw_metrics.copy()
    metrics["mae_pct"] = (metrics["real_mae"] / nominal_output) * 100
    metrics["active_mae_pct"] = (metrics["real_active_mae"] / nominal_output) * 100
    metrics["mbe_pct"] = (metrics["real_mbe"] / nominal_output) * 100
    return metrics

def log_epoch_metrics(writer: SummaryWriter, global_epoch: int, train_loss: float, val_loss: float, t_metrics: dict, v_metrics: dict, current_lr: float) -> None:
    """
    Handles SummaryWriter logs for training and validation loops.
    """
    writer.add_scalar('Global/Learning_Rate', current_lr, global_epoch)
    writer.add_scalars('Global/Scaled_Loss', {'Train': train_loss, 'Val': val_loss}, global_epoch)
    writer.add_scalars('Global/Real_Error_All', {'Train': t_metrics["real_mae"], 'Val': v_metrics["real_mae"]}, global_epoch)
    writer.add_scalars('Global/Real_Error_Active', {'Train': t_metrics["real_active_mae"], 'Val': v_metrics["real_active_mae"]}, global_epoch)
    writer.add_scalars('Global/Real_RMSE', {'Train': t_metrics["real_rmse"], 'Val': v_metrics["real_rmse"]}, global_epoch)
    writer.add_scalars('Global/Real_MBE', {'Train': t_metrics["real_mbe"], 'Val': v_metrics["real_mbe"]}, global_epoch)

def log_test_metrics(writer: SummaryWriter, global_epoch: int, test_loss: float, metrics: dict, args: argparse.Namespace) -> None:
    """
    Handles SummaryWriter logs for final test execution.
    """
    writer.add_scalar('Test/Scaled_Loss', test_loss, global_epoch)
    writer.add_scalar('Test/Real_Error_All', metrics['real_mae'], global_epoch)
    writer.add_scalar('Test/Real_Error_Active', metrics['real_active_mae'], global_epoch)
    writer.add_scalar('Test/Real_RMSE', metrics['real_rmse'], global_epoch)
    
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    writer.add_hparams(hparams, {
        "hparams/Test_Real_MAE_All": metrics['real_mae'],
        "hparams/Test_MAE_All_Pct": metrics['mae_pct'],
        "hparams/Test_Real_RMSE": metrics['real_rmse']
    })