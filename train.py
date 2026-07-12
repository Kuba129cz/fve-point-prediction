import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
import os
from datetime import datetime
import json
import numpy as np

from src.data_splitting import Splitter
from src.models import model_attention
from src.trainer import load_dataset, run_cross_validation, evaluate_test_set

def get_parser() -> argparse.ArgumentParser:
    """
    Creates and returns the argument parser with all hyperparameters.
    """
    parser = argparse.ArgumentParser()
    
    # --- General Training ---
    parser.add_argument("--seed", default=6497, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=28, type=int)
    parser.add_argument("--learning_rate", default=0.000635, type=float)
    parser.add_argument("--weight_decay", default=0.000841, type=float)
    parser.add_argument("--eta_min", default=1e-6, type=float)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--patience', type=int, default=6)
    
    # --- Paths and Names ---
    parser.add_argument("--exp_name", default="openmeteo_15min_train_expand_wind", type=str)
    parser.add_argument("--model_name", default="openmeteo_15min_train_expand_wind", type=str)
    parser.add_argument("--dataset_path", default="data/openmeteo_15min_train_expand_wind.csv", type=str)
    
    # --- Model Architecture ---
    parser.add_argument("--past_hidden_size", default=16, type=int)
    parser.add_argument("--past_cnn_filters", default=16, type=int)
    parser.add_argument("--past_dropout", default=0.44, type=float)
    parser.add_argument("--past_kernel", default=7, type=int)
    
    parser.add_argument("--future_hidden_size", default=32, type=int)
    parser.add_argument("--future_cnn_filters", default=32, type=int)
    parser.add_argument("--future_kernel_L0", default=5, type=int)
    parser.add_argument("--future_kernel_L1", default=3, type=int)
    parser.add_argument("--future_dropout", default=0.086, type=float)
    
    parser.add_argument("--attention_dim", default=128, type=int)
    parser.add_argument("--decoder_dropout", default=0.071, type=float)
    
    # --- Data & Windows ---
    parser.add_argument("--nominal_output", default=1293, type=int)
    parser.add_argument("--target_col", default="energy", type=str)
    parser.add_argument("--lookback", default=24, type=int)
    parser.add_argument("--horizon", default=24, type=int)
    parser.add_argument("--test_ratio", default=0.15, type=float)
    parser.add_argument("--train_ratio", default=0.80, type=float)
    
    # --- Strategy ---
    parser.add_argument("--strategy", default="simple", choices=["simple", "expanding", "rolling"])
    parser.add_argument("--initial_train_size", default=10000, type=int)
    parser.add_argument("--step_size", default=5000, type=int)
    
    parser.add_argument("--lookback_cols", nargs="+", 
        default=[
            "energy", "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
            "shortwave_radiation_0", "shortwave_radiation_15", "shortwave_radiation_30", "shortwave_radiation_45",
            "diffuse_radiation_0", "diffuse_radiation_15", "diffuse_radiation_30", "diffuse_radiation_45",
            "cloud_cover", 'temperature_2m', 'relative_humidity_2m_0', 'wind_u', 'wind_v'
        ]
    )
    
    parser.add_argument("--horizon_cols", nargs="+", 
        default=[
            "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
            "shortwave_radiation_0", "shortwave_radiation_15", "shortwave_radiation_30", "shortwave_radiation_45",
            "diffuse_radiation_0", "diffuse_radiation_15", "diffuse_radiation_30", "diffuse_radiation_45",
            'temperature_2m', 'cloud_cover', 'relative_humidity_2m_0', 'wind_u', 'wind_v'
        ]
    )
    return parser

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_model(model: torch.nn.Module, args: argparse.Namespace, path_dir: str) -> None:
    os.makedirs(path_dir, exist_ok=True) 
    torch.save(model.state_dict(), f"{path_dir}/model.pth")
    with open(f"{path_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dev_df, test_df = load_dataset(dataset_path=args.dataset_path, test_ratio=args.test_ratio)
    
    splitter = Splitter(data=dev_df)
    df_splits = splitter.get_splits(
        strategy=args.strategy, 
        train_ratio=args.train_ratio, 
        initial_train_size=args.initial_train_size, 
        step_size=args.step_size, 
        window_size=args.step_size
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{timestamp}_{args.exp_name}")
    model = model_attention.Model(args=args).to(device)

    best_state, final_preprocessor, global_epoch = run_cross_validation(args, model, df_splits, writer, device)
    
    if final_preprocessor is not None:
        final_preprocessor.save_scalers(f"checkpoints/scalers/{args.model_name}")

    if best_state is not None:
        model.load_state_dict(best_state)
        save_dir = f"checkpoint/{args.model_name}"
        save_model(model, args, save_dir)
        print(f"\n-> Final best model saved to {save_dir}/")

    if final_preprocessor is not None:
        evaluate_test_set(args, model, test_df, final_preprocessor, writer, device, global_epoch)
    
    writer.close()
    print("Pipeline finished successfully. To view learning curves, run: tensorboard --logdir=runs")

if __name__ == "__main__":
    parser = get_parser()
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)