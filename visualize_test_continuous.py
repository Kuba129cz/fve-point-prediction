import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import model_attention

def load_ensemble_models(model_dir, device):
    """Loads the configuration and ALL trained models from the folder."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    args = argparse.Namespace(**config_dict)
    
    model_paths = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_paths:
        raise ValueError(f"No models (.pth) found in folder {model_dir}!")
        
    print(f"Loading {len(model_paths)} models into ensemble...")
    models = []
    for path in model_paths:
        model = model_attention.Model(args=args).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        
    return models, args

def apply_physics_mask(preds, nominal_output):
    """Prevents the model from predicting below zero or above the absolute physical limit."""
    return np.clip(preds, a_min=0.0, a_max=nominal_output * 1.05)

def generate_all_sequence_plots(model_dir, data_path, save_dir="ensemble_sequence_plots"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Load configuration and ensemble models
    models, args = load_ensemble_models(model_dir, device)
    nominal_output = getattr(args, 'nominal_output', 1293)
    
    # 2. Load and prepare dataset split
    dataset = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    dataset = dataset.sort_index().asfreq('1h')
    
    test_split_idx = int(len(dataset) * (1 - args.test_ratio))
    test_df = dataset.iloc[test_split_idx:].copy()
    
    # 3. Restore preprocessor state and transform test set
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col, fold_idx=0)
    preprocessor.load_scalers(save_dir=model_dir) 
    
    test_scaled = preprocessor.transform(test_df, include_target=True)
    
    target_mean = preprocessor.target_scaler.mean_[0]
    target_std = preprocessor.target_scaler.scale_[0]

    # 4. Initialize PyTorch Dataset
    test_dataset = Dataset(data=test_scaled, lookback=args.lookback, horizon=args.horizon, 
                           lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)
    
    print(f"Total available test steps: {len(test_dataset)}")
    print(f"Generating charts stepping by t+{args.horizon} hours...")

    sequence_counter = 0

    # 5. Iterate over the entire test set with a stride equal to the horizon (24 hours)
    with torch.no_grad():
        for idx in range(0, len(test_dataset), args.horizon):
            x_past, x_future, targets = test_dataset[idx]
            x_past = x_past.unsqueeze(0).to(device)
            x_future = x_future.unsqueeze(0).to(device)
            
            # Inverse transform target values to real kW
            y_true = targets.numpy() * target_std + target_mean
            
            # Get predictions from all models in the ensemble
            preds_list = []
            for model in models:
                p = model(x_past, x_future).squeeze(-1)
                preds_list.append(p[0].cpu().numpy() * target_std + target_mean)
                
            # Average the predictions and apply the physical constraint mask
            y_pred = apply_physics_mask(np.mean(preds_list, axis=0), nominal_output)
            
            # Extract actual timestamps for the 24-hour horizon to display on the X-axis
            start_timestamp_idx = idx + args.lookback
            end_timestamp_idx = start_timestamp_idx + args.horizon
            time_axis = test_df.index[start_timestamp_idx:end_timestamp_idx]
            
            # String representations for filenames and titles
            seq_date_str = time_axis[0].strftime("%Y-%m-%d")
            seq_time_str = time_axis[0].strftime("%Y-%m-%d %H:%M")
            
            # 6. Plotting the sequence
            plt.figure(figsize=(11, 5))
            
            plt.plot(time_axis, y_true, label='Real Production', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
            plt.plot(time_axis, y_pred, label=f'Ensemble Prediction ({len(models)} models)', color='#ff7f0e', linewidth=2.0, linestyle='--', marker='s', markersize=4)
            
            plt.title(f"PV Power Comparison | Sequence #{sequence_counter:03d} (Starts: {seq_time_str})", fontsize=12, fontweight='bold')
            plt.xlabel("Timeline (Date & Hour)", fontsize=11)
            plt.ylabel("PV Power Output [kW]", fontsize=11)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(fontsize=10, loc='upper right')
            
            plt.ylim(-20, nominal_output * 1.1)
            
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            filename = os.path.join(save_dir, f"seq_{sequence_counter:03d}_{seq_date_str}.png")
            plt.savefig(filename, dpi=150)
            plt.close()
            
            if (sequence_counter + 1) % 10 == 0 or sequence_counter == 0:
                print(f" -> Generated {sequence_counter + 1} plots...")
                
            sequence_counter += 1
            
    print(f"\n[Success] All {sequence_counter} individual sequence charts saved to '{save_dir}/'")

if __name__ == "__main__":
    MODEL_DIRECTORY = "./ensemble_models" 
    DATASET_PATH = "data/fve_aba_dataset.csv"
    
    generate_all_sequence_plots(model_dir=MODEL_DIRECTORY, data_path=DATASET_PATH)