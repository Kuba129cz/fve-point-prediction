import torch
import pandas as pd
import copy
import os
from torch.utils.tensorboard import SummaryWriter

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.metrics import ErrorTracker, calculate_metrics, log_epoch_metrics, log_test_metrics

def load_dataset(dataset_path: str, test_ratio: float, index_col: str = "timestamp", freq: str = "1h") -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(filepath_or_buffer=dataset_path, index_col=index_col, parse_dates=True)
    dataset = dataset.sort_index().asfreq(freq=freq)
    
    test_split_idx = int(len(dataset) * (1 - test_ratio))
    dev_df = dataset.iloc[:test_split_idx].copy()
    test_df = dataset.iloc[test_split_idx:].copy()
    return dev_df, test_df

def prepare_fold_data(args, train_df: pd.DataFrame, val_df: pd.DataFrame, fold_idx: int) -> tuple:
    preprocessor = Preprocessor(lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col, fold_idx=fold_idx)
    train_scaled, val_scaled = preprocessor.process_fold(train_df=train_df, val_df=val_df)
    
    train_loader = torch.utils.data.DataLoader(
        Dataset(data=train_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        Dataset(data=val_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col), batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader, preprocessor

def prepare_test_data(args, test_df: pd.DataFrame, preprocessor: Preprocessor) -> torch.utils.data.DataLoader:
    test_scaled = test_df.copy()
    if preprocessor.lookback_cols_to_scale:
        test_scaled[preprocessor.lookback_cols_to_scale] = preprocessor.lookback_scaler.transform(test_df[preprocessor.lookback_cols_to_scale])
    if preprocessor.horizon_cols_to_scale:
        test_scaled[preprocessor.horizon_cols_to_scale] = preprocessor.horizon_scaler.transform(test_df[preprocessor.horizon_cols_to_scale])
    
    test_scaled[[args.target_col]] = preprocessor.target_scaler.transform(test_df[[args.target_col]])

    return torch.utils.data.DataLoader(
        Dataset(data=test_scaled, lookback=args.lookback, horizon=args.horizon, 
                lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
        batch_size=args.batch_size, shuffle=False
    )

def run_evaluation_loop(model, dataloader, criterion, device, preprocessor, is_train: bool = False, optimizer=None, curr_epoch: int = 0):
    """
    Unified single-epoch executor for both training and validation modes.
    """
    model.train() if is_train else model.eval()
    total_scaled_loss = 0.0
    tracker = ErrorTracker()
    
    for batch in dataloader:
        x_past, x_future, targets = [b.to(device) for b in batch]
        
        if is_train:
            optimizer.zero_grad()
            outputs = model(x_past, x_future, targets=targets, curr_epoch=curr_epoch).squeeze(-1)   
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(x_past, x_future).squeeze(-1)
                loss = criterion(outputs, targets)
        
        total_scaled_loss += loss.item() * x_past.size(0)
        
        with torch.no_grad():
            y_pred_real = torch.from_numpy(preprocessor.inverse_transform_target(outputs.detach().cpu().numpy()))
            y_true_real = torch.from_numpy(preprocessor.inverse_transform_target(targets.detach().cpu().numpy()))
            tracker.update(y_pred_real, y_true_real)
                
    return total_scaled_loss / len(dataloader.dataset), tracker.compute()

def train_fold(args, model, train_loader, val_loader, preprocessor, criterion, optimizer, scheduler, writer, device, global_epoch):
    fold_best_val_loss = float('inf')
    best_model_state_fold = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_raw = run_evaluation_loop(model, train_loader, criterion, device, preprocessor, is_train=True, optimizer=optimizer, curr_epoch=epoch)
        val_loss, val_raw = run_evaluation_loop(model, val_loader, criterion, device, preprocessor, is_train=False)
        
        t_metrics = calculate_metrics(train_raw, args.nominal_output)
        v_metrics = calculate_metrics(val_raw, args.nominal_output)
        
        if v_metrics["real_mae"] < fold_best_val_loss:
            fold_best_val_loss = v_metrics["real_mae"]
            best_model_state_fold = copy.deepcopy(model.state_dict())
            patience_counter = 0 
        else:
            patience_counter += 1 

        current_lr = scheduler.get_last_lr()[0]
        log_epoch_metrics(writer, global_epoch, train_loss, val_loss, t_metrics, v_metrics, current_lr)
        scheduler.step()
        
        if (epoch + 1) % args.print_freq == 0 or epoch == 0:
            print(f"--- Epoch {epoch + 1}/{args.epochs} (Report #{(epoch + 1) // args.print_freq}) ---")
            print(f"    Scaled Loss      : Train {train_loss:.4f} | Val {val_loss:.4f}")
            print(f"    Real Error (All) : Train {t_metrics['real_mae']:.2f} ({t_metrics['mae_pct']:.2f}%) | Val {v_metrics['real_mae']:.2f} ({v_metrics['mae_pct']:.2f}%)")
            print(f"    Real Error (P>0) : Train {t_metrics['real_active_mae']:.2f} ({t_metrics['active_mae_pct']:.2f}%) | Val {v_metrics['real_active_mae']:.2f} ({v_metrics['active_mae_pct']:.2f}%)")
            print(f"    Real RMSE / MBE  : Val RMSE: {v_metrics['real_rmse']:.2f} | Val MBE: {v_metrics['real_mbe']:.2f}")
            if patience_counter > 0:
                print(f"    [Early Stopping: {patience_counter}/{args.patience} epochs without improvement]")
        
        global_epoch += 1
        if patience_counter >= args.patience:
            print(f"\n[!] Stopping fold early. Validation error has not improved for {args.patience} epochs.")
            break 
            
    return fold_best_val_loss, best_model_state_fold, global_epoch

def run_cross_validation(args, model, df_splits, writer, device):
    criterion = torch.nn.L1Loss()
    global_epoch = 0
    global_best_loss = float('inf') 
    global_best_state = None
    last_preprocessor = None
    
    for fold_idx, (train_df, val_df) in enumerate(df_splits):
        print(f"\n{'='*30}\n--- Starting FOLD {fold_idx + 1} ---\n{'='*30}")
        
        train_loader, val_loader, preprocessor = prepare_fold_data(args, train_df, val_df, fold_idx)
        last_preprocessor = preprocessor
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min)
        
        fold_best_loss, fold_best_state, global_epoch = train_fold(
            args, model, train_loader, val_loader, preprocessor, 
            criterion, optimizer, scheduler, writer, device, global_epoch
        )
        
        if fold_best_loss < global_best_loss and fold_best_state is not None:
            print(f"-> Fold {fold_idx + 1} yielded a new GLOBAL best score: {fold_best_loss:.2f}")
            global_best_loss = fold_best_loss
            global_best_state = copy.deepcopy(fold_best_state)
            
        print(f"-> Fold {fold_idx + 1} completed.")

    return global_best_state, last_preprocessor, global_epoch

def evaluate_test_set(args, model, test_df, preprocessor, writer, device, global_epoch):
    print(f"\n{'='*30}\n--- Testing on hold-out Test Set ---\n{'='*30}")
    
    test_loader = prepare_test_data(args, test_df, preprocessor)
    test_loss, test_raw = run_evaluation_loop(model, test_loader, torch.nn.L1Loss(), device, preprocessor, is_train=False)
    metrics = calculate_metrics(test_raw, args.nominal_output)
    
    print(f"Final Test Error (All)  : {metrics['real_mae']:.2f} ({metrics['mae_pct']:.2f}%)")
    print(f"Final Test Error (P>0)  : {metrics['real_active_mae']:.2f} ({metrics['active_mae_pct']:.2f}%)")
    print(f"Final Test RMSE / MBE   : RMSE: {metrics['real_rmse']:.2f} | MBE: {metrics['real_mbe']:.2f}")

    log_test_metrics(writer, global_epoch, test_loss, metrics, args)