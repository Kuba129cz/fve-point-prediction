# src/preprocessing.py
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """Handles feature and target scaling for a single validation fold to prevent data leakage."""
    
    def __init__(self, feature_cols: list[str], target_col: str, fold_idx: int = 0):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.fold_idx = fold_idx
        
        self.cols_to_scale = [
            col for col in self.feature_cols if not (col.startswith("sin_") or col.startswith("cos_"))
        ]
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def process_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scales training and validation data without data leakage."""
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()

        if self.cols_to_scale:
            train_scaled[self.cols_to_scale] = self.feature_scaler.fit_transform(train_df[self.cols_to_scale])
            val_scaled[self.cols_to_scale] = self.feature_scaler.transform(val_df[self.cols_to_scale])

        train_scaled[[self.target_col]] = self.target_scaler.fit_transform(train_df[[self.target_col]])
        val_scaled[[self.target_col]] = self.target_scaler.transform(val_df[[self.target_col]])

        return train_scaled, val_scaled

    def save_scalers(self, save_dir: str = "checkpoints/scalers"):
        """Saves the fitted scalers and metadata to disk for future inference or evaluation."""
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.feature_scaler, f"{save_dir}/feature_scaler_fold_{self.fold_idx}.pkl")
        joblib.dump(self.target_scaler, f"{save_dir}/target_scaler_fold_{self.fold_idx}.pkl")
        
        joblib.dump(self.cols_to_scale, f"{save_dir}/scaled_features_list_fold_{self.fold_idx}.pkl")
    
    def load_scalers(self, save_dir: str = "checkpoints/scalers"):
        """Loads the fitted scalers and metadata from disk to restore the preprocessor state."""
        feature_scaler_path = f"{save_dir}/feature_scaler_fold_{self.fold_idx}.pkl"
        target_scaler_path = f"{save_dir}/target_scaler_fold_{self.fold_idx}.pkl"
        features_list_path = f"{save_dir}/scaled_features_list_fold_{self.fold_idx}.pkl"

        if not (os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path)):
            raise FileNotFoundError(f"Nebyly nalezeny soubory scalerů pro fold {self.fold_idx} ve složce {save_dir}!")

        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        self.cols_to_scale = joblib.load(features_list_path)
        print(f"Scalery pro fold {self.fold_idx} byly úspěšně načteny z disku.")
    
    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Transforms raw input data using already fitted scalers.
        Safe for test sets and real-time production inference.
        """
        df_scaled = df.copy()

        if self.cols_to_scale:
            df_scaled[self.cols_to_scale] = self.feature_scaler.transform(df[self.cols_to_scale])
            
        if include_target and self.target_col in df.columns:
            df_scaled[[self.target_col]] = self.target_scaler.transform(df[[self.target_col]])

        return df_scaled