import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(hyp):
   
    real_data = pd.concat([
        pd.read_csv(path).assign(batch=path.split('/')[-1].split('_')[0]) 
        for path in hyp["data_paths"]
    ])
    

    synthetic_data = pd.read_csv(hyp["synthetic_data_path"]) if hyp["synthetic_data_path"] else None
    if synthetic_data is not None:
        synthetic_data['batch'] = 'synthetic'
    
    feature_columns = [col for col in real_data.columns if col not in hyp["exclude_columns"]]
    
    
    X_real = real_data[feature_columns].to_numpy()
    y_real = real_data[hyp["target_column"]].values
    
   
    X_real_train, X_test, y_real_train, y_test = train_test_split(
        X_real, y_real,
        test_size=hyp["test_size"],
        random_state=hyp["random_state"],
        stratify=y_real
    )

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_real_train, y_real_train,
        test_size=hyp["val_size"],
        random_state=hyp["random_state"],
        stratify=y_real_train
    )

    if synthetic_data is not None:
        X_synth = synthetic_data[feature_columns].to_numpy()
        y_synth = synthetic_data[hyp["target_column"]].values
        X_train = np.vstack([X_train, X_synth])
        y_train = np.concatenate([y_train, y_synth])
        is_synthetic = np.concatenate([np.zeros(len(X_train) - len(X_synth)), np.ones(len(X_synth))])
        
        print("\nData Split Summary:")
        print(f"Real data: {len(X_real)} samples")
        print(f"Synthetic data: {len(X_synth)} samples")
        print(f"\nTrain set: {len(X_train)} samples ({sum(is_synthetic == 0)} real, {sum(is_synthetic == 1)} synthetic)")
        print(f"Val set: {len(X_val)} samples (all real)")
        print(f"Test set: {len(X_test)} samples (all real)")
        
        print(f"\nClass distribution:")
        print(f"Train set - Real: {sum(y_train[is_synthetic == 0] == 1)}/{sum(is_synthetic == 0)} positive")
        print(f"Train set - Synthetic: {sum(y_train[is_synthetic == 1] == 1)}/{sum(is_synthetic == 1)} positive")
        print(f"Val set: {sum(y_val == 1)}/{len(y_val)} positive")
        print(f"Test set: {sum(y_test == 1)}/{len(y_test)} positive")
    else:
        is_synthetic = np.zeros(len(X_train))
        
        print("\nData Split Summary (No synthetic data):")
        print(f"Total data: {len(X_real)} samples")
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        print(f"\nClass distribution:")
        print(f"Train set: {sum(y_train == 1)}/{len(y_train)} positive")
        print(f"Val set: {sum(y_val == 1)}/{len(y_val)} positive")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    tensors = {
        'train': (
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train),
            torch.FloatTensor(is_synthetic)
        ),
        'val': (
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        ),
        'test': (
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test)
        )
    }
    
    return tensors, feature_columns, scaler
