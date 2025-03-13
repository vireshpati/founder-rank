import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import pickle
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing import load_and_preprocess_data
from src.models.trainer import create_model_and_trainer

def train_model(data_path, output_path):
    """Train founder ranking model"""
    # Define hyperparameters
    hyp = {
        "data_paths": [
            str(PROJECT_ROOT / "data/encoded/S21_encoded_with_outcomes.csv"),
            str(PROJECT_ROOT / "data/encoded/W21_encoded_with_outcomes.csv"),
            str(PROJECT_ROOT / "data/encoded/S17_encoded_with_outcomes.csv"),
            str(PROJECT_ROOT / "data/encoded/W17_encoded_with_outcomes.csv"),
            str(PROJECT_ROOT / "data/encoded/top_companies_encoded_with_outcomes.csv")
        ],
        "synthetic_data_path": data_path,
        "test_size": 0.17,  
        "val_size": 0.175,   
        "random_state": 42,
        "batch_size": 64,     
        "lr": 0.0005,         
        "weight_decay": 1.8e-3,  #l2
        "epochs": 500,       
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exclude_columns": ["success", "exit_value", "funding_amount", "batch"],
        "target_column": "success",
        
        # Regularization parameters
        "diag_penalty": 0.0002,    # individual feature weights
        "l1_penalty": 0.0004,      
        "top_k_penalty": 0.0005,    # top k feature interactions
        "top_k": 15,                
        "dropout": 0.15,           
        
        # Training parameters
        "log_every": 10,
        "early_stopping_patience": 250,
        
        # Evaluation metrics parameters
        "k_values": [10, 25, 50],  # K values for NDCG and Precision
        "early_stopping_metric": "ndcg@50" ,
        
        'ranking_weight': 0.6,
        "pos_weight": None,  # or calculate based on class imbalance
    }
    
    # Make sure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tensors, feature_names, scaler = load_and_preprocess_data(hyp)
    X_train_tensor, y_train_tensor, is_synthetic_train_tensor = tensors['train']
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, is_synthetic_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=hyp["batch_size"], shuffle=True)

    model, trainer = create_model_and_trainer(X_train_tensor.shape[1], hyp, pos_weight=hyp["pos_weight"])
    W_init = trainer.train(train_loader, tensors['val'][0], tensors['val'][1], 
                          tensors['test'][0], tensors['test'][1])
    
    if output_path:
        # Save in the format expected by load_model in ranking.py
        model_data = {
            'model_state_dict': model.state_dict(),
            'W_init': W_init,
            'W_final': model.get_W().detach().cpu().numpy(),
            'feature_names': feature_names,
            'scaler_state': scaler.__getstate__(),
            'hyperparameters': hyp
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {output_path}")
    
    return model, W_init, feature_names, scaler, hyp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=str(PROJECT_ROOT / "data/synth/encoded_founders_composites.csv"))
    parser.add_argument('--output_path', default=str(PROJECT_ROOT / "models/founder_rank.pkl"))
    args = parser.parse_args()
    
    model, W_init, feature_names, scaler, hyp = train_model(args.data_path, args.output_path)
    
