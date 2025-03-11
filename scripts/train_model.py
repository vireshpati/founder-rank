import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import load_and_preprocess_data
from src.models.trainer import create_model_and_trainer

def train_model(data_path, output_path):
    """Train founder ranking model"""
    hyp = {
        "data_paths": [
            "data/encoded/S21_encoded_with_outcomes.csv",
            "data/encoded/W21_encoded_with_outcomes.csv",
            "data/encoded/S17_encoded_with_outcomes.csv",
            "data/encoded/W17_encoded_with_outcomes.csv",
            "data/encoded/top_companies_encoded_with_outcomes.csv"
        ],
        "synthetic_data_path": 'data/synth/encoded_founders_composites.csv',
        "test_size": 0.125,  
        "val_size": 0.125,   
        "random_state": 42,
        "batch_size": 32,     
        "lr": 0.0005,         
        "weight_decay": 1.8e-3,  
        "epochs": 100,       
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exclude_columns": ["success", "exit_value", "funding_amount", "batch"],
        "target_column": "success",
        
        # Regularization parameters
        "diag_penalty": 0.0002,    
        "l1_penalty": 0.0004,      
        "top_k_penalty": 0.0005,   
        "top_k": 15,               
        "dropout": 0.15,           
        
        # Training parameters
        "log_every": 10,
        "early_stopping_patience": 250, 
        
        "pos_weight": 1.0
    }
    

    tensors, feature_names, scaler = load_and_preprocess_data(hyp)
    X_train_tensor, y_train_tensor, is_synthetic_train_tensor = tensors['train']
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, is_synthetic_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=hyp["batch_size"], shuffle=True)

    model, trainer = create_model_and_trainer(X_train_tensor.shape[1], hyp, pos_weight=hyp["pos_weight"])
    W_init = trainer.train(train_loader, tensors['val'][0], tensors['val'][1], 
                          tensors['test'][0], tensors['test'][1])
    
    if output_path:
        model_data = {
            'model': model,
            'W_init': W_init,
            'feature_names': feature_names,
            'scaler': scaler,
            'hyp': hyp
        }
        torch.save(model_data, output_path)
        print(f"Model saved to {output_path}")
    
    return model, W_init, feature_names, scaler, hyp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../data/synth/encoded_founders_composites.csv")
    parser.add_argument('--output_path', default="../models/founder_rank.pkl")
    args = parser.parse_args()
    
    model, W_init, feature_names, scaler, hyp = train_model(args.data_path, args.output_path)
    
