from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
from src.config.config import cfg
import pickle
from src.models.quadratic import QuadMLP

def search_founders(px, limit=10, ids=None):
    """Search for founders based on parameters and return their LinkedIn profiles"""
    
    data = px.person_search(params=cfg.FOUNDER_SEARCH_PARAMS, N=limit, ids=ids)
    print(f"Found {len(data.get('results', []))} profiles")
    
    return data

def rank_profiles(df, feature_matrix, model_dict):
    """Rank profiles using the loaded model"""
    if not model_dict:
        return df
    
    model = model_dict['model']
    
    scaler = StandardScaler()
    scaler.__setstate__(model_dict['scaler_state'])
    
    X = scaler.transform(feature_matrix)
    X = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(X)
        scores = torch.sigmoid(outputs).numpy().flatten()
    
    df['score'] = scores
    ranked_df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return ranked_df


def load_model(model_path='models/founder_rank.pkl'):
    """Load the model and return the model, weight matrices, and feature names."""
    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        input_dim = len(checkpoint['feature_names'])
        model = QuadMLP(input_dim)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return {
            'model': model,
            'W_init': checkpoint['W_init'],
            'W_final': checkpoint['W_final'],
            'feature_names': checkpoint['feature_names'],
            'scaler_state': checkpoint['scaler_state']
        }
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
