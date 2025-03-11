import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import cfg
from src.datagen import datagen

def generate_synthetic_data(n_samples, output_path):
    """Generate synthetic founder dataset"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    dg = datagen.DataGenerator()
    X_syn, exit_syn, fund_syn, _ = dg.generate_dataset(n_samples, cfg.SYNTH['POPULATIONS'])
    
    df = dg.save_synthetic_dataset(X_syn, exit_syn, fund_syn, cfg.MATRIX, output_path)
    print(f"Generated {n_samples} samples to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--output_path', default="data/synth/encoded_founders_composites.csv")
    args = parser.parse_args()
    
    generate_synthetic_data(args.n_samples, args.output_path) 