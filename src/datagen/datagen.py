import numpy as np
import os
import pandas as pd
from src.data.profile_transforms import one_hot_encode_column
from src.config.config import cfg

class DataGenerator:
    def __init__(self, matrix=cfg.MATRIX, seed=42):
        self.MATRIX = matrix
        if seed is not None:
            np.random.seed(seed)

    def sample_ordinal_for_category(self, cat, sampling_probs):
        d = self.MATRIX[cat]["DIMENSION"]
        p = sampling_probs
        if d == 3:
            return np.random.choice([1, 2, 3], p=p)
        else:
            return np.random.choice([0, 1, 2, 3], p=p)

    def sample_exit_and_funding(self, p_fund, mu_fund, sig_fund, p_exit, mu_exit, sig_exit):
        if np.random.rand() < p_fund:
            funding_amt = np.random.lognormal(mu_fund, sig_fund)
            if np.random.rand() < p_exit:
                exit_val = np.random.lognormal(mu_exit, sig_exit)
            else:
                exit_val = 0
        else:
            funding_amt = 0
            exit_val = 0
        return exit_val, funding_amt

    def generate_subpopulation(self, num_samples, pop_cfg):
        X_list, e_list, f_list = [], [], []
        for _ in range(num_samples):
            x_ordinals = []
            for cat in self.MATRIX:
                d = self.MATRIX[cat]["DIMENSION"]
                
                p = pop_cfg["sampling_probs"][cat].copy()
                
                if len(p) != d:
                    raise ValueError(f"Probability vector length ({len(p)}) doesn't match dimension ({d}) for category {cat}")
                
                # Add small noise while ensuring probabilities stay positive
                noise = np.random.normal(0, 0.01, size=d)
                p = p + noise
                p = np.clip(p, 0.01, 0.99)  # Ensure strictly positive
                p = p / p.sum()  # Renormalize
                
                val = self.sample_ordinal_for_category(cat, p)
                x_ordinals.append(val)
            
            x = np.array(x_ordinals)
            
            # Sample exit and funding values
            e_val, f_val = self.sample_exit_and_funding(
                pop_cfg["p_funding"],
                pop_cfg["mu_funding"],
                pop_cfg["sigma_funding"],
                pop_cfg["p_exit"],
                pop_cfg["mu_exit"],
                pop_cfg["sigma_exit"],
            )
            
            # Add noise to funding/exit values
            if np.random.rand() < pop_cfg["p_funding"]:
                noise = np.random.normal(0, 0.2)
                e_val = np.random.lognormal(e_val + noise, pop_cfg["sigma_exit"])
                f_val = np.random.lognormal(f_val + noise, pop_cfg["sigma_funding"])
            
            X_list.append(x)
            e_list.append(e_val)
            f_list.append(f_val)
            
        return np.array(X_list), np.array(e_list), np.array(f_list)

    def generate_dataset(self, total_samples, populations):
        X_all, e_all, f_all = [], [], []
        labels = []
        for pop_name, pop_cfg in populations.items():
            n_sub = int(round(pop_cfg["fraction"] * total_samples))
            X_sub, e_sub, f_sub = self.generate_subpopulation(n_sub, pop_cfg)
            X_all.append(X_sub)
            e_all.append(e_sub)
            f_all.append(f_sub)
            labels += [pop_name] * n_sub

        X_ordinal = np.vstack(X_all)
        exit_final = np.concatenate(e_all)
        fund_final = np.concatenate(f_all)
        labels = np.array(labels[: len(exit_final)])

        # One-hot encode each column
        X_encoded = []
        for i, cat in enumerate(self.MATRIX):
            ordinal_column = X_ordinal[:, i]
            dim = self.MATRIX[cat]["DIMENSION"]
            
            # No need to adjust indexing - the function handles it
            encoded_column = one_hot_encode_column(ordinal_column, dim)
            X_encoded.append(encoded_column)
        
        # Combine the encoded columns
        X_final = np.hstack(X_encoded)

        return X_final, exit_final, fund_final, labels

    def save_synthetic_dataset(self, X, exit_values, funding_amounts, matrix, 
                               output_path="../data/synth/encoded_founders_composites.csv", 
                               success_funding_threshold=None):

        if success_funding_threshold is None:
            success_funding_threshold = cfg.SYNTH["SUCCESS_FUNDING_THRESHOLD"]
        
        feature_names = []
        for cat in matrix:
            dim = matrix[cat]["DIMENSION"]
            for i in range(dim):
                if dim == 3:
                    feature_names.append(f"{cat}_{i + 1}")
                else:
                    feature_names.append(f"{cat}_{i}")
        
        df = pd.DataFrame(X, columns=feature_names)
        df["exit_value"] = exit_values
        df["funding_amount"] = funding_amounts
        df["success"] = ((df["exit_value"] > 0) | (df["funding_amount"] > success_funding_threshold)).astype(int)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
