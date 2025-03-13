import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, ndcg_score
from src.models.quadratic import QuadMLP, QuadraticModel
from src.config.config import cfg
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, scores, labels):
        # Create all possible pairs of scores and corresponding labels
        n = scores.size(0)
        scores_i = scores.repeat_interleave(n, dim=0)
        scores_j = scores.repeat(n)
        labels_i = labels.repeat_interleave(n, dim=0)
        labels_j = labels.repeat(n)
        
        # We only want pairs where labels_i > labels_j (positive should be ranked higher than negative)
        mask = (labels_i > labels_j).float()
        
        if mask.sum() == 0:  # No valid pairs in this batch
            return torch.tensor(0.0, device=scores.device)
        
        # Calculate pairwise ranking loss (similar to RankNet)
        # We want scores_i > scores_j when labels_i > labels_j
        diff = scores_i - scores_j
        loss = -torch.log(torch.sigmoid(diff) + 1e-8) * mask
        
        # Return mean loss over all valid pairs
        return loss.sum() / (mask.sum() + 1e-8)

class ModelTrainer:
    def __init__(self, model, config: Dict[str, Any], pos_weight=None):
        self.model = model.to(config["device"])
        self.config = config
        self.criterion = FocalLoss(alpha=2.0, gamma=2.0)
        self.ranking_criterion = PairwiseRankingLoss(margin=0.0)
        self.ranking_weight = config.get("ranking_weight", 0.5)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01)
        
        # Get K values for metrics from config or use defaults
        self.k_values = config.get("k_values", [10, 25, 50])
        
        # State tracking
        self.best_model_state = None
        self.best_val_metric = float("-inf")
        
        # Initialize metrics history with dynamic K values
        self.metrics_history = {
            'train': {'loss': [], 'auc': [], 'ndcg': [], 'accuracy': []},
            'val': {'loss': [], 'auc': [], 'ndcg': [], 'accuracy': []},
            'test': {'loss': [], 'auc': [], 'ndcg': [], 'accuracy': []}
        }
        
        # Add precision and ndcg metrics for each K value
        for split in self.metrics_history:
            for k in self.k_values:
                self.metrics_history[split][f'ndcg@{k}'] = []
                self.metrics_history[split][f'precision@{k}'] = []
        
        # Create a better organized header grouped by train/val/test
        header = f"{'Epoch':>5} "
        
        # Training metrics
        header += f"{'Train Loss':>10} {'Train AUC':>10} {'Train NDCG':>10} "
        for k in self.k_values:
            header += f"{'Tr N@'+str(k):>8} {'Tr P@'+str(k):>8} "
        header += f"{'Train Acc':>9} "
        
        # Validation metrics
        header += f"{'Val Loss':>10} {'Val AUC':>10} {'Val NDCG':>10} "
        for k in self.k_values:
            header += f"{'Val N@'+str(k):>8} {'Val P@'+str(k):>8} "
        header += f"{'Val Acc':>9} "
        
        # Test metrics
        header += f"{'Test Loss':>10} {'Test AUC':>10} {'Test NDCG':>10} "
        for k in self.k_values:
            header += f"{'Tst N@'+str(k):>8} {'Tst P@'+str(k):>8} "
        header += f"{'Test Acc':>9}"
        
        print(header)
        print("-" * (180 + len(self.k_values) * 32))  # Adjust line length based on number of K values

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X, y, _ in train_loader:
            X, y = X.to(self.config["device"]), y.to(self.config["device"])
            
            # Apply label smoothing
            y_smooth = self.smooth_labels(y, smoothing=0.03)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            
            # Classification loss
            cls_loss = self.criterion(outputs, y_smooth)
            
            # Ranking loss (only if we have enough samples in the batch)
            if X.size(0) > 1:
                rank_loss = self.ranking_criterion(outputs, y)
            else:
                rank_loss = torch.tensor(0.0, device=self.config["device"])
            
            # Combine losses
            loss = cls_loss + self.ranking_weight * rank_loss
            
            # Add regularization penalties
            W = self.model.get_W()
            penalties = (
                self.config["diag_penalty"] * torch.sum(torch.square(torch.diag(W))) +
                self.config["l1_penalty"] * torch.sum(torch.abs(W)) +
                self.config["top_k_penalty"] * torch.sum(torch.topk(torch.abs(W).flatten(), self.config["top_k"])[0])
            )
            
            loss = loss + penalties
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X = X.to(self.config["device"])
        y = y.to(self.config["device"])
        
        outputs = self.model(X)
        loss = self.criterion(outputs, y).item()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > cfg.THRESHOLD).astype(int)
        y = y.cpu().numpy()
        
        # Calculate metrics using sklearn
        y_true = y.reshape(1, -1)
        y_score = probs.reshape(1, -1)
        
        # Only calculate if we have positive examples
        has_positives = np.sum(y) > 0
        has_negatives = np.sum(1 - y) > 0
        
        # Initialize all metrics to avoid KeyError
        results = {
            'loss': loss,
            'accuracy': (preds == y).mean(),
            'auc': 0.0,
            'ndcg': 0.0,
            'probs': probs,
            'preds': preds
        }
        
        # Initialize all K-specific metrics
        for k in self.k_values:
            results[f'ndcg@{k}'] = 0.0
            results[f'precision@{k}'] = 0.0
        
        # AUC requires both positive and negative examples
        if has_positives and has_negatives:
            results['auc'] = roc_auc_score(y, probs)
        
        # NDCG calculation
        if has_positives:
            # Overall NDCG (considers all items in the ranking)
            results['ndcg'] = ndcg_score(y_true, y_score)  # k=None by default
            
            # Calculate NDCG for each K value
            for k in self.k_values:
                k_actual = min(k, len(y))  # Handle case where k is larger than dataset
                if k_actual > 0:
                    results[f'ndcg@{k}'] = ndcg_score(y_true, y_score, k=k_actual)
        
        # Precision@k calculation
        if has_positives:
            for k in self.k_values:
                k_actual = min(k, len(y))  # Handle case where k is larger than dataset
                if k_actual > 0:
                    # Get indices of top k predictions
                    top_k_indices = np.argsort(probs)[::-1][:k_actual]
                    # Calculate precision at k (true positives / k)
                    results[f'precision@{k}'] = np.sum(y[top_k_indices]) / k_actual
        
        return results

    def smooth_labels(self, labels, smoothing=0.03):
        labels = labels.float().to(self.config["device"])
        return labels * (1 - smoothing) + 0.5 * smoothing

    def train(self, train_loader, X_val, y_val, X_test=None, y_test=None):
        W_init = self.model.get_W().detach().cpu().numpy()
        patience = self.config.get("early_stopping_patience", 50)
        patience_counter = 0
        
        # Get the metric to use for early stopping
        early_stopping_metric = self.config.get("early_stopping_metric", "ndcg")
        self.best_val_metric = float("-inf")
        
        print(f"Using k_values: {self.k_values}")
        
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            metrics = {
                'train': self.evaluate(train_loader.dataset.tensors[0], train_loader.dataset.tensors[1]),
                'val': self.evaluate(X_val, y_val),
                'test': self.evaluate(X_test, y_test) if X_test is not None else None
            }
            
            # Update history
            for split in metrics:
                if metrics[split]:
                    for metric, value in metrics[split].items():
                        if metric not in ['probs', 'preds']:
                            self.metrics_history[split][metric].append(value)
            
            # Logging
            if epoch % self.config.get("log_every", 1) == 0:
                log_str = f"{epoch:5d} "
                
                # Training metrics
                log_str += f"{train_loss:10.4f} {metrics['train']['auc']:10.4f} {metrics['train']['ndcg']:10.4f} "
                for k in self.k_values:
                    log_str += f"{metrics['train'][f'ndcg@{k}']:8.4f} {metrics['train'][f'precision@{k}']:8.4f} "
                log_str += f"{metrics['train']['accuracy']:9.4f} "
                
                # Validation metrics
                log_str += f"{metrics['val']['loss']:10.4f} {metrics['val']['auc']:10.4f} {metrics['val']['ndcg']:10.4f} "
                for k in self.k_values:
                    log_str += f"{metrics['val'][f'ndcg@{k}']:8.4f} {metrics['val'][f'precision@{k}']:8.4f} "
                log_str += f"{metrics['val']['accuracy']:9.4f} "
                
                # Test metrics
                test_loss = metrics['test']['loss'] if metrics['test'] else 0
                test_auc = metrics['test']['auc'] if metrics['test'] else 0
                test_ndcg = metrics['test']['ndcg'] if metrics['test'] else 0
                test_acc = metrics['test']['accuracy'] if metrics['test'] else 0
                
                log_str += f"{test_loss:10.4f} {test_auc:10.4f} {test_ndcg:10.4f} "
                
                for k in self.k_values:
                    test_ndcg_k = metrics['test'][f'ndcg@{k}'] if metrics['test'] else 0
                    test_prec_k = metrics['test'][f'precision@{k}'] if metrics['test'] else 0
                    log_str += f"{test_ndcg_k:8.4f} {test_prec_k:8.4f} "
                
                log_str += f"{test_acc:9.4f}"
                
                print(log_str)
            
            # Early stopping based on the specified metric
            current_metric_value = metrics['val'][early_stopping_metric]
            if current_metric_value > self.best_val_metric:
                self.best_val_metric = current_metric_value
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (no improvement in {early_stopping_metric})")
                    break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return W_init

    def predict(self, X_train, X_val, X_test=None):
        with torch.no_grad():
            train_probs = torch.sigmoid(self.model(X_train)).cpu().numpy()
            val_probs = torch.sigmoid(self.model(X_val)).cpu().numpy()
            
            train_preds = (train_probs > cfg.THRESHOLD).astype(int)
            val_preds = (val_probs > cfg.THRESHOLD).astype(int)
            
            results = {
                'train': (train_probs, train_preds),
                'val': (val_probs, val_preds)
            }
            
            if X_test is not None:
                test_probs = torch.sigmoid(self.model(X_test)).cpu().numpy()
                test_preds = (test_probs > cfg.THRESHOLD).astype(int)
                results['test'] = (test_probs, test_preds)
            
            return results

def create_model_and_trainer(input_dim, hyp, pos_weight, mlp=True):
    if mlp:
        model = QuadMLP(input_dim)
    else:
        model = QuadraticModel(input_dim)
  
    model = model.to(hyp['device'])
    trainer = ModelTrainer(model, hyp, pos_weight)
    return model, trainer
