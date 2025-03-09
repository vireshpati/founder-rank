import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, classification_report


class ModelTrainer:
    def __init__(self, model, config: Dict[str, Any], pos_weight=None):
        self.model = model.to(config["device"])
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config["device"]))
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        self.best_model_state = None
        self.best_score = float("-inf")
        # Initialize history tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        self.train_auc_history = []
        self.val_auc_history = []
        self.test_auc_history = []
        self.train_ndcg_history = []
        self.val_ndcg_history = []
        self.test_ndcg_history = []
        self.train_precision_history = []
        self.val_precision_history = []
        self.test_precision_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.test_accuracy_history = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        
        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(self.config["device"])
            y_batch = y_batch.to(self.config["device"])
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            W = self.model.get_W()
            reg_loss = self.regularization_loss(W)
            loss = self.criterion(outputs, y_batch) + reg_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        return epoch_loss / len(train_loader.dataset)

    def regularization_loss(self, W):
        diag_penalty = torch.abs(torch.diag(W)).mean() * self.config.get("diag_penalty", 0.01)
        off_diag = W - torch.diag(torch.diag(W))
        l1_off_diag = torch.abs(off_diag).mean() * self.config.get("l1_penalty", 0.001)
        
        # Encourage top-k strongest interactions
        k = self.config.get("top_k", 10)
        top_k = torch.topk(torch.abs(off_diag).view(-1), k=k)
        top_k_loss = -torch.abs(top_k.values).mean() * self.config.get("top_k_penalty", 0.01)
        
        return diag_penalty + l1_off_diag + top_k_loss

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X = X.to(self.config["device"])
        y = y.to(self.config["device"])
        
        outputs = self.model(X)
        loss = self.criterion(outputs, y).item()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        y = y.cpu().numpy()
        
        metrics = {
            'loss': loss,
            'auc': roc_auc_score(y, probs),
            'accuracy': (preds == y).mean(),
            'ndcg': self.ndcg_score(y, probs),
            'precision_at_k': self.precision_at_k(y, probs, k=10),
            'probs': probs,
            'preds': preds
        }
        
        if metrics['auc'] > self.best_score:
            self.best_score = metrics['auc']
            self.best_probs = probs
            self.best_preds = preds
            self.best_model_state = self.model.state_dict()
        
        return metrics

    def safe_corr(self, x, y, method='spearman'):
        if np.var(x) == 0 or np.var(y) == 0:
            return 0.0
        try:
            if method == 'spearman':
                return stats.spearmanr(x, y)[0]
            return np.corrcoef(x, y)[0,1]
        except:
            return 0.0

    def ndcg_score(self, true_labels, pred_scores, k=None):
        k = k or len(true_labels)
        sorted_idx = np.argsort(pred_scores)[::-1][:k]
        dcg = np.sum((2**true_labels[sorted_idx] - 1) / np.log2(np.arange(2, len(sorted_idx) + 2)))
        
        ideal_idx = np.argsort(true_labels)[::-1][:k]
        idcg = np.sum((2**true_labels[ideal_idx] - 1) / np.log2(np.arange(2, len(ideal_idx) + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0

    def precision_at_k(self, true_labels, pred_scores, k=10):
        top_k = np.argsort(pred_scores)[::-1][:k]
        return np.mean(true_labels[top_k])

    def train(self, train_loader, X_val, y_val, X_test=None, y_test=None):
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(X_val, y_val)
            
            # Track best model based on AUC (consistent with evaluate method)
            if val_metrics['auc'] > self.best_score:
                self.best_score = val_metrics['auc']
                self.best_model_state = self.model.state_dict()
                
            if epoch % self.config.get("log_every", 5) == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
                for metric, value in val_metrics.items():
                    if isinstance(value, (float, int)):  # Only print numeric metrics
                        print(f"Val {metric} = {value:.4f}")
                print("---")
