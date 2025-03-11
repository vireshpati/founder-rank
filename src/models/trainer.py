import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, Any
from sklearn.metrics import roc_auc_score
from src.models.quadratic import QuadMLP, QuadraticModel
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

class ModelTrainer:
    def __init__(self, model, config: Dict[str, Any], pos_weight=None):
        self.model = model.to(config["device"])
        self.config = config
        self.criterion = FocalLoss(alpha=2.0, gamma=2.0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01)
        
        # State tracking
        self.best_model_state = None
        self.best_val_metric = float("-inf")
        self.metrics_history = {
            'train': {'loss': [], 'auc': [], 'ndcg': [], 'precision_at_k': [], 'accuracy': []},
            'val': {'loss': [], 'auc': [], 'ndcg': [], 'precision_at_k': [], 'accuracy': []},
            'test': {'loss': [], 'auc': [], 'ndcg': [], 'precision_at_k': [], 'accuracy': []}
        }

        print(f"{'Epoch':>5} {'Train Loss':>10} {'Val Loss':>10} {'Test Loss':>10} "
              f"{'Val AUC':>8} {'Val NDCG':>8} {'Val P@K':>8} {'Val Acc':>8} "
              f"{'Test AUC':>8} {'Test NDCG':>8} {'Test P@K':>8} {'Test Acc':>8}")
        print("-" * 120)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X, y, _ in train_loader:
            X, y = X.to(self.config["device"]), y.to(self.config["device"])
            
            # Apply label smoothing
            y_smooth = self.smooth_labels(y, smoothing=0.03)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y_smooth)
            
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
        preds = (probs > 0.5).astype(int)
        y = y.cpu().numpy()
        
        return {
            'loss': loss,
            'auc': roc_auc_score(y, probs),
            'accuracy': (preds == y).mean(),
            'ndcg': self.ndcg_score(y, probs),
            'precision_at_k': self.precision_at_k(y, probs, k=10),
            'probs': probs,
            'preds': preds
        }

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

    def smooth_labels(self, labels, smoothing=0.03):
        labels = labels.float().to(self.config["device"])
        return labels * (1 - smoothing) + 0.5 * smoothing

    def train(self, train_loader, X_val, y_val, X_test=None, y_test=None):
        W_init = self.model.get_W().detach().cpu().numpy()
        patience = self.config.get("early_stopping_patience", 50)
        patience_counter = 0
        
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
                print(f"{epoch:5d} {train_loss:10.4f} {metrics['val']['loss']:10.4f} "
                      f"{metrics['test']['loss'] if metrics['test'] else 0:10.4f} "
                      f"{metrics['val']['auc']:8.4f} {metrics['val']['ndcg']:8.4f} "
                      f"{metrics['val']['precision_at_k']:8.4f} {metrics['val']['accuracy']:8.4f} "
                      f"{metrics['test']['auc'] if metrics['test'] else 0:8.4f} "
                      f"{metrics['test']['ndcg'] if metrics['test'] else 0:8.4f} "
                      f"{metrics['test']['precision_at_k'] if metrics['test'] else 0:8.4f} "
                      f"{metrics['test']['accuracy'] if metrics['test'] else 0:8.4f}")
            
            # Early stopping on validation NDCG
            if metrics['val']['ndcg'] > self.best_val_metric:
                self.best_val_metric = metrics['val']['ndcg']
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return W_init

    def predict(self, X_train, X_val, X_test=None):
        with torch.no_grad():
            train_probs = torch.sigmoid(self.model(X_train)).cpu().numpy()
            val_probs = torch.sigmoid(self.model(X_val)).cpu().numpy()
            
            train_preds = (train_probs > 0.5).astype(int)
            val_preds = (val_probs > 0.5).astype(int)
            
            results = {
                'train': (train_probs, train_preds),
                'val': (val_probs, val_preds)
            }
            
            if X_test is not None:
                test_probs = torch.sigmoid(self.model(X_test)).cpu().numpy()
                test_preds = (test_probs > 0.5).astype(int)
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
