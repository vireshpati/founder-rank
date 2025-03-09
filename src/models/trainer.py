import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, classification_report
from src.models.quadratic import QuadMLP, QuadraticModel

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
        
        if metrics['ndcg'] > self.best_score:
            self.best_score = metrics['ndcg']
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
            
            # Track metrics history
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_metrics['loss'])
            self.test_loss_history.append(val_metrics['loss'])
            self.train_auc_history.append(val_metrics['auc'])
            self.val_auc_history.append(val_metrics['auc'])
            self.test_auc_history.append(val_metrics['auc'])
            self.train_ndcg_history.append(val_metrics['ndcg'])
            self.val_ndcg_history.append(val_metrics['ndcg'])
            self.test_ndcg_history.append(val_metrics['ndcg'])
            self.train_precision_history.append(val_metrics['precision_at_k'])
            self.val_precision_history.append(val_metrics['precision_at_k'])
            self.test_precision_history.append(val_metrics['precision_at_k'])
            self.train_accuracy_history.append(val_metrics['accuracy'])
            self.val_accuracy_history.append(val_metrics['accuracy'])
            self.test_accuracy_history.append(val_metrics['accuracy'])
            
            if epoch % self.config.get("log_every", 5) == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
                for metric, value in val_metrics.items():
                    if isinstance(value, (float, int)):  # Only print numeric metrics
                        print(f"Val {metric} = {value:.4f}")
                print("---")

    def train_with_test_tracking(self, train_loader, X_val, y_val, X_test=None, y_test=None):
   
        print(f"{'Epoch':>5} {'Train Loss':>10} {'Val Loss':>10} {'Test Loss':>10} {'Train Acc':>9} {'Val Acc':>9} "
              f"{'Test Acc':>9} {'Val AUC':>8} {'Test AUC':>8} {'Val NDCG':>8} {'Test NDCG':>8} {'P@K':>8}")
        print("-" * 120)
        
        # Save initial weights
        W_init = self.model.get_W().detach().cpu().numpy()
        
        best_val_auc = float('-inf')
        self.best_model_state = None
        
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            
            with torch.no_grad():
                train_metrics = self.evaluate(train_loader.dataset.tensors[0], train_loader.dataset.tensors[1])
                val_metrics = self.evaluate(X_val, y_val)
                
                if X_test is not None and y_test is not None:
                    test_metrics = self.evaluate(X_test, y_test)
                else:
                    test_metrics = {key: 0.0 for key in val_metrics.keys()}
            
            # Track metrics history
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_metrics['loss'])
            self.test_loss_history.append(test_metrics['loss'])
            
            self.train_accuracy_history.append(train_metrics['accuracy'])
            self.val_accuracy_history.append(val_metrics['accuracy'])
            self.test_accuracy_history.append(test_metrics['accuracy'])
            
            self.train_auc_history.append(train_metrics['auc'])
            self.val_auc_history.append(val_metrics['auc'])
            self.test_auc_history.append(test_metrics['auc'])
            
            self.train_ndcg_history.append(train_metrics['ndcg'])
            self.val_ndcg_history.append(val_metrics['ndcg'])
            self.test_ndcg_history.append(test_metrics['ndcg'])
            
            self.train_precision_history.append(train_metrics['precision_at_k'])
            self.val_precision_history.append(val_metrics['precision_at_k'])
            self.test_precision_history.append(test_metrics['precision_at_k'])
            
            if epoch % self.config.get("log_every", 1) == 0:
                print(f"{epoch:5d} {train_loss:10.4f} {val_metrics['loss']:10.4f} {test_metrics['loss']:10.4f} "
                    f"{train_metrics['accuracy']:9.4f} {val_metrics['accuracy']:9.4f} {test_metrics['accuracy']:9.4f} "
                    f"{val_metrics['auc']:8.4f} {test_metrics['auc']:8.4f} "
                    f"{val_metrics['ndcg']:8.4f} {test_metrics['ndcg']:8.4f} "
                    f"{val_metrics['precision_at_k']:8.4f}")
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                self.best_model_state = self.model.state_dict()
        
        if self.best_model_state is not None:
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
