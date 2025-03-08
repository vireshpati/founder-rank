import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


class ModelTrainer:
    def __init__(self, model, config, pos_weight=None):
        self.model = model.to(config["device"])
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config["device"]))
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"] * 0.5,
            weight_decay=config["weight_decay"] * 2,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.3,
            patience=3,
            verbose=False,
        )
        self.best_val_loss = float("inf")
        self.best_val_metric = float("-inf")
        self.patience_counter = 0
        self.best_model_state = None

        self.train_loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.test_acc_history = []

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch, is_synthetic in train_loader:
            X_batch = X_batch.to(self.config["device"])
            y_batch = y_batch.to(self.config["device"])

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            # Add L1 penalty on diagonal elements
            W = self.model.get_W()
            diag_penalty = torch.abs(torch.diag(W)).mean() * 0.01
            
            loss = self.criterion(outputs, y_batch) + diag_penalty + self.interaction_loss(W)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        return epoch_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X = X.to(self.config["device"])
        y = y.to(self.config["device"])

        outputs = self.model(X)
        loss = self.criterion(outputs, y).item()
        probs = torch.sigmoid(outputs)
        
        # Move to CPU for numpy calculations
        probs = probs.cpu().numpy()
        y = y.cpu().numpy()
        
        # Calculate metrics
        acc = ((probs > 0.5) == y).mean()
        
        # Safe correlation calculation
        try:
            pearson_corr = np.corrcoef(probs, y)[0,1] if np.var(probs) > 0 and np.var(y) > 0 else 0.0
            spearman_corr = stats.spearmanr(probs, y)[0] if np.var(probs) > 0 and np.var(y) > 0 else 0.0
        except:
            pearson_corr = 0.0
            spearman_corr = 0.0
        
        # AUC-ROC (only if both classes present)
        try:
            auc_score = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
        except:
            auc_score = 0.5
        
        return {
            'loss': loss,
            'accuracy': acc,
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'auc_roc': auc_score
        }

    def interaction_loss(self, W):
        # Get off-diagonal elements
        off_diag = W - torch.diag(torch.diag(W))
        
        # Encourage sparsity in off-diagonal elements
        l1_off_diag = torch.abs(off_diag).mean()
        
        # Encourage some strong interactions
        top_k = torch.topk(torch.abs(off_diag).view(-1), k=10)
        top_k_loss = -torch.abs(top_k.values).mean()
        
        return l1_off_diag * 0.001 + top_k_loss * 0.01

    def train(self, train_loader, X_val, y_val, X_test, y_test):
        for epoch in range(self.config["epochs"]):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Get all metrics for val/test
            val_metrics = self.evaluate(X_val, y_val)
            test_metrics = self.evaluate(X_test, y_test)
            
            # Log metrics
            if epoch % 5 == 0:
                print(f"Epoch {epoch}:")
                print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_metrics['loss']:.4f}")
                if not np.isnan(val_metrics['pearson']):
                    print(f"Val Pearson = {val_metrics['pearson']:.4f}")
                if not np.isnan(val_metrics['spearman']):
                    print(f"Val Spearman = {val_metrics['spearman']:.4f}")
                if not np.isnan(val_metrics['auc_roc']):
                    print(f"Val AUC-ROC = {val_metrics['auc_roc']:.4f}")
                print("---")
            
            # Early stopping on loss instead of correlation
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_metrics['loss'])
            self.test_loss_history.append(test_metrics['loss'])
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_metrics['accuracy'])
            self.test_acc_history.append(test_metrics['accuracy'])
