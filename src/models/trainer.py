import torch
import torch.nn as nn

class ModelTrainer:
    def __init__(self, model, config, pos_weight=None):
        self.model = model.to(config['device'])
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config['device']))
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.best_val_loss = float('inf')
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
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.config['device'])
            y_batch = y_batch.to(self.config['device'])
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
        return epoch_loss / total, correct / total
    
    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X = X.to(self.config['device'])
        y = y.to(self.config['device'])
        
        outputs = self.model(X)
        loss = self.criterion(outputs, y).item()
        acc = ((outputs > 0).float() == y).float().mean().item()
        
        return loss, acc
    
    def train(self, train_loader, X_val, y_val, X_test, y_test):
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            test_loss, test_acc = self.evaluate(X_test, y_test)
            
            self.scheduler.step(val_loss)
            
            # early stop
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['patience']:
                print(f'Early stopping at epoch {epoch}')
                # Restore best model
                self.model.load_state_dict(self.best_model_state)
                break
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.test_loss_history.append(test_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self.test_acc_history.append(test_acc)
            
            self.scheduler.step(val_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: "
                    f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Test Loss = {test_loss:.4f}, "
                    f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")
