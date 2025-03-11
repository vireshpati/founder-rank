import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import torch


def plot_model_performance(trainer, W_init):
    plt.style.use('seaborn-v0_8-dark')
    
    metrics = [
        ('Training Loss', [
            (trainer.metrics_history['train']['loss'], 'Train', '#2ecc71'),
            (trainer.metrics_history['val']['loss'], 'Val', '#3498db'),
            (trainer.metrics_history['test']['loss'], 'Test', '#e74c3c')
        ], 'Loss'),
        
        ('Classification Accuracy', [
            (trainer.metrics_history['train']['accuracy'], 'Train', '#2ecc71'),
            (trainer.metrics_history['val']['accuracy'], 'Val', '#3498db'),
            (trainer.metrics_history['test']['accuracy'], 'Test', '#e74c3c')
        ], 'Accuracy'),
        
        ('ROC-AUC Score', [
            (trainer.metrics_history['train']['auc'], 'Train', '#2ecc71'),
            (trainer.metrics_history['val']['auc'], 'Val', '#3498db'),
            (trainer.metrics_history['test']['auc'], 'Test', '#e74c3c')
        ], 'AUC'),
        
        ('NDCG Score', [
            (trainer.metrics_history['train']['ndcg'], 'Train', '#2ecc71'),
            (trainer.metrics_history['val']['ndcg'], 'Val', '#3498db'),
            (trainer.metrics_history['test']['ndcg'], 'Test', '#e74c3c')
        ], 'NDCG'),
        
        ('Precision@K', [
            (trainer.metrics_history['train']['precision_at_k'], 'Train', '#2ecc71'),
            (trainer.metrics_history['val']['precision_at_k'], 'Val', '#3498db'),
            (trainer.metrics_history['test']['precision_at_k'], 'Test', '#e74c3c')
        ], 'P@K')
    ]
    
    epochs = range(len(trainer.metrics_history['train']['loss']))
    
    # Plot performance metrics
    for title, data_series, ylabel in metrics:
        plt.figure(figsize=(12, 6))
        for data, label, color in data_series:
            plt.plot(epochs, data, label=label, color=color, linewidth=2)
            
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    # Plot weight matrices
    W_final = trainer.model.get_W().detach().cpu().numpy()
    W_diff = W_final - W_init
    
    matrices = [
        ('Initial Weight Matrix', W_init),
        ('Final Weight Matrix', W_final),
        ('Weight Change (Final - Initial)', W_diff)
    ]
    
    for title, matrix in matrices:
        plt.figure(figsize=(10, 8))
        vmax = abs(matrix).max()
        im = plt.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        plt.colorbar(im)
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Feature Index', fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_ranked_feature_importance(W, feature_names, figsize=(15, 10)):
    plt.style.use('seaborn-v0_8-dark')
    
    # Get diagonal importance (individual feature effects)
    diag_importance = np.diag(W)  
    diag_ranks = pd.DataFrame({
        'Feature': feature_names,
        'Importance': diag_importance,
        'Type': 'Individual'
    })
    
    # Get interaction importance (sum of off-diagonal elements, preserving sign)
    interaction_importance = []
    for i, feat in enumerate(feature_names):
        # Sum off-diagonal elements for each feature, preserving sign
        importance = np.sum(W[i, :]) + np.sum(W[:, i]) - 2*W[i,i]
        interaction_importance.append(importance)
    
    interaction_ranks = pd.DataFrame({
        'Feature': feature_names,
        'Importance': interaction_importance,
        'Type': 'Interaction'
    })
    
    # Combine and normalize while preserving sign
    all_ranks = pd.concat([diag_ranks, interaction_ranks])
    all_ranks['Importance'] = all_ranks.groupby('Type')['Importance'].transform(
        lambda x: x / np.max(np.abs(x)) 
    )
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual importance
    individual = all_ranks[all_ranks['Type'] == 'Individual'].sort_values('Importance')
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in individual['Importance']]
    ax1.barh(range(len(individual)), individual['Importance'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(individual)))
    ax1.set_yticklabels(individual['Feature'], fontsize=10)
    ax1.set_title('Individual Feature Importance', fontsize=14, pad=15)
    ax1.set_xlabel('Normalized Importance', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Interaction importance  
    interaction = all_ranks[all_ranks['Type'] == 'Interaction'].sort_values('Importance')
    colors = ['#e74c3c' if x < 0 else '#3498db' for x in interaction['Importance']]
    ax2.barh(range(len(interaction)), interaction['Importance'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(interaction)))
    ax2.set_yticklabels(interaction['Feature'], fontsize=10)
    ax2.set_title('Feature Interaction Importance', fontsize=14, pad=15)
    ax2.set_xlabel('Normalized Importance', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(W, feature_names, figsize=(12, 10)):
    plt.style.use('seaborn-v0_8-dark')
    
    mask = np.zeros_like(W, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = np.abs(W).max()
    sns.heatmap(W, 
                mask=mask,
                cmap='coolwarm',
                center=0,
                vmin=-vmax,
                vmax=vmax,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                xticklabels=feature_names,
                yticklabels=feature_names,
                ax=ax,
                annot=True,  
                fmt='.2f',   
                annot_kws={'size': 8})  
    
    plt.title('Feature Interaction Importance', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_sample_predictions(model, X, y, matrix, scaler, n_samples=5, device='cuda'):
    """Plot sample predictions with feature importance visualization"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        logits = model(X_tensor).cpu().numpy().flatten()
        probs = 1 / (1 + np.exp(-logits))  # Manual sigmoid to ensure proper calculation
    
    # Select a mix of correct and incorrect predictions
    pred_y = probs >= 0.5
    correct_idx = np.where(pred_y == y)[0]
    incorrect_idx = np.where(pred_y != y)[0]
    
    # Try to get a balanced sample of correct and incorrect predictions
    n_correct = min(n_samples // 2, len(correct_idx))
    n_incorrect = min(n_samples - n_correct, len(incorrect_idx))
    
    if n_correct > 0:
        selected_correct = np.random.choice(correct_idx, size=n_correct, replace=False)
    else:
        selected_correct = []
        
    if n_incorrect > 0:
        selected_incorrect = np.random.choice(incorrect_idx, size=n_incorrect, replace=False)
    else:
        selected_incorrect = []
    
    # Fill remaining slots if needed
    remaining = n_samples - (n_correct + n_incorrect)
    if remaining > 0:
        remaining_pool = np.setdiff1d(np.arange(len(X)), np.concatenate([selected_correct, selected_incorrect]))
        if len(remaining_pool) > 0:
            selected_remaining = np.random.choice(remaining_pool, size=remaining, replace=False)
            selected_idx = np.concatenate([selected_correct, selected_incorrect, selected_remaining])
        else:
            selected_idx = np.concatenate([selected_correct, selected_incorrect])
    else:
        selected_idx = np.concatenate([selected_correct, selected_incorrect])
    
    X_original = scaler.inverse_transform(X)
    
    # Process categories and create mapping dictionaries
    categories = list(matrix.keys())
    category_values = np.zeros((len(X), len(categories)))
    start_idx = 0
    value_mappings = {}
    
    for i, cat in enumerate(categories):
        dim = matrix[cat]["DIMENSION"]
        one_hot_section = X_original[:, start_idx:start_idx + dim]
        category_values[:, i] = np.argmax(one_hot_section, axis=1)
        
        # Create mapping for this category
        if "VALUES" in matrix[cat]:
            value_mappings[cat] = matrix[cat]["VALUES"]
        else:
            value_mappings[cat] = list(range(dim))
        
        start_idx += dim
    
    # Create visualization
    fig = plt.figure(figsize=(20, n_samples * 1.2 + 0.5))
    gs = gridspec.GridSpec(n_samples + 1, 2, width_ratios=[4, 1], height_ratios=[*[1]*n_samples, 0.2])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    custom_cmap = mcolors.ListedColormap(colors)
    
    for i, idx in enumerate(selected_idx):
        # Feature barcode
        ax1 = plt.subplot(gs[i, 0])
        im = ax1.imshow(category_values[idx].reshape(1, -1),
                       cmap=custom_cmap,
                       aspect='auto',
                       interpolation='nearest')
        
        # Add category labels
        if i == len(selected_idx) - 1:
            ax1.set_xticks(range(len(categories)))
            ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
        else:
            ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Add grid lines
        for x in range(len(categories)-1):
            ax1.axvline(x + 0.5, color='white', linewidth=2)
        
        # Add value labels directly on the barcode
        for j, cat in enumerate(categories):
            val_idx = int(category_values[idx, j])
            val = value_mappings[cat][val_idx]
            ax1.text(j, 0, str(val), ha='center', va='center', 
                    color='white', fontsize=9, fontweight='bold')
        
        # Prediction visualization
        ax2 = plt.subplot(gs[i, 1])
        correct = pred_y[idx] == y[idx]
        color = '#2ecc71' if correct else '#e74c3c'
        ax2.barh([0], [probs[idx]], color=color, alpha=0.8)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.set_xticks([0, 0.5, 1])
        
        if i == len(selected_idx) - 1:
            ax2.set_xlabel('Prediction Probability', fontsize=10)
        
        # Add prediction labels
        label_color = 'darkgreen' if correct else 'darkred'
        ax2.text(1.05, 0, f'Pred: {int(pred_y[idx])}', va='center', fontsize=10, color=label_color)
        ax2.text(1.25, 0, f'True: {int(y[idx])}', va='center', fontsize=10)
        
        # Add confidence percentage
        conf_text = f'{probs[idx]*100:.1f}%'
        ax2.text(probs[idx]/2, 0, conf_text, ha='center', va='center', color='white', fontsize=9)
    
    # Add legend for category values
    ax_legend = plt.subplot(gs[-1, :])
    ax_legend.axis('off')
    legend_text = ""
    for cat in categories:
        legend_text += f"{cat}: {value_mappings[cat]}\n"
    ax_legend.text(0, 0.5, "Category Values:\n" + legend_text, 
                  fontsize=10, va='center', ha='left')
    
    plt.tight_layout()
    plt.show()
    
def display_final_metrics(trainer):
    """Display final metrics in a tabular format with numerical values"""
    # Get the final metrics
    final_metrics = {
        'Train': {
            'NDCG': trainer.metrics_history['train']['ndcg'][-1],
            'P@K': trainer.metrics_history['train']['precision_at_k'][-1],
            'Accuracy': trainer.metrics_history['train']['accuracy'][-1],
            'AUC': trainer.metrics_history['train']['auc'][-1]
        },
        'Validation': {
            'NDCG': trainer.metrics_history['val']['ndcg'][-1],
            'P@K': trainer.metrics_history['val']['precision_at_k'][-1],
            'Accuracy': trainer.metrics_history['val']['accuracy'][-1],
            'AUC': trainer.metrics_history['val']['auc'][-1]
        },
        'Test': {
            'NDCG': trainer.metrics_history['test']['ndcg'][-1],
            'P@K': trainer.metrics_history['test']['precision_at_k'][-1],
            'Accuracy': trainer.metrics_history['test']['accuracy'][-1],
            'AUC': trainer.metrics_history['test']['auc'][-1]
        }
    }
    
    # Create a DataFrame for better display
    metrics_df = pd.DataFrame(final_metrics)
    
    # Create a bar chart for visual comparison
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', rot=0, figsize=(12, 6))
    plt.title('Final Model Performance Metrics', fontsize=14, pad=15)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Dataset')
    
    # Add value labels on top of bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_top_feature_interactions(W, feature_names, top_k=10, exclude_same_category=True):
    """Plot top positive and negative feature interactions, excluding same-category combinations."""
    plt.style.use('seaborn-v0_8-dark')
    
    # Create pairs of features and their interaction strengths
    n = len(feature_names)
    interactions = []
    
    for i in range(n):
        for j in range(i+1, n):
            # Extract category names from feature names (assuming format "CATEGORY_X")
            cat1 = feature_names[i].split('_')[0]
            cat2 = feature_names[j].split('_')[0]
            
            # Skip if features are from same category and we want to exclude those
            if exclude_same_category and cat1 == cat2:
                continue
                
            interaction_strength = W[i,j] + W[j,i]  # Symmetric interaction strength
            interactions.append({
                'pair': f"{feature_names[i]} × {feature_names[j]}",
                'strength': interaction_strength,
                'abs_strength': abs(interaction_strength)
            })
    
    # Sort by absolute strength and get top_k positive and negative
    interactions.sort(key=lambda x: x['strength'])
    neg_interactions = interactions[:top_k]
    interactions.sort(key=lambda x: x['strength'], reverse=True)
    pos_interactions = interactions[:top_k]
    
    # Combine and plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Positive interactions
    strengths = [x['strength'] for x in pos_interactions]
    pairs = [x['pair'] for x in pos_interactions]
    ax1.barh(range(len(strengths)), strengths, color='#2ecc71', alpha=0.7)
    ax1.set_yticks(range(len(pairs)))
    ax1.set_yticklabels(pairs, fontsize=10)
    ax1.set_title('Top Positive Feature Interactions', fontsize=14, pad=15)
    ax1.set_xlabel('Interaction Strength', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Negative interactions
    strengths = [x['strength'] for x in neg_interactions]
    pairs = [x['pair'] for x in neg_interactions]
    ax2.barh(range(len(strengths)), strengths, color='#e74c3c', alpha=0.7)
    ax2.set_yticks(range(len(pairs)))
    ax2.set_yticklabels(pairs, fontsize=10)
    ax2.set_title('Top Negative Feature Interactions', fontsize=14, pad=15)
    ax2.set_xlabel('Interaction Strength', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
