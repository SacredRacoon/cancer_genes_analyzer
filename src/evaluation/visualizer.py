import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

class ResultVisualizer:
    def __init__(self, path_manager):
        self.paths = path_manager

    def plot_evolution(self, history: dict):
        save_path = self.paths.plots_dir / "fitness_history.png"
        fig, ax = plt.subplots(figsize=(12,6))
        generations = range(len(history['best_fitness']))

        ax.plot(generations, history['best_fitness'], label='Best fitness', color='blue')
        ax.plot(generations,history['mean_fitness'], label='Mean fitness', color='green')
        ax.fill_between(generations,np.array(history['best_fitness']), np.array(history['mean_fitness']), color='lightblue',alpha=0.2)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness score (F1)')
        ax.set_title('Genetic Algorithm Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Evolution plot in {save_path}")

    def plot_feature_importance(self, importance_df):
        print(importance_df)
        save_path = self.paths.plots_dir / "feature_importance.png"
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
        ax.set_title('Model Feature Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot in {save_path}")

    def plot_confusion_matrix(self,y_test, y_pred):
        save_path = self.paths.plots_dir / "confusion_matrix.png"
        cm = confusion_matrix(y_test, y_pred)
        fig,ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['LGG', 'GBM'], yticklabels=['LGG','GBM'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches = 'tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")