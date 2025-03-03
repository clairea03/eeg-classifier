# Visualization functions


import numpy as np
import matplotlib.pyplot as plt


def visualize_results(results, conf_matrices, save_dir=None):
   
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison for EEG Classification')
    plt.ylim(0, 1)
    
    if save_dir:
        plt.savefig(f"{save_dir}/model_comparison.png")
    
    plt.show()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, len(conf_matrices), figsize=(15, 5))
    
    for i, (name, cm) in enumerate(conf_matrices.items()):
        ax = axes[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix - {name}')
        plt.colorbar(im, ax=ax)
        
        # Label axes
        classes = ['Left Hand', 'Right Hand']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/confusion_matrices.png")
    
    plt.show()