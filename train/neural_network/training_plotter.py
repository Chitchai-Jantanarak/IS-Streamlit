import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import argparse

def load_data(json_file) -> pd.DataFrame :
    with open(json_file, 'r') as file :
        data = json.load(file)
    return pd.DataFrame( data['training_metrics'] )

def plotter(df, output_file, dpi = 300) -> str :
    sns.set_style("whitegrid")

    # Fig Grid- plot layout (row = 2, col = 1)
    flg, axes = plt.subplots(2, 1, figsize = (12, 14))

    # Plot 1: Acc & loss
    # map(epoch, _ => {.trainAccuracy, .valAccuracy, .trainLoss, .valLoss} )
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['trainAccuracy'], 'o-', linewidth=2, label='Training Accuracy', color='#3498db')
    ax1.plot(df['epoch'], df['valAccuracy'], 's-', linewidth=2, label='Validation Accuracy', color='#2ecc71')
    
    ax1_loss = ax1.twinx() ###### REF
    ax1_loss.plot(df['epoch'], df['trainLoss'], '--', linewidth=2, label='Training Loss', color='#e74c3c')
    ax1_loss.plot(df['epoch'], df['valLoss'], '--', linewidth=2, label='Validation Loss', color='#f39c12')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1_loss.set_ylabel('Loss')
    ax1.set_title('Training flow Accuracy and Loss', fontsize=16)

    # to_center
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_loss.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    # Plot 2 : Precision & Recall
    # map(epoch, _ => {.trainiPrecision, .valPrecision, .trainRecall, .valRecall} )
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['trainPrecision'], 'o-', linewidth=2, label='Training Precision', color='#8e44ad')
    ax2.plot(df['epoch'], df['valPrecision'], 's-', linewidth=2, label='Validation Precision', color='#9b59b6')
    ax2.plot(df['epoch'], df['trainRecall'], 'o-', linewidth=2, label='Training Recall', color='#1abc9c')
    ax2.plot(df['epoch'], df['valRecall'], 's-', linewidth=2, label='Validation Recall', color='#16a085')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_title('Model Precision and Recall During Training', fontsize=16)
    ax2.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to {output_file} with DPI {dpi}")
    
def main():
    json_path   = './training_data.json'
    output_path = 'training_flows.png'

    try :
        df = load_data(json_file = json_path)
        plotter(df, output_path)
    except Exception as e :
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__" :
    main()