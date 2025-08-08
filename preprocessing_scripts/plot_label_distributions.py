
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def plot_label_distributions(split_data_dir, figures_dir):
    """
    Plots the distribution of labels (Teff, logg, CFe, FeH) for train, val, and test sets.

    Args:
        split_data_dir (str): The root directory containing train, val, and test subdirectories.
        figures_dir (str): The directory to save the output plots.
    """
    os.makedirs(figures_dir, exist_ok=True)

    sets = ['train', 'val', 'test']
    labels_to_plot = ['Teff', 'logg', 'CFe', 'FeH']
    
    # Load data into a single dataframe with a 'set' column for easy plotting
    all_labels_df = pd.DataFrame()
    for s in sets:
        file_path = os.path.join(split_data_dir, s, 'labels.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['set'] = s
            all_labels_df = pd.concat([all_labels_df, df], ignore_index=True)
        else:
            print(f"Warning: {file_path} not found. Skipping.")

    if all_labels_df.empty:
        print("Error: No label files found to plot.")
        return

    # Plot distributions for each label
    for label in labels_to_plot:
        plt.figure(figsize=(12, 7))
        sns.histplot(data=all_labels_df, x=label, hue='set', kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {label} in Train, Validation, and Test Sets', fontsize=16)
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save the figure
        output_path = os.path.join(figures_dir, f'{label}_distribution_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot label distributions for train/val/test sets.')
    parser.add_argument('--split_data_dir', type=str, required=True, help='Directory containing the split data (train, val, test subfolders).')
    parser.add_argument('--figures_dir', type=str, required=True, help='Directory to save the output figures.')
    args = parser.parse_args()

    plot_label_distributions(args.split_data_dir, args.figures_dir)
