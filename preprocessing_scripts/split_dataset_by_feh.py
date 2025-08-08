import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import matplotlib.pyplot as plt

def plot_and_save_distributions(df, set_name, output_dir):
    """Plots and saves the distribution for each specified label in the dataframe."""
    try:
        import seaborn as sns
    except ImportError:
        print("Error: Seaborn library is required for plotting. Please install it using 'pip install seaborn'")
        return

    labels_to_plot = ['Teff', 'logg', 'CFe', 'FeH']
    print(f"Plotting distributions for {set_name} set...")
    for label in labels_to_plot:
        if label not in df.columns:
            print(f"Warning: Label '{label}' not found in {set_name} set. Skipping plot.")
            continue
        if df[label].empty or df[label].isnull().all():
            print(f"Warning: Label '{label}' in {set_name} set is empty or all NaN. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(df[label], kde=True, bins=30)
        plt.title(f'Distribution of {label} in {set_name.capitalize()} Set', fontsize=16)
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plot_path = os.path.join(output_dir, f'{label}_distribution.png')
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  -> Saved {label} distribution plot to {plot_path}")
        except Exception as e:
            print(f"Could not save plot {plot_path}. Error: {e}")
        plt.close()

def split_dataset_by_feh(continuum_path, normalized_path, labels_path, output_dir):
    """
    Splits the dataset and plots label distributions for each subset.
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Loading datasets...")
    continuum_df = pd.read_csv(continuum_path)
    normalized_df = pd.read_csv(normalized_path)
    labels_df = pd.read_csv(labels_path)
    print("Datasets loaded successfully.")

    merged_df = pd.merge(continuum_df[['obsid']], labels_df, on='obsid', how='left')
    merged_df.dropna(subset=['FeH'], inplace=True)

    bins = [-np.inf, -4, -3, -2, -1, 0, np.inf]
    bin_labels_str = ['<-4', '-4_to_-3', '-3_to_-2', '-2_to_-1', '-1_to_0', '>0']
    merged_df['feh_bin'] = pd.cut(merged_df['FeH'], bins=bins, labels=bin_labels_str, right=True)

    train_obsids_list, val_obsids_list, test_obsids_list = [], [], []

    for bin_label in merged_df['feh_bin'].unique():
        bin_df = merged_df[merged_df['feh_bin'] == bin_label]
        if len(bin_df) < 3:
            train_obsids_list.extend(bin_df['obsid'].tolist())
            continue
        
        train_bin, temp_bin = train_test_split(bin_df, test_size=0.2, random_state=42, stratify=bin_df['feh_bin'])
        
        if len(temp_bin) < 2:
            train_obsids_list.extend(train_bin['obsid'].tolist())
            val_obsids_list.extend(temp_bin['obsid'].tolist())
            continue

        val_bin, test_bin = train_test_split(temp_bin, test_size=0.5, random_state=42, stratify=temp_bin['feh_bin'])
        
        train_obsids_list.extend(train_bin['obsid'].tolist())
        val_obsids_list.extend(val_bin['obsid'].tolist())
        test_obsids_list.extend(test_bin['obsid'].tolist())

    print(f"Train set size: {len(train_obsids_list)}")
    print(f"Validation set size: {len(val_obsids_list)}")
    print(f"Test set size: {len(test_obsids_list)}")

    # --- Ensure consistent ordering across all files for each set ---
    print("\nEnsuring consistent order of samples across all files...")
    
    # Filter and Sort Train Set
    train_labels = labels_df[labels_df['obsid'].isin(train_obsids_list)].set_index('obsid').loc[train_obsids_list].reset_index()
    train_continuum = continuum_df[continuum_df['obsid'].isin(train_obsids_list)].set_index('obsid').loc[train_obsids_list].reset_index()
    train_normalized = normalized_df[normalized_df['obsid'].isin(train_obsids_list)].set_index('obsid').loc[train_obsids_list].reset_index()

    # Filter and Sort Validation Set
    val_labels = labels_df[labels_df['obsid'].isin(val_obsids_list)].set_index('obsid').loc[val_obsids_list].reset_index()
    val_continuum = continuum_df[continuum_df['obsid'].isin(val_obsids_list)].set_index('obsid').loc[val_obsids_list].reset_index()
    val_normalized = normalized_df[normalized_df['obsid'].isin(val_obsids_list)].set_index('obsid').loc[val_obsids_list].reset_index()

    # Filter and Sort Test Set
    test_labels = labels_df[labels_df['obsid'].isin(test_obsids_list)].set_index('obsid').loc[test_obsids_list].reset_index()
    test_continuum = continuum_df[continuum_df['obsid'].isin(test_obsids_list)].set_index('obsid').loc[test_obsids_list].reset_index()
    test_normalized = normalized_df[normalized_df['obsid'].isin(test_obsids_list)].set_index('obsid').loc[test_obsids_list].reset_index()

    # Plot distributions for each set
    plot_and_save_distributions(train_labels, 'train', train_dir)
    plot_and_save_distributions(val_labels, 'val', val_dir)
    plot_and_save_distributions(test_labels, 'test', test_dir)

    # Save CSV files
    print("\nSaving split CSV files...")
    train_continuum.to_csv(os.path.join(train_dir, 'continuum.csv'), index=False)
    val_continuum.to_csv(os.path.join(val_dir, 'continuum.csv'), index=False)
    test_continuum.to_csv(os.path.join(test_dir, 'continuum.csv'), index=False)
    train_normalized.to_csv(os.path.join(train_dir, 'normalized.csv'), index=False)
    val_normalized.to_csv(os.path.join(val_dir, 'normalized.csv'), index=False)
    test_normalized.to_csv(os.path.join(test_dir, 'normalized.csv'), index=False)
    train_labels.to_csv(os.path.join(train_dir, 'labels.csv'), index=False)
    val_labels.to_csv(os.path.join(val_dir, 'labels.csv'), index=False)
    test_labels.to_csv(os.path.join(test_dir, 'labels.csv'), index=False)
    
    print("All files saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset and plot label distributions.')
    parser.add_argument('--continuum_path', type=str, required=True, help='Path to the continuum spectra CSV file.')
    parser.add_argument('--normalized_path', type=str, required=True, help='Path to the normalized spectra CSV file.')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files and plots.')
    args = parser.parse_args()

    if not all(os.path.exists(p) for p in [args.continuum_path, args.normalized_path, args.labels_path]):
        print("Error: One or more input files not found.")
    else:
        split_dataset_by_feh(args.continuum_path, args.normalized_path, args.labels_path, args.output_dir)
