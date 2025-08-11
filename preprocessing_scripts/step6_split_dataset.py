import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
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
        if label not in df.columns or df[label].empty or df[label].isnull().all():
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

def process_step(resampled_spectra_path, labels_path, output_dir):
    """
    Main function to split the dataset and generate plots.
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Loading datasets for splitting...")
    spectra_df = pd.read_csv(resampled_spectra_path)
    labels_df = pd.read_csv(labels_path)
    print("Datasets loaded successfully.")

    master_obsids = spectra_df['obsid']
    merged_df = pd.merge(pd.DataFrame(master_obsids, columns=['obsid']), labels_df, on='obsid', how='inner')
    merged_df.dropna(subset=['FeH'], inplace=True)

    bins = [-np.inf, -4, -3, -2, -1, 0, np.inf]
    bin_labels_str = ['<-4', '-4_to_-3', '-3_to_-2', '-2_to_-1', '-1_to_0', '>0']
    merged_df['feh_bin'] = pd.cut(merged_df['FeH'], bins=bins, labels=bin_labels_str, right=True)

    train_obsids_list, val_obsids_list, test_obsids_list = [], [], []

    for bin_label in merged_df['feh_bin'].cat.categories:
        bin_df = merged_df[merged_df['feh_bin'] == bin_label]
        if bin_df.empty:
            continue
        
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

    print("\nEnsuring consistent order of samples across all files...")
    labels_df.set_index('obsid', inplace=True)
    spectra_df.set_index('obsid', inplace=True)

    # Filter, reorder, and save train set
    if train_obsids_list:
        train_labels = labels_df.loc[train_obsids_list].reset_index()
        train_spectra = spectra_df.loc[train_obsids_list].reset_index()
        plot_and_save_distributions(train_labels, 'train', train_dir)
        train_spectra.to_csv(os.path.join(train_dir, 'spectra.csv'), index=False)
        train_labels.to_csv(os.path.join(train_dir, 'labels.csv'), index=False)

    # Filter, reorder, and save validation set
    if val_obsids_list:
        val_labels = labels_df.loc[val_obsids_list].reset_index()
        val_spectra = spectra_df.loc[val_obsids_list].reset_index()
        plot_and_save_distributions(val_labels, 'val', val_dir)
        val_spectra.to_csv(os.path.join(val_dir, 'spectra.csv'), index=False)
        val_labels.to_csv(os.path.join(val_dir, 'labels.csv'), index=False)

    # Filter, reorder, and save test set
    if test_obsids_list:
        test_labels = labels_df.loc[test_obsids_list].reset_index()
        test_spectra = spectra_df.loc[test_obsids_list].reset_index()
        plot_and_save_distributions(test_labels, 'test', test_dir)
        test_spectra.to_csv(os.path.join(test_dir, 'spectra.csv'), index=False)
        test_labels.to_csv(os.path.join(test_dir, 'labels.csv'), index=False)
    
    print("\nAll files saved successfully.")