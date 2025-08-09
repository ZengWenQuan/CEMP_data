import pandas as pd
import numpy as np
from tqdm import tqdm

def create_wavelength_grid(config):
    """Creates a single wavelength grid from a configuration list."""
    grids = []
    for start, end, step in config:
        num_points = int(np.round((end - start) / step)) + 1
        grids.append(np.linspace(start, end, num_points))
    return np.concatenate(grids)

def check_spectrum_quality(wavelengths, config, edge_threshold_points=10, internal_gap_threshold_points=10):
    """ 
    Checks the quality of a single spectrum based on its coverage within the target wavelength ranges.

    A spectrum is considered bad if, for any given sampling range, it has:
    1. A large gap at the beginning.
    2. A large gap at the end.
    3. A large internal gap.
    """
    if len(wavelengths) < 2:
        return False # Not enough data to be valid

    median_step = np.median(np.diff(wavelengths))
    if median_step <= 0:
        return False # Invalid step size

    edge_gap_threshold = edge_threshold_points * median_step
    internal_gap_threshold = internal_gap_threshold_points * median_step

    for start, end, _ in config:
        segment_mask = (wavelengths >= start) & (wavelengths <= end)
        segment_wavelengths = wavelengths[segment_mask]

        if len(segment_wavelengths) < 2:
            return False # This entire segment is missing, bad quality

        # 1. Check for gap at the beginning of the segment
        if (segment_wavelengths[0] - start) > edge_gap_threshold:
            return False

        # 2. Check for gap at the end of the segment
        if (end - segment_wavelengths[-1]) > edge_gap_threshold:
            return False

        # 3. Check for large internal gaps within the segment
        internal_diffs = np.diff(segment_wavelengths)
        if np.any(internal_diffs > internal_gap_threshold):
            return False
            
    return True # Spectrum is good

def process_step(spectra_data, wavelength_config):
    """
    Resamples spectra, performs quality control, and returns a DataFrame.
    """
    print("Creating new wavelength grid for resampling...")
    target_wavelengths = create_wavelength_grid(wavelength_config)
    print(f"New grid created with {len(target_wavelengths)} points.")

    resampled_data = []
    rejected_obsids = set()

    for spectrum in tqdm(spectra_data, desc="Resampling Denoised Spectra"):
        obsid = spectrum['obsid']
        original_wavelengths = spectrum['wavelength']
        original_flux = spectrum['flux_denoised']

        sort_indices = np.argsort(original_wavelengths)
        original_wavelengths_sorted = original_wavelengths[sort_indices]
        original_flux_sorted = original_flux[sort_indices]

        # --- Quality Control Step ---
        is_good = check_spectrum_quality(original_wavelengths_sorted, wavelength_config)
        if not is_good:
            rejected_obsids.add(obsid)
            continue

        resampled_flux = np.interp(target_wavelengths, original_wavelengths_sorted, original_flux_sorted)

        row = {'obsid': obsid}
        for i, wav in enumerate(target_wavelengths):
            row[f"{wav:.4f}"] = resampled_flux[i]
        resampled_data.append(row)

    if rejected_obsids:
        print(f"\nWarning: Rejected {len(rejected_obsids)} spectra due to data gaps.")

    if not resampled_data:
        print("\nError: No valid spectra remained after quality control. Returning empty DataFrame.")
        return pd.DataFrame()

    final_df = pd.DataFrame(resampled_data)
    
    if 'obsid' in final_df.columns:
        cols = ['obsid'] + [col for col in final_df.columns if col != 'obsid']
        final_df = final_df[cols]
    
    return final_df
