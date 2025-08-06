
"""
步骤 2: 红移校正
"""
import pandas as pd

def process_step(spectra_data, redshift_file):
    df_z = pd.read_csv(redshift_file)
    redshift_map = {row['obsid']: row['z'] for _, row in df_z.iterrows()}
    corrected_spectra = []
    total_spectra = len(spectra_data)
    print(f"开始对 {total_spectra} 个光谱进行红移校正...")
    for i, spec in enumerate(spectra_data):
        if (i + 1) % 10 == 0:
            print(f"  正在处理光谱 {i + 1}/{total_spectra}...")
        obsid = int(spec['obsid'])
        if obsid in redshift_map:
            z = redshift_map[obsid]
            corrected_spec = spec.copy()
            corrected_spec['wavelength_rest'] = spec['wavelength'] / (1 + z)
            corrected_spec['z'] = z
            corrected_spectra.append(corrected_spec)
    print("红移校正完成！")
    return corrected_spectra
