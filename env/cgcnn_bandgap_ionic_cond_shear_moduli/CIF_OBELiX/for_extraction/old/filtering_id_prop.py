import os
import pandas as pd

cif_folder = r'C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\cifs'
id_prop_path = r'C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\for_extraction\id_prop2.csv'

if not os.path.exists(id_prop_path):
    raise FileNotFoundError(f"{id_prop_path} does not exist!")

# Load your id_prop.csv
df = pd.read_csv(id_prop_path, header=None, names=['ID', 'target'])

# List of CIF files (without extension)
cif_files = {f.replace('.cif', '') for f in os.listdir(cif_folder) if f.endswith('.cif')}

# Filter df to keep only rows where ID has a matching CIF file
df_filtered = df[df['ID'].isin(cif_files)]

print(f"Before filtering: {len(df)} entries")
print(f"After filtering: {len(df_filtered)} entries")

# Save filtered CSV (overwrite or new file)
df_filtered.to_csv(id_prop_path, index=False, header=False)
