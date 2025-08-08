import zipfile

cifs_zip = "all_cifs.zip"
extract_dir = r'C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\for_extraction\cifs_obelix'


with zipfile.ZipFile(cifs_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extraction complete")