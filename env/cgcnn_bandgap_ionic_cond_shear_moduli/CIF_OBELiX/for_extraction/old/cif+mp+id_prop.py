import os
import pandas as pd
import time
from mp_api.client import MPRester
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import defaultdict
import shutil
import logging


# --- Configuration ---

API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"  # Your Materials Project API key

OBELIX_XLSX_PATH = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\for_extraction\OBELiX_data.xlsx"

CIF_FOLDER = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\cifs"

OUTPUT_CIF_FOLDER = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\combined_cifs"

ID_PROP_CSV = "id_prop2.csv"

DELAY_BETWEEN_REQUESTS = 0.2  # seconds (adjust based on API rate limits if needed)


# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Load OBELiX dataset ---

logger.info(f"Loading OBELiX dataset from {OBELIX_XLSX_PATH}...")
df = pd.read_excel(OBELIX_XLSX_PATH)

# Normalize OBELiX IDs
df['ID'] = df['ID'].astype(str).str.strip().str.lower()

# Convert ionic conductivity column to numeric, drop NaNs
cond_col = 'Ionic conductivity (S cm-1)'
df[cond_col] = pd.to_numeric(df[cond_col], errors='coerce')
df = df.dropna(subset=[cond_col])

# Normalize OBELiX composition using pymatgen Composition with corrected property access
df['Norm_Composition'] = df['Composition'].apply(lambda x: Composition(x).reduced_formula if pd.notna(x) else "")

# Normalize OBELiX space group number (integer)
df['Space group number'] = pd.to_numeric(df['Space group number'], errors='coerce').fillna(0).astype(int)

# Convert OBELiX dataframe rows into a list of dicts for quick access
obelix_data = []
for _, row in df.iterrows():
    obelix_data.append({
        'ID': row['ID'],
        'Norm_Composition': row['Norm_Composition'],
        'SpaceGroup': row['Space group number'],
        'IonicConductivity': row[cond_col]
    })


# --- Parse CIF files metadata ---

def parse_cif_metadata(cif_path):
    """Parse CIF file to get composition, space group number, and structure."""
    try:
        struct = Structure.from_file(cif_path)
        sg_num = SpacegroupAnalyzer(struct).get_space_group_number()
        comp = struct.composition.reduced_formula  # property, not method
        return comp, sg_num, struct
    except Exception as e:
        logger.warning(f"Failed to parse {cif_path}: {e}")
        return None, None, None


logger.info(f"Parsing CIF files in {CIF_FOLDER} ...")
cif_metadata = {}
for f in os.listdir(CIF_FOLDER):
    if f.lower().endswith('.cif'):
        path = os.path.join(CIF_FOLDER, f)
        comp, sg, struct = parse_cif_metadata(path)
        if comp and sg:
            cif_metadata[f[:-4].lower()] = {'composition': comp, 'spacegroup': sg, 'filepath': path, 'structure': struct}
logger.info(f"Parsed {len(cif_metadata)} CIF files")


# --- Matching CIFs to OBELiX entries ---

def compositions_close(comp_a, comp_b, tolerance=0.05):
    """
    Check if two pymatgen compositions are close by comparing atomic fractions of elements.
    """
    try:
        a = Composition(comp_a)
        b = Composition(comp_b)
    except Exception:
        return False

    elements = set(a.elements) | set(b.elements)
    for el in elements:
        frac_a = a.get_atomic_fraction(el)
        frac_b = b.get_atomic_fraction(el)
        if abs(frac_a - frac_b) > tolerance:
            return False
    return True


# Build inverted index for OBELiX by (space group, normalized composition)
obelix_index = defaultdict(list)
for entry in obelix_data:
    key = (entry['SpaceGroup'], entry['Norm_Composition'])
    obelix_index[key].append(entry)

matched_pairs = []  # (cif_filename, obelix_id, ionic_conductivity)

logger.info("Matching CIF files to OBELiX entries by composition and space group...")

for cif_id, meta in cif_metadata.items():
    matched = False
    key = (meta['spacegroup'], meta['composition'])
    candidates = obelix_index.get(key, [])

    if candidates:
        # If multiple candidates, pick first closest by composition
        for candidate in candidates:
            if compositions_close(meta['composition'], candidate['Norm_Composition']):
                matched_pairs.append((cif_id, candidate['ID'], candidate['IonicConductivity']))
                matched = True
                break

    if not matched:
        # Looser matching: any obelix entry with same spacegroup and close composition
        possible_candidates = [e for e in obelix_data if e['SpaceGroup'] == meta['spacegroup']]
        for candidate in possible_candidates:
            if compositions_close(meta['composition'], candidate['Norm_Composition']):
                matched_pairs.append((cif_id, candidate['ID'], candidate['IonicConductivity']))
                matched = True
                break

    if not matched:
        # No match found for this CIF - optionally log here if needed
        logger.debug(f"No OBELiX match found for CIF file '{cif_id}'")


logger.info(f"Total matched CIF files: {len(matched_pairs)}")


# --- Save matched CIF files to output folder ---

os.makedirs(OUTPUT_CIF_FOLDER, exist_ok=True)
saved_files = set()

for cif_id, obelix_id, _ in matched_pairs:
    src_path = cif_metadata[cif_id]['filepath']
    dst_path = os.path.join(OUTPUT_CIF_FOLDER, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
        saved_files.add(cif_id)
        logger.info(f"Copied {os.path.basename(src_path)} to '{OUTPUT_CIF_FOLDER}'")
    else:
        logger.debug(f"File {os.path.basename(src_path)} already exists in output folder")

logger.info(f"Copied {len(saved_files)} CIF files to '{OUTPUT_CIF_FOLDER}'")


# --- Generate id_prop.csv ---

# Save ALL matched pairs using CIF filename as ID to avoid losing duplicates
with open(ID_PROP_CSV, 'w') as f:
    for cif_id, obelix_id, ionic_cond in matched_pairs:
        f.write(f"{cif_id},{ionic_cond}\n")

logger.info(f"Saved {ID_PROP_CSV} with {len(matched_pairs)} entries")
