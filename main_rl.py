import os
import sys
import torch
import pandas as pd
import numpy as np


from env.sei_predictor import SEIPredictor
from env.cei_predictor import CEIPredictor
from env.cgcnn_bandgap_ionic_cond_shear_moduli.cgcnn_pretrained import cgcnn_predict


from env.cgcnn_bandgap_ionic_cond_shear_moduli.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.cgcnn_bandgap_ionic_cond_shear_moduli.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.cgcnn_bandgap_ionic_cond_shear_moduli.main import Normalizer


print("Running main_rl.py:", __file__)


def run_sei_prediction(cif_file_path: str):
    predictor = SEIPredictor()
    results = predictor.predict_from_cif(cif_file_path)
    return results


def run_cei_prediction(cif_file_path: str):
    predictor = CEIPredictor()
    results = predictor.predict_from_cif(cif_file_path)
    return results


def run_cgcnn_prediction(model_checkpoint: str, cif_file_path: str):
    """Run CGCNN prediction on a single CIF file"""
    try:
        results = cgcnn_predict.main([model_checkpoint, cif_file_path])
        return results
    except Exception as e:
        print(f"Error running CGCNN prediction: {e}")
        return None


def run_finetuned_cgcnn_prediction(checkpoint_path: str, dataset_root: str, cif_file_path: str):
    """
    dataset_root: path to CIF_OBELiX folder
    CIF files and id_prop.csv are in dataset_root/cifs/
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Point to the cifs subfolder inside dataset_root
    cifs_folder = os.path.join(dataset_root, "cifs")

    # Read id_prop.csv inside the cifs folder to get the list of CIF ids
    id_prop_path = os.path.join(cifs_folder, "id_prop.csv")
    id_prop_df = pd.read_csv(id_prop_path)
    # Assuming first column of id_prop.csv has CIF ids without ".cif"
    cif_ids = id_prop_df.iloc[:, 0].tolist()
    # Append ".cif" to create CIF filenames
    cif_filenames = [cid + ".cif" for cid in cif_ids]

    cif_basename = os.path.basename(cif_file_path)
    sample_index = None
    for idx, fname in enumerate(cif_filenames):
        if fname == cif_basename:
            sample_index = idx
            break

    if sample_index is None:
        raise ValueError(f"CIF file {cif_file_path} not found in dataset folder {cifs_folder}")

    # Load dataset with CIFData pointing at cifs folder (where CIF files live)
    dataset = CIFData(cifs_folder)

    # Prepare single sample batch
    sample = [dataset[sample_index]]
    input_data, targets, cif_ids_result = collate_pool(sample)

    orig_atom_fea_len = input_data[0].shape[-1]
    nbr_fea_len = input_data[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load Normalizer state for denormalization, if available
    normalizer = None
    if 'normalizer' in checkpoint:
        normalizer = Normalizer(torch.tensor([0.0]))
        normalizer.load_state_dict(checkpoint['normalizer'])

    input_vars = (
        input_data[0].to(device),
        input_data[1].to(device),
        input_data[2].to(device),
        input_data[3],  # crystal_atom_idx (list of tensors, stays on CPU)
    )

    with torch.no_grad():
        output = model(*input_vars)
        pred = output.cpu().numpy().flatten()[0]

    # Denormalize prediction if normalizer is available
    if normalizer is not None:
        pred_tensor = torch.tensor([pred])
        pred_denorm = normalizer.denorm(pred_tensor).item()
    else:
        pred_denorm = pred

    # Uncomment below if you used log or log10 transform on target during training
    # pred_final = np.exp(pred_denorm)        # For natural log
    # pred_final = 10 ** pred_denorm           # For log10
    # Otherwise, use pred_denorm directly
    pred_final = pred_denorm

    results = {
        'cif_ids': cif_ids_result,
        'predictions': [pred_final],
        'mae': checkpoint.get('best_mae_error', None),
    }
    return results


def format_cgcnn_prediction_only(results, property_name):
    if results is None:
        return f"{property_name} prediction failed or no results"

    if not results.get('predictions'):
        return f"No predictions available for {property_name}"

    prediction = results['predictions'][0]

    output = f"Prediction for {property_name}: {prediction:.4f}\n"
    if 'mae' in results and results['mae'] is not None:
        output += f"Model MAE: {results['mae']:.4f}\n"

    return output


if __name__ == "__main__":
    cif_path = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\cifs\test_CIF.cif"

    if not os.path.isfile(cif_path):
        print(f"Error: CIF file not found at {cif_path}")
        sys.exit(1)

    dataset_root = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX"

    # Run and save results without printing yet
    try:
        sei_results = run_sei_prediction(cif_path)
    except Exception as e:
        sei_results = None
        print(f"SEI prediction failed: {e}")

    try:
        cei_results = run_cei_prediction(cif_path)
    except Exception as e:
        cei_results = None
        print(f"CEI prediction failed: {e}")

    try:
        bandgap_results = run_cgcnn_prediction(
            r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\cgcnn_pretrained\band-gap.pth.tar",
            cif_path
        )
    except Exception as e:
        bandgap_results = None
        print(f"CGCNN Bandgap prediction failed: {e}")

    try:
        shear_results = run_cgcnn_prediction(
            r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\cgcnn_pretrained\shear-moduli.pth.tar",
            cif_path
        )
    except Exception as e:
        shear_results = None
        print(f"CGCNN Shear Moduli prediction failed: {e}")

    try:
        finetuned_checkpoint_path = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\checkpoint.pth.tar"
        finetuned_results = run_finetuned_cgcnn_prediction(finetuned_checkpoint_path, dataset_root, cif_path)
    except Exception as e:
        finetuned_results = None
        print(f"Fine-tuned CGCNN prediction failed: {e}")

    # Final consolidated print of all results:
    print("\n=== Final Prediction Summary ===\n")

    if sei_results is not None:
        print(f"SEI Score: {sei_results.get('sei_score', 'N/A')}")
        if 'overall_properties' in sei_results:
            print("Overall SEI Properties:")
            for prop, val in sei_results['overall_properties'].items():
                print(f"  {prop}: {val:.3f}")
    else:
        print("SEI Prediction: Failed or no results")

    print()

    if cei_results is not None:
        print(f"CEI Score: {cei_results.get('cei_score', 'N/A')}")
        if 'overall_properties' in cei_results:
            print("Overall CEI Properties:")
            for prop, val in cei_results['overall_properties'].items():
                print(f"  {prop}: {val:.3f}")
    else:
        print("CEI Prediction: Failed or no results")

    print()

    print(format_cgcnn_prediction_only(bandgap_results, "Bandgap"))
    print()
    print(format_cgcnn_prediction_only(shear_results, "Shear Moduli"))
    print()
    print(format_cgcnn_prediction_only(finetuned_results, "Fine-tuned CGCNN (Ionic Conductivity)"))
    print()
