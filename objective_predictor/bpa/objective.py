import numpy as np
import os
import joblib
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Contrib.SA_Score import sascorer
# from tdc.Tox import Tox

# We don't need tdc.Tox or tdc.Oracle for *prediction*,
# only for the initial training (which you've already done).

# --- Setup and Configuration ---
warnings.filterwarnings("ignore")

try:
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    MODULE_DIR = os.getcwd()  # Fallback for interactive environments

# Define the directory where your trained .pkl models are saved
MODELS_DIR = os.path.join(MODULE_DIR, 'bpa_surrogate_models')
if not os.path.exists(MODELS_DIR):
    print(f"Warning: Model directory not found at {MODELS_DIR}. "
          f"Please ensure 'er_alpha_model.pkl' and 'ar_antagonist_model.pkl' are present.")
    # We don't create it here, as the models should already exist.


# --- Helper Function ---

def get_morgan_fingerprint(smiles_string):
    """Generates a Morgan fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)


class BPA_Scorer:
    """
    Calculates a reward for molecules as non-toxic BPA alternatives.

    This version is optimized for:
    - 70% Non-Toxicity (ER-alpha & AR binding)
    - 30% Synthetic Accessibility (relative to BPA)
    """

    def __init__(self, tdc_device='cpu'):  # tdc_device is unused, kept for API compatibility
        print("Initializing BPA-Free Objective (Toxicity + SA)...")

        # --- 1. Define Model Paths ---
        self.model_paths = {
            'er_alpha': os.path.join(MODELS_DIR, 'er_alpha_model.pkl'),
            'ar': os.path.join(MODELS_DIR, 'ar_antagonist_model.pkl'),
        }

        # --- 2. Load Pre-trained Surrogate Models ---
        self.models = {}
        print("Loading surrogate toxicity models...")
        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}. "
                    f"Please run the 'BPA_Alternative_Scorer' (your other objective file)"
                    f" as a main script first to train and save the surrogate models."
                )
            # No try/except here, as requested. Will crash if loading fails.
            self.models[name] = joblib.load(path)

        print(f"  Loaded {len(self.models)} surrogate models.")

        # --- 3. Calculate BPA SA Score Baseline ---
        bpa_smiles = 'CC(C)(c1ccc(O)cc1)c2ccc(O)cc2'
        bpa_mol = self._get_rdkit_mol(bpa_smiles)
        if bpa_mol is None:
            raise ValueError("Could not parse BPA SMILES for SA baseline.")
        self.SA_BPA = sascorer.calculateScore(bpa_mol)
        print(f"  BPA SA Score (Baseline): {self.SA_BPA:.4f}")

        # --- 4. Define Reward Weights (Toxicity + SA) ---
        # Total score = 70% * (Tox Score) + 30% * (SA Score)
        # Tox Score = 50% * f_er + 50% * f_ar
        # Final weights:
        self.w_sa = 0.30
        self.w_er = 0.35  # 0.7 * 0.5
        self.w_ar = 0.35  # 0.7 * 0.5

        print("✅ BPA-Free Objective (Toxicity + SA) initialized.")

    def _get_rdkit_mol(self, smiles: str):
        """Helper to safely get a valid RDKit molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            # Catch sanitization/parsing errors for invalid SMILES
            return None

    def _calculate_property_score(self, mol: Chem.Mol, smiles: str):
        """Calculates the reward for chemical properties."""

        # --- Toxicity Scores (f_tox) ---
        fp = get_morgan_fingerprint(smiles)
        if fp is None:
            raise ValueError(f"Failed to get fingerprint for valid SMILES: {smiles}")

        fp = fp.reshape(1, -1)

        # predict_proba(fp)[0][1] gets P(class 1 = toxic)
        f_er = 1.0 - self.models['er_alpha'].predict_proba(fp)[0][1]
        f_ar = 1.0 - self.models['ar'].predict_proba(fp)[0][1]

        # --- SA Score (f_sa) ---
        # Reward is 1.0 if SA_mol <= SA_BPA.
        # Reward decreases linearly to 0.0 as SA_mol approaches 10.0.
        sa_score_raw = sascorer.calculateScore(mol)

        # Calculate penalty only if score is worse (higher) than BPA's
        sa_penalty = max(0.0, sa_score_raw - self.SA_BPA)

        # Normalize penalty based on the range from BPA's score to the max (10.0)
        penalty_range = 10.0 - self.SA_BPA
        normalized_penalty = sa_penalty / penalty_range if penalty_range > 0 else 0.0

        f_sa = 1.0 - np.clip(normalized_penalty, 0.0, 1.0)

        # Combine property scores
        score = (self.w_sa * f_sa) + (self.w_er * f_er) + (self.w_ar * f_ar)
        return score

    def __call__(self, smiles_list: list) -> np.ndarray:
        """Calculates the final objective score for a list of SMILES."""

        final_scores = []
        for smiles in smiles_list:
            mol = self._get_rdkit_mol(smiles)

            # Handle invalid SMILES input robustly.
            if mol is None:
                final_scores.append(0.0)
                continue

            # Calculate the property score
            score_properties = self._calculate_property_score(mol, smiles)

            # The final score IS the property score
            final_score = score_properties

            # Ensure final score is non-negative
            final_scores.append(max(0.0, final_score))

        return np.array(final_scores)


# --- Example Usage ---
if __name__ == "__main__":
    # This block is for testing.
    # It assumes your models (er_alpha_model.pkl, ar_antagonist_model.pkl)
    # already exist in the 'bpa_surrogate_models' directory.

    try:
        scorer = BPA_Scorer()

        test_smiles = [
            'CC(C)(c1ccc(O)cc1)c2ccc(O)cc2',  # Bisphenol A (BPA) - bad tox, perfect SA
            'c1ccccc1',  # Benzene - good tox, good SA
            'O-c1ccccc1-c2ccccc2-O',  # Biphenol - better tox, good SA
            'O-c1cccc(O)c1',  # Resorcinol - good tox, good SA
            'CCC',  # Propane - good tox, good SA
            'CCO',  # Ethanol - good tox, good SA
            'O-c1cc(C)ccc1-C(C)(C)-c2ccc(O)cc2'  # Complex - better tox, ok SA
        ]

        scores = scorer(test_smiles)

        print("\n--- Scoring Results (70% Tox, 30% SA, Higher is Better) ---")
        for smi, score in zip(test_smiles, scores):
            print(f"SMILES: {smi}\nScore: {score:.4f}\n")

    except FileNotFoundError as e:
        print("\n--- ERROR ---")
        print(e)
        print("Please run the 'BPA_Alternative_Scorer' (your other objective file)")
        print("as a main script first to train and save the surrogate models.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")












# import numpy as np
# import os
# import joblib
# import warnings
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
# from rdkit.Contrib.SA_Score import sascorer
#
# # We don't need tdc.Tox or tdc.Oracle for *prediction*,
# # only for the initial training (which you've already done).
#
# # --- Setup and Configuration ---
# warnings.filterwarnings("ignore")
#
# try:
#     MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     MODULE_DIR = os.getcwd()  # Fallback for interactive environments
#
# # Define the directory where your trained .pkl models are saved
# MODELS_DIR = os.path.join(MODULE_DIR, 'bpa_surrogate_models')
# if not os.path.exists(MODELS_DIR):
#     print(f"Warning: Model directory not found at {MODELS_DIR}. "
#           f"Please ensure 'er_alpha_model.pkl' and 'ar_antagonist_model.pkl' are present.")
#     # We don't create it here, as the models should already exist.
#
#
# # --- Helper Function ---
#
# def get_morgan_fingerprint(smiles_string):
#     """Generates a Morgan fingerprint for a given SMILES string."""
#     mol = Chem.MolFromSmiles(smiles_string)
#     if mol is None:
#         return None
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#     return np.array(fp)
#
#
# class BPA_Scorer:
#     """
#     Calculates a reward for molecules as non-toxic BPA alternatives.
#
#     This function is optimized *only* for non-toxicity based on
#     Estrogen Receptor (ER-alpha) and Androgen Receptor (AR) binding.
#     """
#
#     def __init__(self, tdc_device='cpu'):  # tdc_device is unused, kept for API compatibility
#         print("Initializing BPA-Free Objective (Toxicity-Only)...")
#
#         # --- 1. Define Model Paths ---
#         self.model_paths = {
#             'er_alpha': os.path.join(MODELS_DIR, 'er_alpha_model.pkl'),
#             'ar': os.path.join(MODELS_DIR, 'ar_antagonist_model.pkl'),
#         }
#
#         # --- 2. Load Pre-trained Surrogate Models ---
#         self.models = {}
#         print("Loading surrogate toxicity models...")
#         for name, path in self.model_paths.items():
#             if not os.path.exists(path):
#                 raise FileNotFoundError(
#                     f"Model file not found: {path}. "
#                     f"Please run the 'BPA_Alternative_Scorer' (your other objective file)"
#                     f" as a main script first to train and save the surrogate models."
#                 )
#             # No try/except here, as requested. Will crash if loading fails.
#             self.models[name] = joblib.load(path)
#
#         print(f"  Loaded {len(self.models)} surrogate models.")
#
#         # --- 3. Define Reward Weights ---
#         # We are only optimizing properties
#         self.W_PROPERTIES = 1.0
#
#         # --- Sub-weights for Properties (sum to 1.0) ---
#         # Only include toxicity weights
#         self.w_er = 0.5
#         self.w_ar = 0.5
#
#         print("✅ BPA-Free Objective (Toxicity-Only) initialized.")
#
#     def _get_rdkit_mol(self, smiles: str):
#         """Helper to safely get a valid RDKit molecule."""
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 return None
#             Chem.SanitizeMol(mol)
#             return mol
#         except Exception:
#             # Catch sanitization/parsing errors for invalid SMILES
#             return None
#
#     def _calculate_property_score(self, mol: Chem.Mol, smiles: str):
#         """Calculates the reward for chemical properties."""
#
#         # --- Toxicity Scores (f_tox) ---
#         # We need a fingerprint for the surrogate models
#         fp = get_morgan_fingerprint(smiles)
#         if fp is None:
#             # This should not happen if _get_rdkit_mol worked,
#             # but as a safeguard, we raise an error.
#             raise ValueError(f"Failed to get fingerprint for valid SMILES: {smiles}")
#
#         fp = fp.reshape(1, -1)
#
#         # No try/except block, as requested.
#         # This will crash if prediction fails.
#         # predict_proba(fp)[0][1] gets P(class 1 = toxic)
#         f_er = 1.0 - self.models['er_alpha'].predict_proba(fp)[0][1]
#         f_ar = 1.0 - self.models['ar'].predict_proba(fp)[0][1]
#
#         # --- SA Score (f_sa) ---
#         # REMOVED
#
#         # Combine property scores
#         score = (self.w_er * f_er) + (self.w_ar * f_ar)
#         return score
#
#     def __call__(self, smiles_list: list) -> np.ndarray:
#         """Calculates the final objective score for a list of SMILES."""
#
#         final_scores = []
#         for smiles in smiles_list:
#             mol = self._get_rdkit_mol(smiles)
#
#             # Handle invalid SMILES input robustly.
#             # This is not a "silent failure" but correct input handling.
#             if mol is None:
#                 final_scores.append(0.0)
#                 continue
#
#             # Calculate the property score
#             score_properties = self._calculate_property_score(mol, smiles)
#
#             # The final score IS the property score
#             final_score = score_properties
#
#             # Ensure final score is non-negative
#             final_scores.append(max(0.0, final_score))
#
#         return np.array(final_scores)
#
#
# # --- Example Usage ---
# if __name__ == "__main__":
#     # This block is for testing.
#     # It assumes your models (er_alpha_model.pkl, ar_antagonist_model.pkl)
#     # already exist in the 'bpa_surrogate_models' directory.
#
#     try:
#         scorer = BPA_Scorer()
#
#         test_smiles = [
#             'CC(C)(c1ccc(O)cc1)c2ccc(O)cc2',  # Bisphenol A (BPA) - should score poorly on tox
#             'c1ccccc1',  # Benzene
#             'O-c1ccccc1-c2ccccc2-O',  # Biphenol
#             'O-c1cccc(O)c1',  # Resorcinol
#             'CCC',  # Propane
#             'CCO',  # Ethanol
#             'O-c1cc(C)ccc1-C(C)(C)-c2ccc(O)cc2'  # Complex
#         ]
#
#         scores = scorer(test_smiles)
#
#         print("\n--- Scoring Results (Toxicity-Only, Higher is Better) ---")
#         for smi, score in zip(test_smiles, scores):
#             print(f"SMILES: {smi}\nScore: {score:.4f}\n")
#
#     except FileNotFoundError as e:
#         print("\n--- ERROR ---")
#         print(e)
#         print("Please run the 'BPA_Alternative_Scorer' (your other objective file)")
#         print("as a main script first to train and save the surrogate models.")
#     except Exception as e:
#         print(f"\nAn unexpected error occurred: {e}")