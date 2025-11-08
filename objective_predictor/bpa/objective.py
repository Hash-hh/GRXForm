import numpy as np
import os
import warnings
import pandas as pd
import re  # For parsing PolyNC output

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
from rdkit.Contrib.SA_Score import sascorer

# --- NEW IMPORTS for PolyNC ---
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    print("Error: 'torch' or 'transformers' package not found.")
    print("Please install them using: pip install torch transformers")
    exit()

# --- NEW IMPORT for ADMET-AI ---
try:
    from admet_ai import ADMETModel
except ImportError:
    print("Error: 'admet-ai' package not found.")
    print("Please install it using: pip install admet-ai")
    exit()

# --- Setup and Configuration ---
warnings.filterwarnings("ignore")

try:
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    MODULE_DIR = os.getcwd()  # Fallback for interactive environments


class BPA_Scorer:
    """
    Calculates a multi-objective reward for BPA alternatives, optimizing for:
    1.  Low Toxicity (admet-ai, 18 endpoints, thresholded)
    2.  High Glass Transition Temp (Tg) (PolyNC model)
    3.  Hard Structural Constraints (rings, -OH groups, distance)

    Also maintains a 'leaderboard' of the top 20 molecules seen.
    """

    def __init__(self, config):

        print("Initializing BPA-Free Polymer Objective (Toxicity + Tg)...")

        # --- 1. ADMET-AI (Toxicity) Initialization ---
        print("  Loading 'admet-ai' model suite...")
        try:
            self.admet_model = ADMETModel()
            print("  ✅ 'admet-ai' loaded.")
        except Exception as e:
            print(f"Error initializing ADMETModel: {e}")
            raise

        polync_device = config.polync_device

        # --- 2. PolyNC (Tg) Initialization ---
        print(f"  Loading 'PolyNC' model onto {polync_device}...")
        try:
            self.polync_device = torch.device(polync_device)
            self.polync_tokenizer = T5Tokenizer.from_pretrained("hkqiu/PolyNC")
            self.polync_model = T5ForConditionalGeneration.from_pretrained("hkqiu/PolyNC").to(self.polync_device)
            self.polync_model.eval()
            print("  ✅ 'PolyNC' model loaded.")
        except Exception as e:
            print(f"Error initializing PolyNC model: {e}")
            raise

        # --- 3. Define Toxicity Endpoints (Probabilities) ---
        self.tox_prob_columns = [
            'hERG', 'ClinTox', 'AMES', 'DILI', 'Carcinogens_Lagunin',
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]

        # --- 4. Define Structural Constraint Constants ---
        self.MIN_OH_DISTANCE = 9
        self.phenol_pattern = Chem.MolFromSmarts('[OH]c')
        self.psmiles_phenol_H_pattern = Chem.MolFromSmarts('[H][O]c')  # For pSMILES construction

        # --- 5. Define Objective Weights ---
        self.w_tox = 0.5  # 50% weight for non-toxicity
        self.w_tg = 0.5  # 50% weight for high Tg

        # --- 6. Define Tg Normalization Range ---
        self.tg_min = 50.0  # Min desired Tg (°C)
        self.tg_max = 400.0  # Max desired Tg (°C)
        self.tg_range = self.tg_max - self.tg_min

        # --- 7. NEW: Leaderboard Setup ---
        self.top_k = 20
        self.results_dir = os.path.join(MODULE_DIR, 'bpa_results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.leaderboard_file = os.path.join(self.results_dir, 'bpa_top20_leaderboard.txt')
        self.leaderboard_cols = ['smiles', 'combined_score', 'tox_score', 'tg_score', 'actual_tg']
        # Load existing leaderboard or create new one
        try:
            self.leaderboard_df = pd.read_csv(self.leaderboard_file, sep='\t')
            print(f"  Loaded existing leaderboard from {self.leaderboard_file}")
        except FileNotFoundError:
            self.leaderboard_df = pd.DataFrame(columns=self.leaderboard_cols)
            print("  Initialized new leaderboard.")

        print(f"  Tracking {len(self.tox_prob_columns)} toxicity endpoints.")
        print("  Tracking 1 Tg regression endpoint (PolyNC).")
        print("  Enforcing hard structural constraints.")
        print("✅ BPA-Free Polymer Objective initialized.")

    def _get_rdkit_mol(self, smiles: str):
        """Helper to safely get a valid RDKit molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def _check_constraints(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule meets the hard structural constraints."""
        try:
            # 1. Must have at least two rings
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            if len(atom_rings) < 2: return False

            # 2. Rings must be size 5 or 6
            for ring in atom_rings:
                if len(ring) > 6 or len(ring) < 5: return False

            # 3. Must have exactly 2 phenols and 0 alcohols
            aromatic_oh = Fragments.fr_Ar_OH(mol)
            aliphatic_oh = Fragments.fr_Al_OH(mol)
            if aromatic_oh != 2 or aliphatic_oh > 0: return False

            # 4. Find the two phenol oxygen atoms
            phenol_matches = mol.GetSubstructMatches(self.phenol_pattern)
            if len(phenol_matches) != 2: return False
            oh_oxygen_atoms = [phenol_matches[0][0], phenol_matches[1][0]]

            # 5. Calculate the topological distance
            path = Chem.GetShortestPath(mol, oh_oxygen_atoms[0], oh_oxygen_atoms[1])
            distance = len(path) - 1

            # 6. Set a distance threshold.
            if distance < self.MIN_OH_DISTANCE: return False

            return True  # All checks passed
        except Exception:
            return False

    def _construct_psmiles(self, mol: Chem.Mol) -> str:
        """
        Converts a valid bisphenol monomer (HO-R-OH) into its
        polycarbonate repeating unit pSMILES (*O-R-O-C(=O)*).
        """
        try:
            # 1. Add explicit hydrogens
            mol_with_H = Chem.AddHs(mol)

            # 2. Find the two phenol groups
            matches = mol_with_H.GetSubstructMatches(self.psmiles_phenol_H_pattern)
            if len(matches) != 2:
                raise ValueError(f"Expected 2 phenol H-matches, found {len(matches)}")

            # 3. Get atom indices for the two (H)ydrogen and (O)xygen atoms
            h1_idx, o1_idx = matches[0][0], matches[0][1]
            h2_idx, o2_idx = matches[1][0], matches[1][1]

            # 4. Convert to an editable molecule
            rw_mol = Chem.RWMol(mol_with_H)

            # 5. Modification 1: Replace the first H with a wildcard '*'
            h1_atom = rw_mol.GetAtomWithIdx(h1_idx)
            h1_atom.SetAtomicNum(0)
            h1_atom.SetIsotope(0)
            h1_atom.SetFormalCharge(0)
            h1_atom.SetNoImplicit(True)  # This is crucial

            # 6. Modification 2: Replace the second H with a carbonate group

            # 6a. Remove the second hydrogen atom completely
            rw_mol.RemoveAtom(h2_idx)

            # 6b. Add the three new atoms for the carbonate: C, O, *
            c_carb_idx = rw_mol.AddAtom(Chem.Atom(6))  # Carbonyl C
            o_carb_idx = rw_mol.AddAtom(Chem.Atom(8))  # Carbonyl O
            star_2_idx = rw_mol.AddAtom(Chem.Atom(0))  # Second wildcard

            # 6c. Set the second wildcard atom's properties
            star_2_atom = rw_mol.GetAtomWithIdx(star_2_idx)
            star_2_atom.SetIsotope(0)
            star_2_atom.SetFormalCharge(0)
            star_2_atom.SetNoImplicit(True)

            # 6d. Add the new bonds to connect the carbonate group
            rw_mol.AddBond(o2_idx, c_carb_idx, Chem.BondType.SINGLE)  # -O-C...
            rw_mol.AddBond(c_carb_idx, o_carb_idx, Chem.BondType.DOUBLE)  # -C=O
            rw_mol.AddBond(c_carb_idx, star_2_idx, Chem.BondType.SINGLE)  # -C-*

            # 7. Get the molecule back (still has other hydrogens)
            final_mol_with_H = rw_mol.GetMol()

            # 8. Remove all *other* hydrogens, leaving our two '*'
            final_mol = Chem.RemoveHs(final_mol_with_H)

            # 9. Sanitize and create the final pSMILES
            Chem.SanitizeMol(final_mol)

            print("original smiles: ", Chem.MolToSmiles(mol))
            print("polymer smiles: ", Chem.MolToSmiles(final_mol))
            return Chem.MolToSmiles(final_mol)

        except Exception as e:
            print(f"Error in _construct_psmiles: {e}")
            return None

    def _parse_polync_output(self, text_output: str) -> float:
        """Parses the text output from PolyNC (e.g., "103.85 (°C)")"""
        try:
            match = re.search(r"[-+]?\d*\.\d+|\d+", text_output)
            if match:
                return float(match.group(0))
            else:
                return -np.inf  # Return -inf if no number is found
        except ValueError:
            return -np.inf

    def _predict_tg(self, psmiles_list: list) -> list:
        """Predicts Tg for a batch of pSMILES strings using PolyNC."""
        if not psmiles_list:
            return []

        valid_psmiles_list = [ps for ps in psmiles_list if ps is not None]
        if not valid_psmiles_list:
            return [-np.inf] * len(psmiles_list)

        prompts = [f"Predict the Tg of the following SMILES: {s}" for s in valid_psmiles_list]
        try:
            inputs = self.polync_tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=150
            ).to(self.polync_device)

            with torch.no_grad():
                outputs = self.polync_model.generate(**inputs, max_new_tokens=8)

            decoded_outputs = self.polync_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            scores = [self._parse_polync_output(out) for out in decoded_outputs]

            score_iter = iter(scores)
            final_scores = [-np.inf if ps is None else next(score_iter) for ps in psmiles_list]
            return final_scores

        except Exception as e:
            print(f"Error during PolyNC prediction: {e}")
            return [-np.inf] * len(psmiles_list)

    def _update_leaderboard(self, batch_df: pd.DataFrame):
        """Updates the top-K leaderboard and saves to file."""
        if batch_df.empty:
            return  # Nothing to update
        try:
            # 1. Combine old and new
            combined_df = pd.concat([self.leaderboard_df, batch_df], ignore_index=True)

            # 2. Sort by best score
            combined_df = combined_df.sort_values(by='combined_score', ascending=False)

            # 3. Drop duplicates, keeping the best score (which is now 'first')
            combined_df = combined_df.drop_duplicates(subset='smiles', keep='first')

            # 4. Get top K
            self.leaderboard_df = combined_df.head(self.top_k).reset_index(drop=True)

            # 5. Save to file
            self.leaderboard_df.to_csv(
                self.leaderboard_file,
                sep='\t',
                index=False,
                float_format='%.4f'  # Format numbers for readability
            )
        except Exception as e:
            # This should not stop the main scoring
            print(f"Warning: Could not update leaderboard file. Error: {e}")

    def __call__(self, smiles_list: list) -> np.ndarray:
        """
        Calculates the final multi-objective score for a list of SMILES.
        """
        final_scores = np.zeros(len(smiles_list))

        # --- 1. Pre-filter molecules based on constraints ---
        valid_mols_map = {}  # Maps original index to a valid mol
        valid_smiles_to_idx = {}  # Maps valid monomer SMILES to list of original indices

        for i, smiles in enumerate(smiles_list):
            mol = self._get_rdkit_mol(smiles)
            if mol is None or not self._check_constraints(mol):
                final_scores[i] = 0.0  # Penalty for invalid/failed constraints
                continue

            # Store valid mol
            valid_mols_map[i] = mol

            # Update SMILES-to-index mapping
            valid_smiles = Chem.MolToSmiles(mol)
            if valid_smiles not in valid_smiles_to_idx:
                valid_smiles_to_idx[valid_smiles] = []
            valid_smiles_to_idx[valid_smiles].append(i)

        if not valid_mols_map:
            return final_scores  # No molecules passed filters

        # --- 2. Prepare Batches for Prediction ---
        unique_mols = {smiles: valid_mols_map[indices[0]] for smiles, indices in valid_smiles_to_idx.items()}
        unique_monomer_smiles = list(unique_mols.keys())

        # --- 3. Run Predictions in Batches ---
        try:
            # --- Part A: Toxicity Prediction (admet-ai) ---
            preds_df = self.admet_model.predict(smiles=unique_monomer_smiles)

            # --- Part B: Tg Prediction (PolyNC) ---
            unique_psmiles = [self._construct_psmiles(mol) for mol in unique_mols.values()]
            tg_values_list = self._predict_tg(unique_psmiles)
            tg_series = pd.Series(tg_values_list, index=unique_monomer_smiles)  # actual_tg

            # --- 4. Calculate Scores ---

            # --- Toxicity Score ---
            tox_probs_df = preds_df[self.tox_prob_columns]
            penalties = (tox_probs_df - 0.5) * 2.0
            clipped_penalties = penalties.clip(lower=0.0)
            toxicity_scores = 1.0 - clipped_penalties
            final_tox_score = toxicity_scores.mean(axis=1)  # pd.Series (0-1 score)

            # --- Tg Score ---
            normalized_tg = (tg_series - self.tg_min) / self.tg_range
            final_tg_score = normalized_tg.clip(0.0, 1.0)  # pd.Series (0-1 score)

            # --- Part C: Combine Scores ---
            combined_scores = (self.w_tox * final_tox_score) + (self.w_tg * final_tg_score)

            # --- 5. NEW: Update Leaderboard ---
            # Create a DataFrame of the new results
            batch_df = pd.DataFrame({
                'smiles': combined_scores.index,
                'combined_score': combined_scores.values,
                'tox_score': final_tox_score.reindex(combined_scores.index).values,
                'tg_score': final_tg_score.reindex(combined_scores.index).values,
                'actual_tg': tg_series.reindex(combined_scores.index).values
            })
            # Update the leaderboard
            self._update_leaderboard(batch_df)

            # --- 6. Map Scores Back to Original List ---
            for smiles, score in combined_scores.items():
                if pd.isna(score): score = 0.0  # Handle potential NaNs
                original_indices = valid_smiles_to_idx[smiles]
                for idx in original_indices:
                    final_scores[idx] = max(0.0, score)

        except KeyError as e:
            print(f"--- ERROR ---")
            print(f"A required column name was not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during scoring: {e}")

        return np.array(final_scores)


# --- Example Usage ---
if __name__ == "__main__":
    try:
        # This will take a moment to load both admet-ai and PolyNC models
        scorer = BPA_Scorer()

        test_smiles = [
            # --- Should FAIL constraints ---
            'c1ccccc1',  # Fails ring count < 2
            'O-c1ccccc1-O',  # Fails ring count < 2
            'O-c1ccccc1-c2ccccc2-O',  # Biphenol (fails distance < 8)

            # --- Should PASS constraints & be scored ---
            'CC(C)(c1ccc(O)cc1)c2ccc(O)cc2',  # BPA (distance = 8, passes)
            'O-c1ccc(S(=O)(=O)c2ccc(O)cc2)cc1',  # Bisphenol S (BPS, distance=8)
            'O-c1ccc(C(F)(F)c2ccc(O)cc2)cc1',  # Bisphenol AF (BPAF, distance=8)
            'O-c1ccc(C(C)(C)c2ccc(O)cc2)cc1',  # 4,4'-t-butyl-BPA (distance=8)

            # --- Invalid SMILES ---
            'invalid-smiles-string',  # Fails _get_rdkit_mol
        ]

        # Let's add Biphenol (dist=7) and BPF (dist=7) back to test the filter
        test_smiles.extend(['O-c1ccccc1-c2ccccc2-O', 'O-c1ccc(Cc2ccc(O)cc2)cc1'])

        print("\nRunning predictions (this may take a moment)...")
        scores = scorer(test_smiles)

        print(f"\n--- Scoring Results (Toxicity + Tg, Constraints) ---")
        for smi, score in zip(test_smiles, scores):
            status = "FAILED constraints" if score == 0.0 else "PASSED & SCORED"
            print(f"SMILES: {smi}\nScore: {score:.4f} ({status})\n")

        print(f"\n--- Top 20 Leaderboard ---")
        print(f"Saved to: {scorer.leaderboard_file}")
        print(scorer.leaderboard_df)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")