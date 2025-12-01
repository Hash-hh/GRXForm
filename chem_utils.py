from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs, AllChem
from sklearn.ensemble import RandomForestClassifier
# from sklearn.exceptions import NotFittedError


class SurrogateModel:
    """
    A lightweight Random Forest Classifier acting as a 'Fake Oracle'.
    It predicts the probability of a molecule being 'Good' (top percentile).
    """

    def __init__(self, use_classifier=True, percentile_threshold=80):
        # self.model = RandomForestClassifier(n_estimators=100, n_jobs=1, max_depth=10)
        self.model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,  # Use all CPUs
            max_depth=None,  # Let trees grow deep to learn chemical nuances
            min_samples_leaf=1,  # if >1, regularization: prevents memorizing singletons (better than max_depth)
            class_weight="balanced"  # Handle the 80/20 split correctly
        )
        self.is_fitted = False
        self.percentile_threshold = percentile_threshold
        self.fp_cache = {}  # Simple cache for speed within an epoch

    def _get_fp(self, smiles: str) -> Optional[np.ndarray]:
        if smiles in self.fp_cache:
            return self.fp_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 2048 bits for better resolution
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        self.fp_cache[smiles] = arr
        return arr


    def train(self, smiles_list: List[str], scores_list: List[float]):
        """
        Trains the RF classifier.
        Labels are created dynamically: 1 if score >= percentile_threshold, else 0.
        """
        if not smiles_list:
            return

        # 1. Prepare Data
        X = []
        valid_scores = []
        seen_smiles = set()

        for smi, score in zip(smiles_list, scores_list):
            if score == float("-inf") or score is None:
                continue
            if smi in seen_smiles:
                continue
            fp = self._get_fp(smi)
            if fp is not None:
                X.append(fp)
                valid_scores.append(score)
                seen_smiles.add(smi)

        if len(X) < 50:  # Don't train on too little data
            print(f"[Surrogate] Not enough data to train yet ({len(X)} samples).")
            return

        # 2. Define Label Threshold (Dynamic)
        # E.g., Top 20% of molecules seen so far are "Class 1"
        threshold = np.percentile(valid_scores, self.percentile_threshold)
        y = [1 if s >= threshold else 0 for s in valid_scores]

        # 3. Train
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            acc = self.model.score(X, y)
            print(f"[Surrogate] Trained on {len(X)} samples. Threshold: {threshold:.3f}. Train Acc: {acc:.3f}")
        except Exception as e:
            print(f"[Surrogate] Training failed: {e}")
            self.is_fitted = False

        # Clear cache after training to save memory
        self.fp_cache = {}

    def predict_ranking_scores(self, smiles_list: List[str]) -> np.ndarray:
        """
        Returns a score for ranking.
        For Classifier: Returns probability of Class 1 (Good).
        """
        if not self.is_fitted:
            # Fallback: random scores if not trained yet
            return np.random.rand(len(smiles_list))

        X = []
        indices_kept = []

        for i, smi in enumerate(smiles_list):
            fp = self._get_fp(smi)
            if fp is not None:
                X.append(fp)
                indices_kept.append(i)

        if not X:
            return np.zeros(len(smiles_list))

        # Predict Probabilities (Class 1)
        # try:
        probs = self.model.predict_proba(X)[:, 1]
        # except:
        #     # Handle edge case where model only learned one class
        #     probs = np.zeros(len(X))

        # Reconstruct full array matching input length
        final_scores = np.zeros(len(smiles_list))
        for idx, p in zip(indices_kept, probs):
            final_scores[idx] = p

        return final_scores


def get_scaffold(smiles: str) -> str:
    """
    Extracts the Bemis-Murcko scaffold.
    If the molecule is acyclic (returns ''), returns the original SMILES.
    """
    # try:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "C"

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smi = Chem.MolToSmiles(scaffold)

    if not scaffold_smi:
        return smiles

    return scaffold_smi
    # except:
    #     return "C"


def update_hall_of_fame(
        current_hof: List[Dict],
        new_candidates: List[Dict],
        max_size: int = 10,
        similarity_threshold: float = 0.75
) -> List[Dict]:
    """
    Updates HoF with new candidates, enforcing diversity.
    Optimized to compute fingerprints only once.
    """
    # 1. Pre-compute fingerprints for existing HoF to avoid re-doing it in the loop
    # Structure: List of (dict_data, fingerprint_object)
    hof_with_fps: List[Tuple[Dict, Any]] = []

    for entry in current_hof:
        mol = Chem.MolFromSmiles(entry['smiles'])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            hof_with_fps.append((entry, fp))

    # 2. Sort candidates best-first (highest objective)
    candidates = sorted(new_candidates, key=lambda x: x['obj'], reverse=True)

    for cand in candidates:
        if not cand['smiles']: continue

        cand_mol = Chem.MolFromSmiles(cand['smiles'])
        if not cand_mol: continue

        # Canonicalize SMILES to ensure clean string storage
        cand['smiles'] = Chem.MolToSmiles(cand_mol)
        cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2, nBits=1024)

        added_or_replaced = False

        # Check against dynamic HoF
        for i, (elite_entry, elite_fp) in enumerate(hof_with_fps):
            sim = DataStructs.TanimotoSimilarity(cand_fp, elite_fp)

            if sim > similarity_threshold:
                # Similar structure found (or Exact Duplicate if sim == 1.0)
                if cand['obj'] > elite_entry['obj']:
                    # Replace the existing entry with the better candidate
                    hof_with_fps[i] = (cand, cand_fp)

                added_or_replaced = True
                break

        if not added_or_replaced:
            # Novel scaffold found, add to our list
            hof_with_fps.append((cand, cand_fp))

    # 3. Unpack, Sort, and Prune
    # We only want the dicts back, sorted by score
    final_hof = [item[0] for item in hof_with_fps]
    final_hof = sorted(final_hof, key=lambda x: x['obj'], reverse=True)

    return final_hof[:max_size]


def sample_prompt_for_epoch(
        hof: List[Dict],
        elite_prob: float = 0.5
) -> str:
    """
    Returns a SINGLE prompt string for the epoch.
    """
    if not hof:
        return "C"

    if np.random.random() > elite_prob:
        return "C"

    elite_entry = random.choice(hof)
    return get_scaffold(elite_entry['smiles'])