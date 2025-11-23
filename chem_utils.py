from typing import List, Dict, Any, Tuple
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs, AllChem


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