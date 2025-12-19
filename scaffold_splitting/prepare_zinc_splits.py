import os
import random
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# --- Configuration ---
SMILES_FILE = r"../data/zinc/zinc.smiles"
OUTPUT_DIR = "zinc_splits"
SEEDS = [42, 43, 44]

# --- SPLIT CONFIGURATION ---
TRAIN_RATIO = 0.8
NUM_VAL_SCAFFOLDS = 100  # Stolen from the Test set


def get_scaffold(smiles):
    """Get Murcko scaffold for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None


def main():
    # Step 1: Load SMILES
    print(f"--- Loading SMILES from {SMILES_FILE} ---")
    if not os.path.exists(SMILES_FILE):
        print(f"Error: File {SMILES_FILE} not found.")
        return

    with open(SMILES_FILE, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(smiles_list)} SMILES.")

    allowed_vocabulary = [  # put multi-character occurences first
        "[NH3+]", "[SH+]", "[C@]", "[O+]", "[NH+]", "[nH+]", "[C@@H]", "[CH2-]", "[C@H]", "[NH2+]", "[S+]", "[CH-]",
        "[S@]", "[N-]",
        "[s+]", "[nH]", "[S@@]", "[n+]", "[o+]", "[NH-]", "[C@@]", "[S-]", "[N+]", "[OH+]", "[O-]", "[n-]",
        "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7",
        "C", "O", "S", "s", "F", "P", "p", "Cl", "Br", "I"
    ]

    filtered_smiles = []
    for smile in tqdm(smiles_list):
        temp = smile
        for voc in allowed_vocabulary:
            temp = temp.replace(voc, "")
        if len(temp) == 0:
            filtered_smiles.append(smile)

    if len(filtered_smiles) == len(smiles_list):
        print("All SMILES passed the vocabulary filter.")

    # Step 2: Compute scaffolds and group molecules
    print("--- Computing Scaffolds ---")
    scaffold_to_molecules = defaultdict(list)

    for smi in tqdm(smiles_list, desc="Computing scaffolds"):
        scaffold = get_scaffold(smi)
        if scaffold is not None:
            scaffold_to_molecules[scaffold].append(smi)

    scaffolds = list(scaffold_to_molecules.keys())
    print(f"Found {len(scaffolds)} unique scaffolds.")

    if len(scaffolds) * (1 - TRAIN_RATIO) < NUM_VAL_SCAFFOLDS:
        print("Error: Test set is too small to extract 100 validation scaffolds!")
        return

    # Step 3: Generate splits for each seed
    print("--- Generating Splits ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in SEEDS:
        print(f"\nProcessing Seed {seed}...")
        random.seed(seed)

        shuffled_scaffolds = scaffolds.copy()
        random.shuffle(shuffled_scaffolds)

        # 1. Initial 80/20 Split
        split_idx = int(len(shuffled_scaffolds) * TRAIN_RATIO)
        train_scaffolds = shuffled_scaffolds[:split_idx]
        test_pool = shuffled_scaffolds[split_idx:]

        # 2. Extract Validation from the Test Pool
        # We take the first 100 scaffolds from the pool for Validation
        val_scaffolds = test_pool[:NUM_VAL_SCAFFOLDS]

        # 3. Finalize Test Set
        # The remainder of the pool becomes the actual Test set
        test_scaffolds = test_pool[NUM_VAL_SCAFFOLDS:]

        # Helper to collect molecules
        def get_mols(scaffold_list):
            mols = []
            for s in scaffold_list:
                mols.extend(scaffold_to_molecules[s])
            return mols

        train_molecules = get_mols(train_scaffolds)
        val_molecules = get_mols(val_scaffolds)
        test_molecules = get_mols(test_scaffolds)

        # Save to files
        run_dir = os.path.join(OUTPUT_DIR, f"run_seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Save Molecules
        with open(os.path.join(run_dir, "train_molecules.txt"), 'w') as f:
            f.write('\n'.join(train_molecules))
        with open(os.path.join(run_dir, "val_molecules.txt"), 'w') as f:
            f.write('\n'.join(val_molecules))
        with open(os.path.join(run_dir, "test_molecules.txt"), 'w') as f:
            f.write('\n'.join(test_molecules))

        # Save Scaffolds
        with open(os.path.join(run_dir, "train_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(train_scaffolds))
        with open(os.path.join(run_dir, "val_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(val_scaffolds))
        with open(os.path.join(run_dir, "test_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(test_scaffolds))

        # --- ADDED STATS LOGGING ---
        def get_atom_stats(scaff_list):
            if not scaff_list: return 0.0, 0, 0
            # GetNumAtoms() returns heavy atoms for implicit-H SMILES
            sizes = [Chem.MolFromSmiles(s).GetNumAtoms() for s in scaff_list if Chem.MolFromSmiles(s)]
            if not sizes: return 0.0, 0, 0
            return sum(sizes) / len(sizes), min(sizes), max(sizes)

        print(f"  [Saved to {run_dir}]")
        print(f"  Train: {len(train_scaffolds)} scaffolds (80%)")
        print(f"  Val:   {len(val_scaffolds)} scaffolds (Fixed 100 from Test Pool)")
        print(f"  Test:  {len(test_scaffolds)} scaffolds (Remainder of Test Pool)")

        train_avg, train_min, train_max = get_atom_stats(train_scaffolds)
        val_avg, val_min, val_max = get_atom_stats(val_scaffolds)
        test_avg, test_min, test_max = get_atom_stats(test_scaffolds)

        print(f"  Avg Atoms Train Scaffolds: {train_avg:.2f}")
        print(f"  Min Atoms Train Scaffolds: {train_min}")
        print(f"  Max Atoms Train Scaffolds: {train_max}")

        print(f"  Avg Atoms Val Scaffolds:   {val_avg:.2f}")
        print(f"  Min Atoms Val Scaffolds:   {val_min}")
        print(f"  Max Atoms Val Scaffolds:   {val_max}")

        print(f"  Avg Atoms Test Scaffolds:  {test_avg:.2f}")
        print(f"  Min Atoms Test Scaffolds:  {test_min}")
        print(f"  Max Atoms Test Scaffolds:  {test_max}")

    print("\nDone!")


if __name__ == "__main__":
    main()