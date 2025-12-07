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
TRAIN_RATIO = 0.8


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
    with open(SMILES_FILE, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(smiles_list)} SMILES.")

    # Step 2: Compute scaffolds and group molecules
    print("--- Computing Scaffolds ---")
    scaffold_to_molecules = defaultdict(list)
    for smi in tqdm(smiles_list, desc="Computing scaffolds"):
        scaffold = get_scaffold(smi)
        if scaffold is not None:
            scaffold_to_molecules[scaffold].append(smi)

    scaffolds = list(scaffold_to_molecules.keys())
    print(f"Found {len(scaffolds)} unique scaffolds.")

    # Step 3: Generate splits for each seed
    print("--- Generating Splits ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in SEEDS:
        print(f"\nProcessing Seed {seed}...")
        random.seed(seed)

        shuffled_scaffolds = scaffolds.copy()
        random.shuffle(shuffled_scaffolds)

        split_idx = int(len(shuffled_scaffolds) * TRAIN_RATIO)
        train_scaffolds = shuffled_scaffolds[:split_idx]
        test_scaffolds = shuffled_scaffolds[split_idx:]

        # Collect molecules
        train_molecules = []
        for scaffold in train_scaffolds:
            train_molecules.extend(scaffold_to_molecules[scaffold])

        test_molecules = []
        for scaffold in test_scaffolds:
            test_molecules.extend(scaffold_to_molecules[scaffold])

        # Save to files
        run_dir = os.path.join(OUTPUT_DIR, f"run_seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "train_molecules.txt"), 'w') as f:
            f.write('\n'.join(train_molecules))

        with open(os.path.join(run_dir, "train_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(train_scaffolds))

        with open(os.path.join(run_dir, "test_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(test_scaffolds))

        with open(os.path.join(run_dir, "test_molecules.txt"), 'w') as f:
            f.write('\n'.join(test_molecules))

        print(f"  [Saved to {run_dir}]")
        print(f"  Train Scaffolds: {len(train_scaffolds)}")
        print(f"  Test Scaffolds:  {len(test_scaffolds)} (Strictly unseen in train)")
        print(f"  Train Molecules: {len(train_molecules)}")
        print(f"  Test Molecules:  {len(test_molecules)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
