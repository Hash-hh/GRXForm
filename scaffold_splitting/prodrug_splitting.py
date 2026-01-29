import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# For Visualization
from sklearn.manifold import TSNE

# --- CONFIGURATION ---
SMILES_FILE = r"../data/zinc/zinc.smiles"
OUTPUT_DIR = "prodrug_splits"
SEEDS = [42]
MIN_VAL_SIZE = 200
MIN_STANDARD_TEST_SIZE = 500  # Size for the quantitative benchmark

# --- THE "HERO" TEST SET (REAL PARENTS) ---
# These are polar drugs that historically needed prodrug strategies.
# We use them as "Inputs" to see if the model can "Fix" them.
HERO_TEST_SET = [
    # (SMILES, Name)
    ("OC1=C(O)C=CC(CCN)=C1", "Dopamine"),  # Needs BBB crossing
    ("NCC1=CC=C(C=C1)C(O)=O", "GABA_Analog"),  # Polar zwitterion
    ("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1c5", "Morphine"),  # Poor oral bioavailability
    ("OC(=O)C1=CC=CC=C1O", "Salicylic_Acid"),  # GI irritation, Aspirin parent
    ("CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)C3=CC=CC=C3)C(=O)N2[C@H]1C(O)=O", "Ampicillin"),  # Pivampicillin parent
    ("NC1=NC2=C(N1)C(=O)NC(N)=N2", "Acyclovir_Base"),  # Valacyclovir parent (poor absorption)
    ("O=c1[nH]cc(F)c(=O)[nH]1", "5-Fluorouracil"),  # Tegafur parent
    ("OC1=CC=CC=C1", "Phenol_Generic"),  # Simple control
]


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None


def is_prodrug_candidate(mol):
    """
    Filters for 'Polar Parents':
    1. Low LogP (Needs lipidization)
    2. Has H-Donors (OH/NH) to attach to
    3. Not too huge (MW < 500)
    """
    if mol is None: return False

    # 1. LogP Check (We want polar things, e.g., < 2.5)
    logp = Crippen.MolLogP(mol)
    if logp > 2.5: return False

    # 2. Functional Handle Check (Need OH or NH to attach ester/amide)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    if hbd == 0: return False

    # 3. Size Check (Leave room for the tail)
    mw = Descriptors.MolWt(mol)
    if mw > 500: return False

    return True


def compute_fingerprints(smiles_list):
    """Generates FPs and keeps valid indices."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_idxs = [i for i, m in enumerate(mols) if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, 1024) for i in valid_idxs]
    return valid_idxs, fps


def memory_efficient_clustering(scaffolds, fps, cutoff=0.4):
    """
    Leader-Follower Clustering (Sphere Exclusion)
    """
    print(f"  Starting Lazy Clustering on {len(scaffolds)} scaffolds...")

    # Sort by length (proxy for complexity/size)
    scaffold_idxs = list(range(len(scaffolds)))
    scaffold_idxs.sort(key=lambda i: len(scaffolds[i]), reverse=True)

    clusters = []
    assigned = set()
    pbar = tqdm(total=len(scaffolds), desc="Clustering")

    for leader_idx in scaffold_idxs:
        if leader_idx in assigned:
            continue

        cluster = [leader_idx]
        assigned.add(leader_idx)
        pbar.update(1)

        leader_fp = fps[leader_idx]
        sims = DataStructs.BulkTanimotoSimilarity(leader_fp, fps)

        for i, score in enumerate(sims):
            if i not in assigned and score >= cutoff:
                cluster.append(i)
                assigned.add(i)
                pbar.update(1)

        clusters.append(cluster)

    pbar.close()
    return clusters


def run_visual_analysis(train_mols, val_mols, test_std_mols, hero_mols, output_path):
    print("  Generating t-SNE visualization...")
    LIMIT = 2000

    def sample_and_fp(mols, label):
        if len(mols) > LIMIT:
            sampled = random.sample(mols, LIMIT)
        else:
            sampled = mols

        _, fps = compute_fingerprints(sampled)
        arrs = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            arrs.append(arr)
        return np.array(arrs), [label] * len(arrs)

    X_tr, y_tr = sample_and_fp(train_mols, "Train")
    X_va, y_va = sample_and_fp(val_mols, "Val")
    X_te, y_te = sample_and_fp(test_std_mols, "Test (ZINC Holdout)")
    X_hero, y_hero = sample_and_fp(hero_mols, "Test (Hero Drugs)")

    if len(X_tr) == 0: return

    X = np.vstack([X_tr, X_va, X_te, X_hero])
    y = y_tr + y_va + y_te + y_hero

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))

    # Plot Train/Val/StandardTest as clouds
    n_cloud = len(y_tr) + len(y_va) + len(y_te)
    sns.scatterplot(x=X_embedded[:n_cloud, 0],
                    y=X_embedded[:n_cloud, 1],
                    hue=y[:n_cloud],
                    palette={"Train": "lightgrey", "Val": "lightblue", "Test (ZINC Holdout)": "orange"},
                    alpha=0.6, s=15)

    # Plot Hero Drugs as large red stars
    sns.scatterplot(x=X_embedded[n_cloud:, 0],
                    y=X_embedded[n_cloud:, 1],
                    color='red', marker='*', s=300, label="HERO Test Set")

    plt.title("Chemical Space: Train vs Holdout vs Hero Targets")
    plt.savefig(output_path)
    plt.close()


def main():
    print(f"--- Loading ZINC from {SMILES_FILE} ---")
    if not os.path.exists(SMILES_FILE):
        print("Error: File not found.")
        return

    with open(SMILES_FILE, 'r') as f:
        raw_smiles = [line.strip().split()[0] for line in f if line.strip()]

    # 1. PREPARE HERO TEST SET (Canonize)
    print("Preparing Hero Test Set...")
    hero_smiles = []
    hero_canon_set = set()
    for s, name in HERO_TEST_SET:
        m = Chem.MolFromSmiles(s)
        if m:
            can = Chem.MolToSmiles(m, canonical=True)
            hero_smiles.append(can)
            hero_canon_set.add(can)

    print(f"Hero Test Set Size: {len(hero_smiles)}")

    # 2. FILTER ZINC (Prodrug Candidates)
    print("Filtering ZINC for Prodrug Candidates (Polar, H-Donors)...")
    train_pool = []

    for smi in tqdm(raw_smiles):
        mol = Chem.MolFromSmiles(smi)
        if is_prodrug_candidate(mol):
            can = Chem.MolToSmiles(mol, canonical=True)
            # CRITICAL: Exclude Hero Set from Pool
            if can not in hero_canon_set:
                train_pool.append(can)

    print(f"Filtered ZINC Pool Size: {len(train_pool)}")

    # 3. SCAFFOLD CLUSTERING
    print("Computing Scaffolds...")
    scaffold_to_molecules = defaultdict(list)
    for smi in tqdm(train_pool):
        scaffold = get_scaffold(smi)
        if scaffold:
            scaffold_to_molecules[scaffold].append(smi)

    unique_scaffolds = list(scaffold_to_molecules.keys())
    print(f"Unique Scaffolds: {len(unique_scaffolds)}")

    print("Clustering Scaffolds...")
    _, scaffold_fps = compute_fingerprints(unique_scaffolds)
    clusters = memory_efficient_clustering(unique_scaffolds, scaffold_fps, cutoff=0.4)

    # 4. SPLIT GENERATION
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in SEEDS:
        print(f"\nProcessing Seed {seed}...")
        random.seed(seed)
        random.shuffle(clusters)

        train_scaffolds, val_scaffolds, test_standard_scaffolds = [], [], []
        counts = {'val': 0, 'test_std': 0}

        # Logic:
        # 1. Fill Test Standard (Holdout ZINC)
        # 2. Fill Val
        # 3. Rest to Train

        for cluster in clusters:
            scaffs = [unique_scaffolds[i] for i in cluster]
            current_mols_count = sum([len(scaffold_to_molecules[s]) for s in scaffs])

            if counts['test_std'] < MIN_STANDARD_TEST_SIZE:
                test_standard_scaffolds.extend(scaffs)
                counts['test_std'] += current_mols_count
            elif counts['val'] < MIN_VAL_SIZE:
                val_scaffolds.extend(scaffs)
                counts['val'] += current_mols_count
            else:
                train_scaffolds.extend(scaffs)

        # Expand to molecules
        def get_mols(scaff_list):
            return [m for s in scaff_list for m in scaffold_to_molecules[s]]

        train_mols = get_mols(train_scaffolds)
        val_mols = get_mols(val_scaffolds)
        test_standard_mols = get_mols(test_standard_scaffolds)  # ZINC Holdout

        # Hero set is fixed
        hero_mols = hero_smiles

        print(
            f"  [Stats] Train: {len(train_mols)} | Val: {len(val_mols)} | Test (ZINC): {len(test_standard_mols)} | Hero: {len(hero_mols)}")

        # Save
        run_dir = os.path.join(OUTPUT_DIR, f"run_seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Drop-in replacement files
        with open(os.path.join(run_dir, "train_molecules.txt"), 'w') as f:
            f.write('\n'.join(train_mols))
        with open(os.path.join(run_dir, "val_molecules.txt"), 'w') as f:
            f.write('\n'.join(val_mols))

        # SAVE BOTH TEST SETS
        # 1. The Standard one for Tables (so your code doesn't crash expecting a big file)
        with open(os.path.join(run_dir, "test_molecules.txt"), 'w') as f:
            f.write('\n'.join(test_standard_mols))

        # 2. The Hero one for Case Studies (Special file)
        with open(os.path.join(run_dir, "test_molecules_hero.txt"), 'w') as f:
            f.write('\n'.join(hero_mols))

        # Visual Proof (Now includes all 4 sets)
        run_visual_analysis(train_mols, val_mols, test_standard_mols, hero_mols,
                            os.path.join(run_dir, "chemical_space_final.png"))


if __name__ == "__main__":
    main()
