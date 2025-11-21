import pandas as pd
import random
from tqdm import tqdm
import os

# Configuration
prepare_data = True
# num_validation = 0
num_validation = 20000
# Update this path to point to your ZINC CSV file
csv_file_path = "./data/zinc/zinc250k.csv"
output_dir = "./data/zinc"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

if prepare_data:
    print(f"Loading SMILES from {csv_file_path}...")

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file_path)

        # Check if 'smiles' column exists (case-insensitive check)
        smiles_col = None
        for col in df.columns:
            if col.strip().lower() == 'smiles':
                smiles_col = col
                break

        if smiles_col:
            print(f"Found SMILES column: '{smiles_col}'")
            # Drop NaNs and empty strings, then convert to list
            all_smiles = df[smiles_col].dropna().astype(str).tolist()
            all_smiles = [s.strip() for s in all_smiles if len(s.strip()) > 0]
        else:
            # Fallback: Assume the first column contains SMILES if no header matches
            print("Warning: 'smiles' column header not found. Using the first column as SMILES.")
            all_smiles = df.iloc[:, 0].dropna().astype(str).tolist()
            all_smiles = [s.strip() for s in all_smiles if len(s.strip()) > 0]

        print(f"Loaded {len(all_smiles)} SMILES")

        print("Shuffling data...")
        random.shuffle(all_smiles)

        # Adjust num_validation if dataset is smaller than expected
        if len(all_smiles) < num_validation:
            print(
                f"Warning: Dataset size ({len(all_smiles)}) is smaller than requested validation size ({num_validation}).")
            print("Using 10% for validation instead.")
            num_validation = int(len(all_smiles) * 0.1)

        print(f"Saving validation set ({num_validation} molecules)")
        with open(f"{output_dir}/zinc_valid.smiles", 'w') as f:
            for line in all_smiles[:num_validation]:
                f.write(f"{line}\n")

        print(f"Saving training set ({len(all_smiles) - num_validation} molecules)")
        with open(f"{output_dir}/zinc_train.smiles", 'w') as f:
            for line in all_smiles[num_validation:]:
                f.write(f"{line}\n")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        exit(1)
    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")
        exit(1)

print("========")
# The rest of the filtering logic remains the same
for datatype in ["train", "valid"]:
    print("Processing", datatype)

    input_path = f"{output_dir}/zinc_{datatype}.smiles"
    output_path = f"{output_dir}/zinc_{datatype}_filtered.smiles"

    if not os.path.exists(input_path):
        print(f"Skipping {datatype}: {input_path} does not exist.")
        continue

    with open(input_path) as f:
        unfiltered_smiles = [line.rstrip() for line in f]

    allowed_vocabulary = [  # put multi-character occurences first
        "[NH3+]", "[SH+]", "[C@]", "[O+]", "[NH+]", "[nH+]", "[C@@H]", "[CH2-]", "[C@H]", "[NH2+]", "[S+]", "[CH-]",
        "[S@]", "[N-]",
        "[s+]", "[nH]", "[S@@]", "[n+]", "[o+]", "[NH-]", "[C@@]", "[S-]", "[N+]", "[OH+]", "[O-]", "[n-]",
        "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7",
        "C", "O", "S", "s", "F", "P", "p", "Cl", "Br", "I"
    ]

    print("unfiltered:", len(unfiltered_smiles))
    filtered_smiles = []
    for mol in tqdm(unfiltered_smiles):
        temp = mol
        for voc in allowed_vocabulary:
            temp = temp.replace(voc, "")
        if len(temp) == 0:
            filtered_smiles.append(mol)

    print("filtered:", len(filtered_smiles))

    with open(output_path, 'w') as f:
        for line in filtered_smiles:
            f.write(f"{line}\n")