import time
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "./data/FDB-17-fragmentset.smi"
OUTPUT_FILE = "./data/FDB-17-filtered.txt"

# Define the vocabulary filter up-front
allowed_vocabulary = [  # put multi-character occurences first
        "[NH3+]","[SH+]","[C@]","[O+]","[NH+]","[nH+]","[C@@H]","[CH2-]","[C@H]","[NH2+]","[S+]","[CH-]","[S@]","[N-]",
        "[s+]","[nH]","[S@@]","[n+]","[o+]","[NH-]","[C@@]","[S-]","[N+]","[OH+]","[O-]","[n-]",
        "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7",
        "C", "O", "S", "s", "F", "P", "p", "Cl", "Br", "I"
    ]


def is_valid_vocabulary(smiles: str) -> bool:
    """
    Checks if a SMILES string is composed *only* of the allowed vocabulary
    by replacing all allowed tokens and checking if the result is empty.
    """
    temp = smiles
    for voc in allowed_vocabulary:
        temp = temp.replace(voc, "")
    # If the string is empty after all replacements, it's valid.
    return len(temp) == 0


# --- Main Processing ---
print(f"Starting filtering of {INPUT_FILE}...")
all_filtered_smiles = []
lines_processed = 0

# Get a total line count for a nice progress bar
try:
    with open(INPUT_FILE, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Found {total_lines:,} total lines in input file.")
except Exception as e:
    print(f"Could not count lines, progress bar will not show a total. Error: {e}")
    total_lines = 0

start_time = time.time()

# Read and filter in one pass
with open(INPUT_FILE, 'r') as f:
    for i, line in enumerate(tqdm(f, total=total_lines, desc="Reading and filtering")):
        # if i == 0:  # Skip header
        #     continue

        lines_processed += 1

        try:
            s = line.rstrip().split()
            smiles = s[0]

            # Apply both filters at once:
            # 1. Must not be empty
            # 2. Must pass the vocabulary check
            if len(smiles) > 0 and is_valid_vocabulary(smiles):
                all_filtered_smiles.append(smiles)

        except IndexError:
            # Handle potential blank lines or lines without a tab
            continue

end_time = time.time()

print("---")
print(f"Processing complete in {end_time - start_time:.2f} seconds.")
print(f"Processed {lines_processed:,} SMILES lines.")
print(f"Found {len(all_filtered_smiles):,} valid and filtered SMILES.")

# Save the single, huge output file
print(f"Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    # Use \n join for a potentially faster write
    f.write("\n".join(all_filtered_smiles))

print("========\nDone.")