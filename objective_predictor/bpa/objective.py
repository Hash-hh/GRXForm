import os
import warnings
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# from tdc.single_pred import Tox
# from tdc.oracles import Oracle

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# --- Setup and Configuration ---

warnings.filterwarnings("ignore")

try:
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    MODULE_DIR = os.getcwd()  # Fallback for interactive environments

MODELS_DIR = os.path.join(MODULE_DIR, 'bpa_surrogate_models')
DATA_DIR = os.path.join(MODULE_DIR, 'bpa_training_data')


# --- Helper Functions ---

def get_morgan_fingerprint(smiles_string):
    """Generates a Morgan fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)


def normalize_value(value, min_val, max_val):
    """Normalizes a value to a 0-1 scale."""
    if max_val == min_val:
        return 0.5  # Avoid division by zero
    if value > max_val: return 1.0
    if value < min_val: return 0.0
    return (value - min_val) / (max_val - min_val)


# --- The Main Scorer Class ---

class BPA_Alternative_Scorer:
    """
    A self-contained class to handle all objective scoring for BPA alternatives.
    It's initialized once, loading all surrogate models into memory.
    """

    def __init__(self, config=None):
        print("Initializing BPA_Alternative_Scorer...")
        self.config = config

        # --- 1. Define Model Paths ---
        # We are only using models with real data for this demo.
        self.model_paths = {
            'er_alpha': os.path.join(MODELS_DIR, 'er_alpha_model.pkl'),
            'ar': os.path.join(MODELS_DIR, 'ar_antagonist_model.pkl'),

            # TODO: Add paths for new models here as you train them
            # 'tg': os.path.join(MODELS_DIR, 'tg_model.pkl'),
            # 'modulus': os.path.join(MODELS_DIR, 'modulus_model.pkl'),
            # 'gper1': os.path.join(MODELS_DIR, 'gper1_model.pkl'),
            # 'err_gamma': os.path.join(MODELS_DIR, 'err_gamma_model.pkl'),
            # 'persistence': os.path.join(MODELS_DIR, 'persistence_model.pkl'),
        }

        # --- 2. Define *Relative* Weights for the Objective Function ---
        # These represent relative importance. They will be normalized to sum to 1.
        self.relative_weights = {
            'w_sa': 1.0,  # Synthetic Accessibility
            'w_er_alpha': 2.0,  # Estrogen Receptor (critical safety)
            'w_ar': 2.0,  # Androgen Receptor (critical safety)

            # TODO: Add weights for new objectives here
            # 'w_tg': 1.0,
            # 'w_modulus': 1.0,
            # 'w_gper1': 1.5,
            # 'w_err_gamma': 1.5,
            # 'w_persistence': 1.0,
        }

        # --- Weight normalization ---
        total_weight = sum(self.relative_weights.values())
        self.weights = {key: val / total_weight for key, val in self.relative_weights.items()}
        print(f"Using {len(self.weights)} objectives.")
        print(f"Total relative weight: {total_weight:.1f}. Normalized weights used.")

        # --- 3. Define Normalization Ranges for Each Objective ---
        self.norm_ranges = {
            'sa': (1, 10),  # Synthetic Accessibility Score (lower is better)

            # TODO: Add normalization ranges for new regression models here
            # 'tg': (-150, 500),      # Glass Transition Temp in °C
            # 'modulus': (0, 10),     # Young's Modulus in GPa
        }

        # --- 4. Load Pre-trained Surrogate Models ---
        self.models = {}
        print("Loading surrogate models...")
        for name, path in self.model_paths.items():
            try:
                self.models[name] = joblib.load(path)
            except FileNotFoundError:
                raise RuntimeError(
                    f"Model file not found: {path}. Please run this script directly to train models first.")

        # --- 5. Load TDC Oracle for Calculated Properties ---
        self.sa_oracle = Oracle(name='SA')
        print("✅ BPA_Alternative_Scorer initialized successfully.")

    def calculate_score(self, smiles):
        """Calculates the full objective score for a single SMILES string."""
        fp = get_morgan_fingerprint(smiles)
        if fp is None:
            return 0.0  # Invalid SMILES should get the worst possible score
        fp = fp.reshape(1, -1)

        # --- Get Raw Predictions from all models ---
        raw_preds = {}
        f = {}  # This holds the f_i(m) scores (0-1, higher is better)

        # --- Models (Classification) ---
        # predict_proba(fp)[0][1] gets the probability of class 1 (the "bad" class)
        raw_preds['er_alpha'] = self.models['er_alpha'].predict_proba(fp)[0][1]
        raw_preds['ar'] = self.models['ar'].predict_proba(fp)[0][1]

        # Transform: f(m) = 1 - probability_of_bad_class
        f['er_alpha'] = 1.0 - raw_preds['er_alpha']
        f['ar'] = 1.0 - raw_preds['ar']

        # --- Calculated Properties (Oracle) ---
        raw_preds['sa'] = self.sa_oracle(smiles)

        # Normalize & Transform: f(m) = 1 - normalized_score
        f['sa'] = 1.0 - normalize_value(raw_preds['sa'], *self.norm_ranges['sa'])

        # --- TODO: Add predictions for new models here ---
        # Example for a regression model (like Tg):
        # raw_preds['tg'] = self.models['tg'].predict(fp)[0]
        # f['tg'] = normalize_value(raw_preds['tg'], *self.norm_ranges['tg'])

        # Example for a classification model (like Persistence):
        # raw_preds['persistence'] = self.models['persistence'].predict_proba(fp)[0][1]
        # f['persistence'] = 1.0 - raw_preds['persistence']
        # --- End TODO ---

        # --- Calculate Final Weighted Score ---
        # F(m) = sum(w_i * f_i(m))
        final_score = (
                self.weights['w_sa'] * f['sa'] +
                self.weights['w_er_alpha'] * f['er_alpha'] +
                self.weights['w_ar'] * f['ar']

            # TODO: Add new weighted scores to the sum here
            # + self.weights['w_tg'] * f['tg']
            # + self.weights['w_persistence'] * f['persistence']
        )

        return final_score

    def __call__(self, smiles_list: list):
        """Calculates scores for a batch of SMILES strings."""
        scores = [self.calculate_score(smi) for smi in smiles_list]
        return np.array(scores)


# --- Model Training and Setup Function ---

def train_and_save_models():
    """
    Checks if surrogate models exist. If not, it trains them using data
    from TDC or placeholder CSV files and saves them to the models directory.
    """
    print("--- Checking for BPA Alternative Surrogate Models ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Define model training tasks ---
    # Only training models for which we have real data.
    tasks = [
        {
            'name': 'er_alpha',  # For f_ERalpha
            'path': os.path.join(MODELS_DIR, 'er_alpha_model.pkl'),
            'type': 'classifier',
            'source': 'tdc',
            'dataset': 'Tox21',
            'label_name': 'NR-ER-LBD'  # Estrogen Receptor Ligand Binding Domain
        },
        {
            'name': 'ar',  # For f_AR
            'path': os.path.join(MODELS_DIR, 'ar_antagonist_model.pkl'),
            'type': 'classifier',
            'source': 'tdc',
            'dataset': 'Tox21',
            'label_name': 'NR-AR'  # Androgen Receptor (agonist assay)
        },

        # TODO: Add new model training tasks here as you find data
        # Example for a dummy CSV-based model:
        # {
        #     'name': 'tg',
        #     'path': os.path.join(MODELS_DIR, 'tg_model.pkl'),
        #     'type': 'regressor',
        #     'source': 'csv',
        #     'data_file': 'tg_data.csv',
        #     'smiles_col': 'smiles',
        #     'label_col': 'tg'
        # },
    ]

    for task in tasks:
        if not os.path.exists(task['path']):
            print(f"Model for '{task['name']}' not found. Training new model...")

            # --- Load Data ---
            try:
                if task['source'] == 'tdc':
                    print(f"  -> Loading REAL data from TDC '{task['dataset']}' for target '{task['label_name']}'...")
                    data = Tox(name=task['dataset'], label_name=task['label_name'])
                    df = data.get_data()
                    smiles_col, label_col = 'Drug', 'Y'
                    df = df.dropna(subset=[label_col])  # Remove rows with missing labels
                    print(f"  -> Found {len(df)} valid data points.")

                elif task['source'] == 'csv':
                    # This block is now a placeholder for when you add new CSV tasks
                    data_path = os.path.join(DATA_DIR, task['data_file'])
                    smiles_col, label_col = task['smiles_col'], task['label_col']

                    if not os.path.exists(data_path):
                        print(f"  -> WARNING: Data file '{task['data_file']}' not found.")
                        print(f"  -> Creating a DUMMY placeholder file at '{data_path}'.")
                        print(f"  -> PLEASE REPLACE THIS with your actual training data.")
                        if task['type'] == 'regressor':
                            dummy_data = {
                                smiles_col: ['c1ccccc1', 'CC', 'CCO', 'c1cnccc1', 'C(=O)O', 'CCN'],
                                label_col: [100, -50, 0, 150, 50, -20]
                            }
                        else:
                            dummy_data = {
                                smiles_col: ['c1ccccc1', 'CC', 'CCO', 'c1cnccc1', 'C[C@H](O)c1ccccc1',
                                             'O=C(O)c1ccccc1'],
                                label_col: [1, 0, 0, 1, 1, 0]
                            }
                        pd.DataFrame(dummy_data).to_csv(data_path, index=False)

                    df = pd.read_csv(data_path)

            except Exception as e:
                print(f"  -> ERROR: Could not load data for '{task['name']}'. Skipping. Error: {e}")
                continue

                # --- Preprocess Data ---
            df['fingerprint'] = df[smiles_col].apply(get_morgan_fingerprint)
            df = df.dropna(subset=['fingerprint', label_col]).reset_index(drop=True)

            X = np.array(df['fingerprint'].tolist())
            y = df[label_col].values

            if len(X) < 1:
                print("  -> ERROR: No data at all. Skipping.")
                continue

            # --- Train Model ---
            if task['type'] == 'regressor':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:  # classifier
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')

            model.fit(X, y)

            # --- Save Model ---
            joblib.dump(model, task['path'])
            print(f"✅ Saved '{task['name']}' model to {task.get('path')}")
        else:
            print(f"✅ Found existing model for '{task['name']}' at {task['path']}")


# --- Main Execution Block ---

if __name__ == '__main__':
    print("Running setup for BPA Alternative Scorer...")
    # This will train and save the 2 real models from Tox21.
    # It will download Tox21 data if needed, which may take a moment.
    train_and_save_models()

    print("\n--- Example Usage ---")
    try:
        scorer = BPA_Alternative_Scorer()

        test_smiles = [
            'CC(C)(c1ccc(O)cc1)c2ccc(O)cc2',  # Bisphenol A (BPA)
            'O=S(=O)(c1ccc(O)cc1)c2ccc(O)cc2',  # Bisphenol S (BPS)
            'C1C[C@@H]2[C@H](O1)C[C@H](O)CO2'  # Isosorbide (a safe bio-based alternative)
        ]

        # Calculate scores
        scores = scorer(test_smiles)

        print("\n--- Scoring Results (Higher is Better) ---")
        print(f"--- (Based on SA, ER-alpha, and AR only) ---")
        for smi, score in zip(test_smiles, scores):
            print(f"SMILES: {smi}\nScore: {score:.4f}\n")

    except Exception as e:
        print(f"\n--- ERROR during scorer initialization or use ---")
        print(f"An error occurred: {e}")
        print("This likely means a model failed to train or TDC data failed to download.")