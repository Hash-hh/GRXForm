import os
import warnings
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc.single_pred import ADME, Tox
from tdc.oracles import Oracle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")

# Get the directory where this file is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODULE_DIR, 'models')


def get_morgan_fingerprint(smiles_string):
    """Generates a Morgan fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)


# --- The Main Scorer Class ---
class ProdrugObjectiveScorer:
    """
    A self-contained class to handle all prodrug objective scoring.
    It's initialized once per worker, loading all models into memory.
    """

    def __init__(self, config=None):
        print("Initializing ProdrugObjectiveScorer...")
        self.config = config

        # Use module-local paths if config doesn't specify them
        self.sol_model_path = os.path.join(MODELS_DIR, 'solubility_model.pkl') if (
                    config is None or not hasattr(config, 'sol_model_path')) else config.sol_model_path
        self.perm_model_path = os.path.join(MODELS_DIR, 'permeability_model.pkl') if (
                    config is None or not hasattr(config, 'perm_model_path')) else config.perm_model_path
        self.herg_model_path = os.path.join(MODELS_DIR, 'herg_model.pkl') if (
                    config is None or not hasattr(config, 'herg_model_path')) else config.herg_model_path

        self.weights = self.config.prodrug_objective_weights if (
                    config and hasattr(config, 'prodrug_objective_weights')) else {
            'w_amino_acid': 1.0, 'w_perm': 1.0, 'w_sol': 1.0,
            'w_herg': 1.0, 'w_sa': 1.0, 'w_qed': 1.0
        }

        # Load pre-trained data-driven models
        self.sol_model = joblib.load(self.sol_model_path)
        self.perm_model = joblib.load(self.perm_model_path)
        self.herg_model = joblib.load(self.herg_model_path)

        # Load TDC Oracles for calculated properties
        self.sa_oracle = Oracle(name='SA')
        self.qed_oracle = Oracle(name='QED')

        # Compile the SMARTS pattern for the expert heuristic once
        self.amino_acid_ester_pattern = Chem.MolFromSmarts('[#7&X3;!$([#7]-C=O)]-&!@[#6&X4]-&!@C(=O)O')
        print("✅ ProdrugObjectiveScorer initialized successfully in worker.")

    def _has_amino_acid_ester(self, mol):
        if mol is None: return 0
        return 1 if mol.HasSubstructMatch(self.amino_acid_ester_pattern) else 0

    def calculate_score(self, rdkit_mol):
        """Calculates the full objective score for a single RDKit molecule."""
        smiles = Chem.MolToSmiles(rdkit_mol)
        fp = get_morgan_fingerprint(smiles)
        if fp is None: return -999.0
        fp = fp.reshape(1, -1)

        # Get all predictions
        pred_amino_acid = self._has_amino_acid_ester(rdkit_mol)
        pred_sol = self.sol_model.predict(fp)[0]
        pred_perm = self.perm_model.predict(fp)[0]
        pred_herg_prob = self.herg_model.predict_proba(fp)[0, 1]
        pred_sa = self.sa_oracle(smiles)
        pred_qed = self.qed_oracle(smiles)

        # Calculate weighted score using weights from the config
        score = (
                (self.weights['w_amino_acid'] * pred_amino_acid) +
                (self.weights['w_perm'] * pred_perm) +
                (self.weights['w_sol'] * pred_sol) +
                (self.weights['w_herg'] * pred_herg_prob) +
                (self.weights['w_sa'] * pred_sa) +
                (self.weights['w_qed'] * pred_qed)
        )
        return score

    def __call__(self, rdkit_mols: list):
        """Calculates scores for a batch of molecules."""
        scores = [self.calculate_score(mol) for mol in rdkit_mols]
        return np.array(scores)


# --- Model Training and Setup Function ---
def train_and_save_models():
    """
    This function checks if the scoring models exist. If not, it trains them
    using TDC and saves them to the models directory within this module.
    """
    print("--- Checking for Prodrug Scoring Models ---")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Define model training tasks with local paths
    tasks = [
        {'name': 'Solubility', 'path': os.path.join(MODELS_DIR, 'solubility_model.pkl'),
         'class': ADME, 'dataset': 'Solubility_AqSolDB', 'type': 'regressor', 'label': None},
        {'name': 'Permeability', 'path': os.path.join(MODELS_DIR, 'permeability_model.pkl'),
         'class': ADME, 'dataset': 'Caco2_Wang', 'type': 'regressor', 'label': None},
        {'name': 'hERG Toxicity', 'path': os.path.join(MODELS_DIR, 'herg_model.pkl'),
         'class': Tox, 'dataset': 'hERG_Central', 'type': 'classifier', 'label': 'hERG_inhib'},
    ]

    for task in tasks:
        if not os.path.exists(task['path']):
            print(f"Model for {task['name']} not found. Training new model...")

            # Load data
            if task['label']:
                data = task['class'](name=task['dataset'], label_name=task['label'])
            else:
                data = task['class'](name=task['dataset'])

            split = data.get_split()
            X_train = np.array(
                [fp for fp in [get_morgan_fingerprint(smi) for smi in split['train']['Drug']] if fp is not None])
            y_train = split['train']['Y'].values[:len(X_train)]

            # Train model
            if task['type'] == 'regressor':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            model.fit(X_train, y_train)

            # Save model
            joblib.dump(model, task['path'])
            print(f"✅ Saved {task['name']} model to {task['path']}")
        else:
            print(f"✅ Found existing model for {task['name']} at {task['path']}")


if __name__ == '__main__':
    print("Running manual model training for prodrug objective...")
    print(f"Models will be saved to: {MODELS_DIR}")
    train_and_save_models()
