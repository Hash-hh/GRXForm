import math
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs


class GuacaMolHardObjective:
    def __init__(self, task_name="osimertinib"):
        """
        Initializes the Hard MPO task based on the provided report.
        Supported tasks: 'osimertinib', 'fexofenadine', 'ranolazine'.
        """
        self.task_name = task_name.lower()
        self._configure_task()

    def _configure_task(self):
        """
        Sets up targets, thresholds, and scoring parameters
        """
        if self.task_name == "osimertinib":
            # Osimertinib: "The Archetype of Conflicting Constraints" [cite: 37]
            self.target_smiles = "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1N(C)CCN(C)C"
            self.sim_threshold = 0.8  # Strict similarity cliff [cite: 47]
            self.fp_type = "ECFP6"  # Circular fingerprint (Radius 3) [cite: 45]
            self.tpsa_target = 95.0  # Target TPSA [cite: 55]
            self.tpsa_sigma = 20.0  # Sigma for Gaussian [cite: 57]
            self.logp_target = 1.0  # Target LogP (low lipophilicity) [cite: 121]
            self.logp_sigma = 1.0  # Very sharp LogP penalty [cite: 121]

        elif self.task_name == "fexofenadine":
            # Fexofenadine: "The Anti-Target Challenge" [cite: 65]
            # Uses Atom Pairs which are harder to "fool" than ECFP[cite: 74].
            self.target_smiles = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
            self.sim_threshold = 0.8  # [cite: 71]
            self.fp_type = "AtomPairs"  # Topological fingerprint [cite: 70]
            self.tpsa_target = 90.0
            self.tpsa_sigma = 2.0  # "Needle in a haystack" constraint [cite: 79]
            self.logp_target = 4.0
            self.logp_sigma = 2.0  # [cite: 77]

        elif self.task_name == "ranolazine":
            # Ranolazine: "The Combinatorial Cliff" [cite: 82]
            # Includes a discrete Fluorine count requirement[cite: 86].
            self.target_smiles = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
            self.sim_threshold = 0.7  # [cite: 94]
            self.fp_type = "AtomPairs"  # [cite: 94]
            self.tpsa_target = 95.0
            self.tpsa_sigma = 20.0  # [cite: 96]
            self.logp_target = 1.5  # [cite: 95]
            self.logp_sigma = 1.0
            self.require_fluorine = True  # Discrete step function [cite: 89]

        # Pre-compute target fingerprint to speed up 150k calls [cite: 109]
        self.target_mol = Chem.MolFromSmiles(self.target_smiles)
        if self.fp_type == "AtomPairs":
            self.target_fp = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(self.target_mol)
        else:
            # ECFP6 equivalent is Morgan Radius 3
            self.target_fp = AllChem.GetMorganFingerprintAsBitVect(self.target_mol, 3, nBits=2048)

    def score(self, smiles, mode='easy'):
        """
        Calculates the RL Reward.
        modes:
          - 'hard': (Benchmark Standard) Returns 0.0 immediately if sim < threshold[cite: 115].
          - 'soft': (Training Curriculum) Returns partial signal (0.1*sim) if sim < threshold.
          - 'easy': (Study) Returns geometric mean of all props, ignoring thresholds.
        """
        if not smiles:
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0.0

        # --- 1. Calculate Similarity ---
        if self.fp_type == "AtomPairs":
            fp = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)

        sim = DataStructs.TanimotoSimilarity(self.target_fp, fp)

        # --- 2. Handle Modes (The Logic Switch) ---

        # MODE: HARD (The "Fail Fast" approach from report Page 5 [cite: 125])
        if mode == 'hard':
            if sim < self.sim_threshold:
                return 0.0  # Zero gradient to punish surrogate models [cite: 48]

        # MODE: SOFT (Curriculum Learning)
        # Give the agent a "scent" of the scaffold so it can learn to climb the hill.
        elif mode == 'soft':
            if sim < self.sim_threshold:
                return 0.1 * sim  # Small positive reward to guide scaffold finding

        # MODE: Easy (Raw Physics)
        # Calculate full score regardless of similarity threshold.
        # Used to verify if the "cliff" is what makes the task hard.
        elif mode == 'easy':
            pass

            # --- 3. Calculate Physicochemical Scores (Gaussians) ---

        # TPSA Score: MaxGaussian(target, sigma) [cite: 56]
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        tpsa_score = math.exp(-0.5 * ((tpsa - self.tpsa_target) / self.tpsa_sigma) ** 2)

        # LogP Score: MinGaussian/Targeted [cite: 61]
        # Note: LogP is the first value in Crippen descriptors
        logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
        logp_score = math.exp(-0.5 * ((logp - self.logp_target) / self.logp_sigma) ** 2)

        # Ranolazine Special: Fluorine Count [cite: 89]
        f_score = 1.0
        if hasattr(self, 'require_fluorine'):
            # Count F atoms (AtomicNum=9)
            f_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
            if f_count > 0:
                # Gaussian centered at 1 (or more)
                f_score = math.exp(-0.5 * ((f_count - 1) / 1.0) ** 2)
            else:
                # If 'hard' or 'soft', usually this is a cliff.
                # In easy, we might still penalize but let's keep it consistent.
                f_score = 0.0 if mode != 'easy' else 0.01

        # --- 4. Aggregate (Geometric Mean) ---
        # "Soft logical AND" gate [cite: 29]
        components = [sim, tpsa_score, logp_score]
        if hasattr(self, 'require_fluorine'):
            components.append(f_score)

        # Geometric Mean: (a * b * c)^(1/N)
        # Avoid math domain error if any component is 0 (can happen with F_score)
        product = math.prod(components)
        if product <= 0:
            return 0.0

        return product ** (1.0 / len(components))

    def is_successful(self, smiles):
        """
        Binary Success Check based on strict report thresholds.
        Used for calculating Success Rate %.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False

        scores = self.individual_scores(smiles)

        # 1. Similarity Threshold Check [cite: 47, 71]
        if scores['similarity'] < self.sim_threshold:
            return False

        # 2. Property Checks (Within ~1 Sigma is a reasonable success definition)
        # The report implies high scores = success, but for binary metrics:
        tpsa_pass = abs(scores['tpsa_raw'] - self.tpsa_target) <= self.tpsa_sigma
        logp_pass = abs(scores['logp_raw'] - self.logp_target) <= self.logp_sigma

        f_pass = True
        if hasattr(self, 'require_fluorine'):
            f_pass = scores['f_count'] >= 1

        return tpsa_pass and logp_pass and f_pass

    def individual_scores(self, smiles):
        """
        Returns raw values for logging and analysis.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return {}

        # Fingerprint
        if self.fp_type == "AtomPairs":
            fp = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)

        sim = DataStructs.TanimotoSimilarity(self.target_fp, fp)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]

        results = {
            "similarity": sim,
            "tpsa_raw": tpsa,
            "logp_raw": logp
        }

        if hasattr(self, 'require_fluorine'):
            results['f_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)

        return results


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize for Osimertinib
    objective = GuacaMolHardObjective(task_name="osimertinib")

    # 2. Test the Target itself (Should be perfect)
    target_smi = "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1N(C)CCN(C)C"
    print(f"Target Score (Hard): {objective.score(target_smi, mode='hard'):.4f}")

    # 3. Test a bad molecule (Benzene)
    bad_smi = "c1ccccc1"

    # HARD MODE: Should be 0.0 (Fail Fast)
    print(f"Benzene (Hard): {objective.score(bad_smi, mode='hard'):.4f}")

    # SOFT MODE: Should be small non-zero (0.1 * similarity)
    # This guides the agent "You are 10% similar, keep trying"
    print(f"Benzene (Soft): {objective.score(bad_smi, mode='soft'):.4f}")

    # Easy MODE: Calculates the geometric mean ignoring the threshold check.
    # It shows what the score *would* be if we removed the "Hard" cliff.
    print(f"Benzene (Easy): {objective.score(bad_smi, mode='easy'):.4f}")

    # 4. Check Success
    print(f"Is Target Successful? {objective.is_successful(target_smi)}")
