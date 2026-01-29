from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, AllChem, DataStructs


class ProdrugMPOObjective:
    def __init__(self):
        # We don't need TDC here, we use RDKit directly for speed
        # SMARTS for cleavable linkers (Esters, Carbonates, Amides)
        # Matches: [Carbon]-C(=O)-[O or N]-[Carbon]
        self.cleavable_pattern = Chem.MolFromSmarts('[#6]C(=O)[O,N][#6]')

    def score(self, smiles, parent_smiles=None):
        """
        Returns the scalar reward for Reinforcement Learning.
        Range: [0.0, 1.0] (roughly, can be slightly higher/lower based on bonuses)

        Note: requires 'parent_smiles' to calculate improvement.
        """
        if not smiles or not parent_smiles:
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        parent = Chem.MolFromSmiles(parent_smiles)
        if not mol or not parent:
            return 0.0

        # 1. Lipidization (LogP) -> Target: Increase LogP
        # We clamp the reward to [0, 1] for stability
        parent_logp = Crippen.MolLogP(parent)
        gen_logp = Crippen.MolLogP(mol)

        logp_diff = gen_logp - parent_logp
        # Reward increases up to a +3.0 boost, then plateaus
        # Using 0.33 factor so +3.0 gives score of 1.0
        r_logp = max(0.0, min(logp_diff * 0.33, 1.0))

        # 2. Masking (HBD Reduction) -> Target: Reduce HBD
        parent_hbd = rdMolDescriptors.CalcNumHBD(parent)
        gen_hbd = rdMolDescriptors.CalcNumHBD(mol)

        # 1.0 if we reduced HBD, 0.0 if not.
        # (Or 0.5 per HBD removed if you want smoother gradients)
        r_mask = 1.0 if gen_hbd < parent_hbd else 0.0

        # 3. Structural Integrity (Tanimoto) -> Target: > 0.4
        # We penalize heavily if we lose the parent structure
        fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        fp_parent = AllChem.GetMorganFingerprintAsBitVect(parent, 2)
        sim = DataStructs.TanimotoSimilarity(fp_gen, fp_parent)

        # Soft sigmoid-like clip. If sim < 0.4, reward is 0.
        # If sim > 0.4, it scales up.
        r_sim = 1.0 if sim > 0.4 else 0.0

        # 4. Cleavable Group -> Target: Must have ester/amide
        # Check if we ADDED a new cleavable bond
        p_match = len(parent.GetSubstructMatches(self.cleavable_pattern))
        g_match = len(mol.GetSubstructMatches(self.cleavable_pattern))
        r_chem = 1.0 if g_match > p_match else 0.0

        # Total Weighted Reward (Average)
        # Weights: Masking/LogP are most important for "Prodrug" function
        total = (0.3 * r_logp) + (0.3 * r_mask) + (0.2 * r_sim) + (0.2 * r_chem)

        return total

    def is_successful(self, smiles, parent_smiles=None):
        """
        Strict Binary Evaluation for Success Rate.
        Criteria:
        1. LogP > 2.0 (Lipophilic enough)
        2. HBD Reduced (Masked)
        3. Similarity > 0.4 (Maintained Pharmacophore)
        4. Added Cleavable Bond (Mechanism exists)
        """
        if not smiles or not parent_smiles: return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            parent = Chem.MolFromSmiles(parent_smiles)

            # Criteria 1: Absolute Lipophilicity
            if Crippen.MolLogP(mol) < 2.0: return False

            # Criteria 2: Masking (Must reduce polar groups)
            if rdMolDescriptors.CalcNumHBD(mol) >= rdMolDescriptors.CalcNumHBD(parent): return False

            # Criteria 3: Similarity (Must not be random noise)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(parent, 2)
            if DataStructs.TanimotoSimilarity(fp1, fp2) <= 0.4: return False

            # Criteria 4: Chemical Valid Prodrug Mechanism
            p_match = len(parent.GetSubstructMatches(self.cleavable_pattern))
            g_match = len(mol.GetSubstructMatches(self.cleavable_pattern))
            if g_match <= p_match: return False

            return True

        except:
            return False

    def individual_scores(self, smiles, parent_smiles=None):
        """
        Returns components for analysis/logging.
        """
        if not smiles or not parent_smiles: return {}

        mol = Chem.MolFromSmiles(smiles)
        parent = Chem.MolFromSmiles(parent_smiles)

        return {
            "LogP_Gen": Crippen.MolLogP(mol),
            "LogP_Delta": Crippen.MolLogP(mol) - Crippen.MolLogP(parent),
            "HBD_Gen": rdMolDescriptors.CalcNumHBD(mol),
            "HBD_Delta": rdMolDescriptors.CalcNumHBD(mol) - rdMolDescriptors.CalcNumHBD(parent),
            "Tanimoto": DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2),
                AllChem.GetMorganFingerprintAsBitVect(parent, 2)
            ),
            "QED": Descriptors.qed(mol)
        }


if __name__ == "__main__":
    # Test with Dopamine -> Prodrug
    obj = ProdrugMPOObjective()

    parent = "OC1=C(O)C=CC(CCN)=C1"  # Dopamine (Polar, LogP ~0.1)

    # Fake Prodrug: Added a simple acetyl group (ester)
    # Note: This is chemically simplified for testing
    prodrug = "CC(=O)Oc1ccc(CCN)cc1O"

    score = obj.score(prodrug, parent)
    success = obj.is_successful(prodrug, parent)
    metrics = obj.individual_scores(prodrug, parent)

    print(f"Parent: {parent}")
    print(f"Prodrug: {prodrug}")
    print(f"Score: {score:.3f}")
    print(f"Successful: {success}")
    print(f"Metrics: {metrics}")