from objective_predictor.Prodrug.base_objective import BaseObjective
from rdkit import Chem
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors


class BBBObjective(BaseObjective):
    """Objective class to evaluate Blood-Brain Barrier (BBB) permeability of prodrugs

    It calculates a reward based on these components:
    1. LogP Change: We want to increase lipophilicity i.e. make it fattier.
    2. TPSA Change: We want to decrease TPSA i.e. make it less polar.
    3 Add Ester: We want to add a cleavable ester group.
    """

    def __init__(self,
                 weight_logp_delta: float = 1.0,
                 weight_tpsa_delta: float = 1.0,
                 weight_ester: float = 1.0):
        self.weight_logp_delta = weight_logp_delta
        self.weight_tpsa_delta = weight_tpsa_delta
        self.weight_ester = weight_ester

        self.ester_smarts = Chem.MolFromSmarts('[CX3](=O)O[CX4]') # Ester functional group SMARTS

    def _calculate_property_delta(self, mol_gen, mol_parent):
        logp_parent = Crippen.MolLogP(mol_parent)
        logp_gen = Crippen.MolLogP(mol_gen)
        logp_delta = logp_gen - logp_parent  # change in logP -> +iv is better

        tpsa_parent = MolDescriptors.CalcTPSA(mol_parent)
        tpsa_gen = MolDescriptors.CalcTPSA(mol_gen)
        tpsa_delta = tpsa_parent - tpsa_gen  # change in TPSA -> +iv is better

        return {
            'logp_parent': logp_parent,
            'logp_gen': logp_gen,
            'logp_delta': logp_delta,
            'tpsa_parent': tpsa_parent,
            'tpsa_gen': tpsa_gen,
            'tpsa_delta': tpsa_delta
        }

    def _calculate_cleavable_ester_reward(self, mol_gen, mol_parent) -> float:
        """Check if new easter bond(s) was added in the generated molecule."""
        parent_ester_count = len(mol_parent.GetSubstructMatches(self.ester_smarts))
        gen_ester_count = len(mol_gen.GetSubstructMatches(self.ester_smarts))

        if gen_ester_count > parent_ester_count:
            return 1.0  # Reward for adding ester
        else:
            return 0.0

    def calculate(self, generated_smiles: Chem.Mol, parent_smiles: Chem.Mol) -> dict:
        prop_deltas = self._calculate_property_delta(generated_smiles, parent_smiles)
        reward_logp = prop_deltas['logp_delta'] * self.weight_logp_delta
        reward_tpsa = prop_deltas['tpsa_delta'] * self.weight_tpsa_delta

        reward_added_ester = (self._calculate_cleavable_ester_reward(generated_smiles, parent_smiles)
                              * self.weight_ester)

        total_score = reward_logp + reward_tpsa + reward_added_ester

        return {
            'total_reward': total_score,
            'reward_logp': reward_logp,
            'reward_tpsa': reward_tpsa,
            'reward_added_ester': reward_added_ester,
            'metrics': {
                **prop_deltas,
                'num_ester_added': reward_added_ester
            }
        }


