import ray
# import torch
import numpy as np
from typing import List, Union, Dict
from rdkit import Chem, RDLogger
from tdc import Oracle
from config import MoleculeConfig
from molecule_design import MoleculeDesign


@ray.remote
class CentralOracle:
    """
    A Singleton Ray Actor that maintains the Global Cache and Budget.
    All workers send SMILES here. This ensures:
    1. We never re-evaluate duplicates (Sample Efficiency).
    2. We count exactly 10,000 unique evaluations.
    """

    def __init__(self, config: MoleculeConfig):
        self.config = config
        # PMO State
        self.global_cache: Dict[str, float] = {}
        self.unique_calls = 0
        self.budget_exceeded = False
        self.max_budget = 10000  # Hard PMO limit

        # Initialize TDC Oracle
        try:
            print(f"[CentralOracle] Initializing TDC Oracle: {self.config.objective_type}")
            self.oracle = Oracle(name=self.config.objective_type)
        except Exception as e:
            print(f"[CentralOracle] Error initializing oracle: {e}")
            self.oracle = None

    def get_budget_status(self):
        """Returns (current_usage, is_exceeded)"""
        return self.unique_calls, self.budget_exceeded

    def predict_batch(self, smiles_list: List[str]) -> List[float]:
        """
        Evaluates a batch of SMILES.
        - If in cache: return cached value (Cost = 0).
        - If new: evaluate, add to cache, increment counter (Cost = 1).
        - If budget exceeded: return -infinity.
        """
        results = []

        for smi in smiles_list:
            # 1. Check Validity & Canonicalize
            if not smi:
                results.append(float("-inf"))
                continue

            try:
                canon_smi = Chem.CanonSmiles(smi)
            except:
                results.append(float("-inf"))
                continue

            # --- FIX: Explicitly ignore trivial "C" (Methane) ---
            # This prevents trivial initialization prompts from consuming budget
            if canon_smi == "C":
                results.append(float("-inf"))
                continue

            # 2. Check Cache (Free)
            if canon_smi in self.global_cache:
                results.append(self.global_cache[canon_smi])
                continue

            # 3. Check Budget
            if self.unique_calls >= self.max_budget:
                self.budget_exceeded = True
                results.append(float("-inf"))  # Soft fail for search
                continue

            # 4. Evaluate (Cost)
            try:
                score = self.oracle(canon_smi)
                # PMO specific: some metrics might need sign flipping?
                # TDC usually returns maximization scores, which aligns with your code.
            except:
                score = float("-inf")

            self.global_cache[canon_smi] = score
            self.unique_calls += 1
            results.append(score)

        return results


# Helper wrapper for local objects to talk to the remote actor easily
class RemoteEvaluatorProxy:
    def __init__(self, actor_handle):
        self.actor = actor_handle

    def predict_objective(self, molecule_designs: List[Union[MoleculeDesign, str]]) -> np.array:
        # 1. Extract SMILES
        smiles_input = []
        for mol in molecule_designs:
            if isinstance(mol, MoleculeDesign):
                # --- FIX: Respect the infeasibility_flag ---
                # MoleculeDesign.finalize() already sets infeasibility_flag=True if smiles=="C"
                # We must check this flag so we don't even send it to the actor.
                if mol.synthesis_done and not mol.infeasibility_flag:
                    smiles_input.append(mol.smiles_string)
                else:
                    smiles_input.append("")
            else:
                smiles_input.append(mol)

        # 2. Call Remote Actor
        # We use ray.get to block until result is ready (synchronous from worker perspective)
        scores = ray.get(self.actor.predict_batch.remote(smiles_input))

        # 3. Assign back to objects
        for i, mol in enumerate(molecule_designs):
            if isinstance(mol, MoleculeDesign):
                mol.objective = scores[i]

        return np.array(scores)