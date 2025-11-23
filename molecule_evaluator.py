import ray
# import torch
import numpy as np
from typing import List, Union, Dict
from rdkit import Chem, RDLogger
from tdc import Oracle
from config import MoleculeConfig
from molecule_design import MoleculeDesign

import ray
import torch
import numpy as np
import os
import time
from typing import List, Union, Dict, Tuple
from rdkit import Chem, RDLogger
from tdc import Oracle
from config import MoleculeConfig
from molecule_design import MoleculeDesign


@ray.remote
class CentralOracle:
    """
    A Singleton Ray Actor that maintains the Global Cache and Budget.
    Logs Top-10 progress continuously for precise PMO AUC calculation.
    """

    def __init__(self, config: MoleculeConfig):
        self.config = config
        # PMO State
        self.global_cache: Dict[str, float] = {}
        self.unique_calls = 0
        self.budget_exceeded = False
        self.max_budget = 10000

        # Tracking for AUC
        self.all_valid_molecules: List[Tuple[float, str]] = []  # (score, smiles)
        self.log_file_path = os.path.join(self.config.results_path, "pmo_tracker.txt")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        # Initialize log file
        with open(self.log_file_path, "w") as f:
            f.write("Calls\tTop10_Avg\tTop10_Molecules\n")

        # Initialize TDC Oracle
        try:
            print(f"[CentralOracle] Initializing TDC Oracle: {self.config.objective_type}")
            self.oracle = Oracle(name=self.config.objective_type)
        except Exception as e:
            print(f"[CentralOracle] Error initializing oracle: {e}")
            self.oracle = None

    def _update_tracker(self):
        """
        Updates the Top-10 list and logs the current state to file.
        """
        # Sort by score descending
        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)

        # Keep top 100 internally to save memory if run is long,
        # but strictly we only need top 10 for the metric.
        if len(self.all_valid_molecules) > 100:
            self.all_valid_molecules = self.all_valid_molecules[:100]

        top_10 = self.all_valid_molecules[:10]

        if not top_10:
            avg_score = 0.0
            mol_str = ""
        else:
            avg_score = sum(x[0] for x in top_10) / len(top_10)
            mol_str = ";".join([f"{x[1]}:{x[0]:.4f}" for x in top_10])

        # Log Format: UniqueCalls <TAB> Top10Mean <TAB> MolData
        with open(self.log_file_path, "a") as f:
            f.write(f"{self.unique_calls}\t{avg_score:.6f}\t{mol_str}\n")

    def get_budget_status(self):
        """Returns (current_usage, is_exceeded)"""
        return self.unique_calls, self.budget_exceeded

    def predict_batch(self, smiles_list: List[str]) -> List[float]:
        """
        Evaluates a batch of SMILES.
        Logs progress after processing the batch.
        """
        results = []
        new_unique_found = False

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

            # Ignore trivial "C"
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
                results.append(float("-inf"))
                continue

            # 4. Evaluate (Cost)
            try:
                score = self.oracle(canon_smi)
            except:
                score = float("-inf")

            self.global_cache[canon_smi] = score
            self.unique_calls += 1
            results.append(score)

            # Add to tracking list if valid
            if score > float("-inf"):
                self.all_valid_molecules.append((score, canon_smi))
                new_unique_found = True

        # Log update if we found anything new
        if new_unique_found:
            self._update_tracker()

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