import ray
# import torch
import numpy as np
import os
import pickle
from typing import List, Union, Dict, Tuple
from rdkit import Chem, RDLogger
from tdc import Oracle
from config import MoleculeConfig
from molecule_design import MoleculeDesign


@ray.remote
class CentralOracle:
    """
    A Singleton Ray Actor that acts as the Gatekeeper for the Benchmark.
    1. Enforces Sample Efficiency (De-duplication via Cache).
    2. Enforces Budget (10k Limit).
    3. Logs EVERY evaluation for audit/debugging.
    4. Logs Top-10 progress for AUC calculation.
    5. Computes Real-time PMO AUC.
    """

    def __init__(self, config: MoleculeConfig):
        self.config = config

        # --- 1. State Initialization ---
        self.global_cache: Dict[str, float] = {}  # {Canonical_SMILES: Score}
        self.unique_calls = 0
        self.budget_exceeded = False
        self.max_budget = 10000

        # AUC Curve Tracking: List of (calls, top10_avg)
        # Start at (0, 0.0)
        self.auc_curve_points: List[Tuple[int, float]] = [(0, 0.0)]

        # --- 2. Paths ---
        # Stores the raw object to persist state between RL and TASAR phases
        self.cache_file_path = os.path.join(self.config.results_path, "oracle_cache.pkl")
        # Stores EVERY call: ID, SMILES, Score
        self.history_file_path = os.path.join(self.config.results_path, "oracle_history.txt")
        # Stores Top-10 state for AUC curves
        self.tracker_file_path = os.path.join(self.config.results_path, "pmo_tracker.txt")

        os.makedirs(os.path.dirname(self.history_file_path), exist_ok=True)

        # --- 3. Load Previous State (Warm Start for TASAR) ---
        self._load_cache()

        # Initialize logs headers if new run
        if not os.path.exists(self.history_file_path):
            with open(self.history_file_path, "w") as f:
                f.write("Call_ID\tSMILES\tScore\n")

        if not os.path.exists(self.tracker_file_path):
            with open(self.tracker_file_path, "w") as f:
                f.write("Calls\tTop10_Avg\tTop10_Molecules\n")

        # Reconstruct 'all_valid_molecules' list for Top-10 tracking
        self.all_valid_molecules: List[Tuple[float, str]] = []
        for smi, score in self.global_cache.items():
            if score > float("-inf"):
                self.all_valid_molecules.append((score, smi))
        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)

        # --- 4. Initialize TDC Oracle ---
        try:
            print(f"[CentralOracle] Initializing TDC Oracle: {self.config.objective_type}")
            self.oracle = Oracle(name=self.config.objective_type)
        except Exception as e:
            print(f"[CentralOracle] Error initializing oracle: {e}")
            self.oracle = None

    def _load_cache(self):
        """Loads state from disk to ensure TASAR continues where RL left off."""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "rb") as f:
                    data = pickle.load(f)
                    self.global_cache = data.get("cache", {})
                    self.unique_calls = data.get("unique_calls", 0)
                    # Restore AUC curve if present, otherwise reset with current state later
                    self.auc_curve_points = data.get("auc_curve_points", [(0, 0.0)])
                print(f"[CentralOracle] State Restored: {self.unique_calls} previous calls loaded.")
            except Exception as e:
                print(f"[CentralOracle] Warning: Failed to load cache: {e}")

    def _save_cache(self):
        """Persists state to disk."""
        try:
            with open(self.cache_file_path, "wb") as f:
                pickle.dump({
                    "cache": self.global_cache,
                    "unique_calls": self.unique_calls,
                    "auc_curve_points": self.auc_curve_points
                }, f)
        except Exception as e:
            print(f"[CentralOracle] Warning: Failed to save cache: {e}")

    def _log_evaluation(self, smiles, score):
        """Logs a single evaluation event."""
        with open(self.history_file_path, "a") as f:
            f.write(f"{self.unique_calls}\t{smiles}\t{score}\n")

    def _update_top10_tracker(self):
        """Updates the summary file used for AUC calculation."""
        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)

        # Keep list manageable
        if len(self.all_valid_molecules) > 200:
            self.all_valid_molecules = self.all_valid_molecules[:200]

        top_10 = self.all_valid_molecules[:10]

        if not top_10:
            avg_score = 0.0
            mol_str = ""
        else:
            avg_score = sum(x[0] for x in top_10) / len(top_10)
            mol_str = ";".join([f"{x[1]}:{x[0]:.4f}" for x in top_10])

        # --- Update AUC Curve State ---
        # Only append if we advanced in calls (avoid duplicate X points)
        if self.unique_calls > self.auc_curve_points[-1][0]:
            self.auc_curve_points.append((self.unique_calls, avg_score))

        with open(self.tracker_file_path, "a") as f:
            f.write(f"{self.unique_calls}\t{avg_score:.6f}\t{mol_str}\n")

    def get_current_pmo_score(self) -> float:
        """
        Calculates the PMO AUC score following the specific benchmark requirement:
        - Sample the Top-10 average at fixed intervals of 100 oracle calls.
        - Uses Trapezoidal Integration on these sampled points.
        - Extrapolates flatly to 10,000 calls.

        This function can be called at ANY time (e.g. step 753) and returns the
        valid projected AUC.
        """
        if not self.auc_curve_points:
            return 0.0

        # 1. Define Sampling Grid (0, 100, 200, ..., N)
        # We create points up to the current unique_calls count
        grid_x = list(range(0, self.unique_calls + 1, 100))

        # 2. Add the EXACT current point if it's not on the grid
        # This ensures we capture the latest state (e.g., 753) for the tail extension
        if grid_x[-1] != self.unique_calls:
            grid_x.append(self.unique_calls)

        # 3. Resample the curve (Step Function Logic)
        # We need to find the value of the curve at each grid point.
        raw_x = [p[0] for p in self.auc_curve_points]
        raw_y = [p[1] for p in self.auc_curve_points]

        # Use searchsorted to find the index of the last update before or at each grid point
        # This implements the "Hold Previous Value" logic required for sparse updates
        indices = np.searchsorted(raw_x, grid_x, side='right') - 1

        grid_y = []
        for idx in indices:
            if idx < 0:
                grid_y.append(0.0)
            else:
                grid_y.append(raw_y[idx])

        # 4. Tail Extension Logic (Current Step -> 10,000)
        # We extend the LAST known score (at self.unique_calls) to the limit
        if grid_x[-1] < self.max_budget:
            grid_x.append(self.max_budget)
            grid_y.append(grid_y[-1])  # Flat line extension
        elif grid_x[-1] > self.max_budget:
            grid_x[-1] = self.max_budget

        # 5. Integration (Trapezoidal Rule on the Grid)
        raw_auc = np.trapz(y=grid_y, x=grid_x)

        # 6. Normalization
        normalized_auc = raw_auc / self.max_budget

        return float(normalized_auc)

    # def get_current_pmo_score(self) -> float:
    #     """
    #     Calculates the PMO AUC score based on the current history.
    #     - Uses Trapezoidal Integration (GenMol standard).
    #     - Extrapolates flatly to 10,000 calls (Tail Extension).
    #     - Normalizes by 10,000.
    #     """
    #     if not self.auc_curve_points:
    #         return 0.0
    #
    #     # Copy points to avoid mutating state during calculation
    #     x = [p[0] for p in self.auc_curve_points]
    #     y = [p[1] for p in self.auc_curve_points]
    #
    #     # 1. Tail Extension logic
    #     last_x = x[-1]
    #     last_y = y[-1]
    #
    #     if last_x < self.max_budget:
    #         # Extend horizontally to 10,000
    #         x.append(self.max_budget)
    #         y.append(last_y)
    #     elif last_x > self.max_budget:
    #         # Clamp to 10,000 if overshot (rare, but safe)
    #         x[-1] = self.max_budget
    #
    #     # 2. Integration (Trapezoidal Rule)
    #     raw_auc = np.trapz(y=y, x=x)
    #
    #     # 3. Normalization
    #     normalized_auc = raw_auc / self.max_budget
    #
    #     return float(normalized_auc)

    def get_budget_status(self):
        return self.unique_calls, self.budget_exceeded

    def predict_batch(self, smiles_list: List[str]) -> List[float]:
        """
        Evaluates a batch of SMILES.
        - Checks Cache (De-duplication).
        - Checks Budget.
        - Logs EVERY new evaluation to history.
        - Logs Top-10 updates IMMEDIATELY when they improve the best-so-far list.
        """
        results = []
        cache_updated = False

        # 1. Determine current threshold to enter Top 10
        # We sort once to be sure we have the correct boundary
        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)

        if len(self.all_valid_molecules) < 10:
            top10_threshold = float("-inf")  # Any valid molecule qualifies
        else:
            top10_threshold = self.all_valid_molecules[9][0]  # The 10th best score

        for smi in smiles_list:
            # --- Validation & Filtering (Unchanged) ---
            if not smi:
                results.append(float("-inf"))
                continue

            try:
                canon_smi = Chem.CanonSmiles(smi)
            except:
                results.append(float("-inf"))
                continue

            if canon_smi == "C":
                results.append(float("-inf"))
                continue

            # --- Cache Hit (Unchanged) ---
            if canon_smi in self.global_cache:
                results.append(self.global_cache[canon_smi])
                continue

            # --- Budget Check (Unchanged) ---
            if self.unique_calls >= self.max_budget:
                self.budget_exceeded = True
                results.append(float("-inf"))
                continue

            # --- Oracle Call (Unchanged) ---
            score = self.oracle(canon_smi)

            # --- Update State (Unchanged) ---
            self.global_cache[canon_smi] = score
            self.unique_calls += 1

            with open(self.history_file_path, "a") as f:
                f.write(f"{self.unique_calls}\t{canon_smi}\t{score}\n")

            results.append(score)
            cache_updated = True

            # --- FINE-GRAINED LOGGING LOGIC ---
            if score > float("-inf"):
                self.all_valid_molecules.append((score, canon_smi))

                # If this new molecule beats the threshold, log IMMEDIATELY.
                if score > top10_threshold:
                    self._update_top10_tracker()  # This sorts and prunes all_valid_molecules

                    # Update local threshold for the next iteration of this loop
                    # (since all_valid_molecules has changed)
                    if len(self.all_valid_molecules) < 10:
                        top10_threshold = float("-inf")
                    else:
                        # The list was sorted inside _update_top10_tracker
                        top10_threshold = self.all_valid_molecules[9][0]

        if cache_updated:
            self._save_cache()

        return results

    # def predict_batch(self, smiles_list: List[str]) -> List[float]:
    #     """
    #     Evaluates a batch of SMILES.
    #     - Checks Cache (De-duplication).
    #     - Checks Budget.
    #     - Logs EVERY new evaluation to history.
    #     - For PMO benchmark: Logs Top-10 updates every 100 unique calls.
    #     """
    #     results = []
    #     cache_updated = False
    #
    #     for smi in smiles_list:
    #         # 1. Validation
    #         if not smi:
    #             results.append(float("-inf"))
    #             continue
    #
    #         try:
    #             canon_smi = Chem.CanonSmiles(smi)
    #         except:
    #             results.append(float("-inf"))
    #             continue
    #
    #         # Filter Trivial
    #         if canon_smi == "C":
    #             results.append(float("-inf"))
    #             continue
    #
    #         # 2. Cache Hit (Sample Efficiency)
    #         if canon_smi in self.global_cache:
    #             results.append(self.global_cache[canon_smi])
    #             continue
    #
    #         # 3. Budget Check
    #         if self.unique_calls >= self.max_budget:
    #             self.budget_exceeded = True
    #             results.append(float("-inf"))
    #             continue
    #
    #         # 4. Oracle Call (The Expensive Part)
    #         score = self.oracle(canon_smi)
    #
    #         # 5. Update State
    #         self.global_cache[canon_smi] = score
    #         self.unique_calls += 1  # Increments on unique calls only
    #
    #         # 6. Log this specific molecule (Raw History)
    #         with open(self.history_file_path, "a") as f:
    #             f.write(f"{self.unique_calls}\t{canon_smi}\t{score}\n")
    #
    #         results.append(score)
    #         cache_updated = True
    #
    #         # Maintain the pool of valid molecules for the Top-10 calculation
    #         if score > float("-inf"):
    #             self.all_valid_molecules.append((score, canon_smi))
    #
    #         # Update the AUC tracker exactly every 100 UNIQUE oracle calls.
    #         # This aligns with the 'freq=100' logic in the benchmark code.
    #         if self.unique_calls % 100 == 0:
    #             self._update_top10_tracker()
    #
    #     # Save cache if any new call was made (persistence)
    #     if cache_updated:
    #         self._save_cache()
    #
    #     return results


# Helper wrapper for local objects to talk to the remote actor easily
class RemoteEvaluatorProxy:
    def __init__(self, actor_handle):
        self.actor = actor_handle

    def predict_objective(self, molecule_designs: List[Union[MoleculeDesign, str]]) -> np.array:
        smiles_input = []
        for mol in molecule_designs:
            if isinstance(mol, MoleculeDesign):
                if mol.synthesis_done and not mol.infeasibility_flag:
                    smiles_input.append(mol.smiles_string)
                else:
                    smiles_input.append("")
            else:
                smiles_input.append(mol)

        # Blocking call to actor
        scores = ray.get(self.actor.predict_batch.remote(smiles_input))

        # Assign back
        for i, mol in enumerate(molecule_designs):
            if isinstance(mol, MoleculeDesign):
                mol.objective = scores[i]

        return np.array(scores)