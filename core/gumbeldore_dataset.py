import copy
import os
import pickle
import sys
import time
from typing import List, Callable, Tuple, Any, Type, Optional

import numpy as np
import ray
import torch
from rdkit import RDLogger
from ray.thirdparty_files import psutil
from tqdm import tqdm

from config import MoleculeConfig
from core.abstracts import Config, Instance, BaseTrajectory
from core.incremental_sbs import IncrementalSBS
import core.stochastic_beam_search as sbs

from model.molecule_transformer import MoleculeTransformer
from molecule_design import MoleculeDesign
from molecule_evaluator import MoleculeObjectiveEvaluator

os.environ["RAY_DEDUP_LOGS"] = "0"


@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results


class GumbeldoreDataset:
    """
    Generates molecules via (T)ASAR / (stochastic) beam search.
    """
    def __init__(self, config: MoleculeConfig,
                 objective_evaluator: MoleculeObjectiveEvaluator):
        self.config = config
        self.gumbeldore_config = config.gumbeldore_config
        self.objective_evaluator = objective_evaluator
        self.devices_for_workers: List[str] = self.gumbeldore_config["devices_for_workers"]

    def generate_dataset(self,
                         network_weights: dict,
                         best_objective: Optional[float] = None,
                         memory_aggressive: bool = False,
                         return_raw: bool = False):
        batch_size_gpu, batch_size_cpu = (self.gumbeldore_config["batch_size_per_worker"],
                                          self.gumbeldore_config["batch_size_per_cpu_worker"])

        if self.config.start_from_c_chains:
            problem_instances = MoleculeDesign.get_c_chains(self.config)
        elif self.config.start_from_smiles is not None:
            problem_instances = [MoleculeDesign.from_smiles(self.config, self.config.start_from_smiles)]
        else:
            problem_instances = MoleculeDesign.get_single_atom_molecules(
                self.config, repeat=self.config.repeat_start_instances
            )

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)

        cpu_cores = [None] * len(self.devices_for_workers)
        if self.gumbeldore_config["pin_workers_to_core"] and sys.platform == "linux":
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        future_tasks = [
            async_sbs_worker.remote(
                self.config, job_pool, network_weights, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                cpu_cores[i], best_objective, memory_aggressive
            )
            for i, device in enumerate(self.devices_for_workers)
        ]

        with tqdm(total=len(problem_instances)) as progress_bar:
            while True:
                do_break = len(ray.wait(future_tasks, num_returns=len(future_tasks), timeout=0.5)[1]) == 0
                fetched_results = ray.get(job_pool.fetch_results.remote())
                for (i, result) in fetched_results:
                    results[i] = result
                if fetched_results:
                    progress_bar.update(len(fetched_results))
                if do_break:
                    break

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        if return_raw:
            flat: List[MoleculeDesign] = []
            for lst in results:
                if lst is None:
                    continue
                flat.extend(lst)
            return flat

        return self.process_results(problem_instances, results)

    def process_results(self, problem_instances, results):
        metrics_return = dict()
        instances_dict = dict()

        for i, _ in enumerate(problem_instances):
            mol_list = results[i]
            if not mol_list:
                continue
            for molecule in mol_list:  # type: MoleculeDesign
                if molecule.objective is not None and molecule.objective > float("-inf"):
                    instances_dict[molecule.smiles_string] = dict(
                        start_atom=molecule.initial_atom,
                        action_seq=molecule.history,
                        smiles=molecule.smiles_string,
                        obj=molecule.objective,
                        sa_score=molecule.sa_score
                    )

        generated_mols = list(instances_dict.values())
        generated_mols = sorted(
            generated_mols,
            key=lambda x: x["obj"],
            reverse=True
        )[:self.gumbeldore_config["num_trajectories_to_keep"]]

        if len(generated_mols) == 0:
            return {
                "mean_best_gen_obj": float("-inf"),
                "mean_best_gen_sa_score": float("-inf"),
                "best_gen_obj": float("-inf"),
                "best_gen_sa_score": float("-inf"),
                "worst_gen_obj": float("-inf"),
                "worst_gen_sa_score": float("-inf"),
                "mean_top_20_obj": float("-inf"),
                "mean_kept_obj": float("-inf"),
                "mean_top_20_sa_score": float("-inf"),
                "top_20_molecules": []
            }

        generated_objs = np.array([x["obj"] for x in generated_mols])
        generated_sa_scores = np.array([x["sa_score"] for x in generated_mols])
        metrics_return["mean_best_gen_obj"] = generated_objs.mean()
        metrics_return["mean_best_gen_sa_score"] = generated_sa_scores.mean()
        metrics_return["best_gen_obj"] = generated_objs[0]
        metrics_return["best_gen_sa_score"] = generated_sa_scores[0]
        metrics_return["worst_gen_obj"] = generated_objs[-1]
        metrics_return["worst_gen_sa_score"] = generated_sa_scores[-1]

        destination_path = self.gumbeldore_config["destination_path"]
        merged_mols = generated_mols
        if destination_path is not None:
            if os.path.isfile(destination_path):
                with open(destination_path, "rb") as f:
                    existing_mols = pickle.load(f)
                temp_d = {x["smiles"]: x for x in existing_mols + merged_mols}
                merged_mols = list(temp_d.values())
                merged_mols = sorted(
                    merged_mols,
                    key=lambda x: x["obj"],
                    reverse=True
                )[:self.gumbeldore_config["num_trajectories_to_keep"]]

            with open(destination_path, "wb") as f:
                pickle.dump(merged_mols, f)

        metrics_return["mean_top_20_obj"] = np.array([x["obj"] for x in merged_mols[:20]]).mean()
        metrics_return["mean_kept_obj"] = np.array([x["obj"] for x in merged_mols]).mean()
        metrics_return["mean_top_20_sa_score"] = np.array([x["sa_score"] for x in merged_mols[:20]]).mean()
        metrics_return["top_20_molecules"] = [{x["smiles"]: x["obj"] for x in merged_mols[:20]}]

        return metrics_return


@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False):
    """
    Worker performing (T)ASAR / stochastic beam search.
    """

    def child_log_probability_fn(trajectories: List[MoleculeDesign]) -> [np.array]:
        return MoleculeDesign.log_probability_fn(trajectories=trajectories, network=network)

    def batch_leaf_evaluation_fn(trajectories: List[MoleculeDesign]) -> np.array:
        objs = objective_evaluator.predict_objective(trajectories)
        for i, obj in enumerate(objs):
            trajectories[i].objective = obj
        return objs

    def child_transition_fn(trajectory_action_pairs: List[Tuple[MoleculeDesign, int]]):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]

    # NEW: Stratified sampling helper (inside worker for simplicity)
    def stratified_sample(designs: List[MoleculeDesign],
                          max_cap: int,
                          low_q: float,
                          high_q: float,
                          fracs: Tuple[float, float, float],
                          allow_fill: bool = True,
                          rng: Optional[np.random.Generator] = None) -> List[MoleculeDesign]:
        if rng is None:
            rng = np.random.default_rng()
        n = len(designs)
        if n <= max_cap:
            return designs
        rewards = np.array([d.objective if (d.objective is not None and np.isfinite(d.objective))
                            else -1e9 for d in designs], dtype=float)
        low_thr = np.quantile(rewards, low_q)
        high_thr = np.quantile(rewards, high_q)

        bottom_idx = np.where(rewards <= low_thr)[0]
        top_idx = np.where(rewards >= high_thr)[0]
        bottom_set = set(bottom_idx.tolist())
        top_set = set(top_idx.tolist())
        middle_idx = [i for i in range(n) if i not in bottom_set and i not in top_set]

        # Collapse: if thresholds identical / no separation, treat all middle
        if len(top_idx) == 0 and len(bottom_idx) == 0:
            middle_idx = list(range(n))

        top_frac, mid_frac, bot_frac = fracs
        s = top_frac + mid_frac + bot_frac
        if abs(s - 1.0) > 1e-6:
            top_frac /= s
            mid_frac /= s
            bot_frac /= s
        t_top = int(round(max_cap * top_frac))
        t_mid = int(round(max_cap * mid_frac))
        t_bot = max_cap - t_top - t_mid

        def sample_bucket(indices, k):
            if len(indices) <= k:
                return list(indices)
            return rng.choice(indices, size=k, replace=False).tolist()

        sampled_top = sample_bucket(top_idx, t_top)
        sampled_mid = sample_bucket(middle_idx, t_mid)
        sampled_bot = sample_bucket(bottom_idx, t_bot)

        if allow_fill:
            assigned = len(sampled_top) + len(sampled_mid) + len(sampled_bot)
            if assigned < max_cap:
                deficit = max_cap - assigned
                taken = set(sampled_top + sampled_mid + sampled_bot)
                remaining = [i for i in range(n) if i not in taken]
                if remaining:
                    extra = rng.choice(remaining, size=min(deficit, len(remaining)), replace=False).tolist()
                    sampled_mid.extend(extra)

        final_indices = sampled_top + sampled_mid + sampled_bot
        if len(final_indices) > max_cap:
            final_indices = final_indices[:max_cap]

        sampled = [designs[i] for i in final_indices]

        # (optional logging) Uncomment if you want per-root stats
        # print(f"[StratifiedSampling] total={n} kept={len(sampled)} "
        #       f"top={len(sampled_top)} mid={len(sampled_mid)} bottom={len(sampled_bot)} "
        #       f"low_thr={low_thr:.4f} high_thr={high_thr:.4f}")

        return sampled

    # Silence RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():
        if config.CUDA_VISIBLE_DEVICES:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        device_t = torch.device(device)
        network = MoleculeTransformer(config, device_t)
        network.load_state_dict(network_weights)
        network.to(network.device)
        network.eval()

        objective_evaluator = MoleculeObjectiveEvaluator(config, torch.device(config.objective_gnn_device))
        rl_mode = getattr(config, "use_dr_grpo", False)

        # Config-driven sampling mode (RL only)
        sampling_mode = config.gumbeldore_config.get("leaf_sampling_mode", "random")
        strat_q_low, strat_q_high = config.gumbeldore_config.get("stratified_quantiles", (0.10, 0.90))
        strat_fracs = config.gumbeldore_config.get("stratified_target_fracs", (0.25, 0.50, 0.25))
        strat_allow_fill = config.gumbeldore_config.get("stratified_allow_shortfall_fill", True)

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))
            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = [instance for _, instance in batch]

            if config.gumbeldore_config["search_type"] == "beam_search":
                beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=root_nodes,
                    beam_width=config.gumbeldore_config["beam_width"],
                    deterministic=True,
                    top_p=config.gumbeldore_config.get("nucleus_top_p", 1.0)
                )
            else:
                inc_sbs = IncrementalSBS(
                    root_nodes,
                    child_log_probability_fn,
                    child_transition_fn,
                    leaf_evaluation_fn=MoleculeDesign.to_max_evaluation_fn,
                    batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                    memory_aggressive=memory_aggressive
                )

                if config.gumbeldore_config["search_type"] == "tasar":
                    beam_leaves_batch = inc_sbs.perform_tasar(
                        beam_width=config.gumbeldore_config["beam_width"],
                        deterministic=config.gumbeldore_config["deterministic"],
                        nucleus_top_p=config.gumbeldore_config["nucleus_top_p"],
                        replan_steps=config.gumbeldore_config["replan_steps"],
                        sbs_keep_intermediate=config.gumbeldore_config["keep_intermediate_trajectories"]
                    )
                elif config.gumbeldore_config["search_type"] == "wor":
                    beam_leaves_batch = inc_sbs.perform_incremental_sbs(
                        beam_width=config.gumbeldore_config["beam_width"],
                        num_rounds=config.gumbeldore_config["num_rounds"],
                        nucleus_top_p=config.gumbeldore_config["nucleus_top_p"],
                        sbs_keep_intermediate=config.gumbeldore_config["keep_intermediate_trajectories"],
                        best_objective=best_objective
                    )
                else:
                    raise ValueError(f"Unknown search_type: {config.gumbeldore_config['search_type']}")

            results_to_push = []
            for j, result_idx in enumerate(idx_list):
                leaves = beam_leaves_batch[j]

                if rl_mode:
                    terminated_designs: List[MoleculeDesign] = []
                    for leaf in leaves:
                        d: MoleculeDesign = leaf.state
                        if d.history and d.history[-1] == 0 and d.synthesis_done:
                            terminated_designs.append(d)

                    max_cap = config.gumbeldore_config.get("max_leaves_per_root", 0)
                    if max_cap and max_cap > 0 and len(terminated_designs) > max_cap:
                        if sampling_mode == "stratified":
                            # Evaluate all before stratifying
                            unevaluated = [d for d in terminated_designs if d.objective is None]
                            if unevaluated:
                                batch_leaf_evaluation_fn(unevaluated)

                            terminated_designs = stratified_sample(
                                designs=terminated_designs,
                                max_cap=max_cap,
                                low_q=strat_q_low,
                                high_q=strat_q_high,
                                fracs=strat_fracs,
                                allow_fill=strat_allow_fill
                            )

                        elif sampling_mode == "topk":
                            # Evaluate all, then keep top max_cap
                            unevaluated = [d for d in terminated_designs if d.objective is None]
                            if unevaluated:
                                batch_leaf_evaluation_fn(unevaluated)
                            terminated_designs.sort(key=lambda x: x.objective if x.objective is not None else float("-inf"),
                                                    reverse=True)
                            terminated_designs = terminated_designs[:max_cap]

                        elif sampling_mode == "random":  # randomly sample
                            # Uniform subset first, then evaluate only sampled
                            idx = np.random.choice(len(terminated_designs), max_cap, replace=False)
                            terminated_designs = [terminated_designs[i] for i in idx]

                    # Evaluate any remaining unevaluated (covers random path or small sets)
                    unevaluated_final = [d for d in terminated_designs if d.objective is None]
                    if unevaluated_final:
                        batch_leaf_evaluation_fn(unevaluated_final)

                    results_to_push.append((result_idx, terminated_designs))
                else:
                    top_k = config.gumbeldore_config["num_trajectories_to_keep"]
                    truncated = leaves[:top_k]
                    designs = [leaf.state for leaf in truncated]
                    if designs and designs[0].objective is None:
                        batch_leaf_evaluation_fn(designs)
                    results_to_push.append((result_idx, designs))

            ray.get(job_pool.push_results.remote(results_to_push))

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()