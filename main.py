import argparse
import copy
import importlib
import os
import time
from typing import List, Optional

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from logger import Logger
from molecule_dataset import RandomMoleculeDataset

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
import ray
import torch
import numpy as np
import wandb
from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu
from molecule_evaluator import MoleculeObjectiveEvaluator
from rl_updates import dr_grpo_update, TrajectoryRecord


def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)

def train_for_one_epoch_supervised(epoch: int,
                                   config: MoleculeConfig,
                                   network: MoleculeTransformer,
                                   network_weights: dict,
                                   optimizer: torch.optim.Optimizer,
                                   objective_evaluator: MoleculeObjectiveEvaluator,
                                   best_objective: float):
    """
    Original supervised fine-tuning path (dataset generation + cross-entropy on heads).
    """
    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )
    metrics = gumbeldore_dataset.generate_dataset(
        network_weights,
        best_objective=best_objective,
        memory_aggressive=False
    )
    print("Generated molecules")
    print(f"Mean obj. over fresh best mols: {metrics['mean_best_gen_obj']:.3f}")
    print(f"Best / worst obj. over fresh best mols: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")
    print(f"Mean obj. over all time top 20 mols: {metrics['mean_top_20_obj']:.3f}")
    print(f"All time best mol: {list(metrics['top_20_molecules'][0].values())[0]:.3f}")
    torch.cuda.empty_cache()
    time.sleep(1)
    print("---- Loading dataset")
    dataset = RandomMoleculeDataset(config,
                                    config.gumbeldore_config["destination_path"],
                                    batch_size=config.batch_size_training,
                                    custom_num_batches=config.num_batches_per_epoch)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=config.num_dataloader_workers,
                            pin_memory=True,
                            persistent_workers=True)

    # Train for one epoch
    network.train()

    # freeze layers except the last (original behavior)

    if config.freeze_all_except_final_layer:
        for parameter in network.parameters():
            parameter.requires_grad = False
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)

        logits_zero, logits_one, logits_two = network(input_data)

        # Apply feasibility masks (True = infeasible)
        logits_zero[input_data["feasibility_mask_level_zero"]] = float("-inf")
        logits_one[input_data["feasibility_mask_level_one"]] = float("-inf")
        logits_two[input_data["feasibility_mask_level_two"]] = float("-inf")

        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
        loss_zero = criterion(logits_zero, target_zero)
        loss_zero = torch.tensor(0.) if torch.isnan(loss_zero) else loss_zero
        loss_one = criterion(logits_one, target_one)
        loss_one = torch.tensor(0.) if torch.isnan(loss_one) else loss_one
        loss_two = criterion(logits_two, target_two)
        loss_two = torch.tensor(0.) if torch.isnan(loss_two) else loss_two
        loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()

        batch_loss = loss.item()
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / num_batches
    metrics["loss_level_one"] = accumulated_loss_lvl_one / num_batches
    metrics["loss_level_two"] = accumulated_loss_lvl_two / num_batches

    top_20_molecules = metrics["top_20_molecules"]
    del metrics["top_20_molecules"]
    return metrics, top_20_molecules

def train_for_one_epoch_rl(epoch: int,
                           config: MoleculeConfig,
                           network: MoleculeTransformer,
                           network_weights: dict,
                           optimizer: torch.optim.Optimizer,
                           objective_evaluator: MoleculeObjectiveEvaluator,
                            gumbeldore_dataset: GumbeldoreDataset,
                           novelty_memory: Optional[dict] = None):
    """
    RL fine-tuning epoch:
      1. Generate trajectories (terminated molecules) with current policy.
      2. Run policy gradient update via dr_grpo_update.
      3. Produce logging artifacts similar in spirit to supervised path.
    """
    print(f"[RL] Generating trajectories (epoch {epoch + 1})...")
    # gumbeldore_dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_evaluator)

    if config.prodrug_mode:
        # Use training set
        current_prompts = config.prodrug_parents_train
    else:
        current_prompts = None  # Let generate_dataset use defaults

    # Return raw terminated trajectories (list of MoleculeDesign)
    trajectories = gumbeldore_dataset.generate_dataset(
        network_weights=network_weights,
        best_objective=None,
        memory_aggressive=False,
        prompts=current_prompts
    )

    if not trajectories or not any(trajectories):
        print("[RL] WARNING: No trajectories generated this epoch. Skipping update.")
        return {
            "num_trajectories": 0,
            "policy_loss": 0.0,
            "baseline": 0.0,
            "mean_reward": float("-inf"),
            "best_reward": float("-inf"),
            "mean_advantage": 0.0,
            "std_advantage": 0.0,
            "fraction_pos_adv": 0.0
        }, ["No molecules"]

    # Freeze backbone (match supervised style)
    if config.freeze_all_except_final_layer:
        for p in network.parameters():
            p.requires_grad = False
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    network.train()

    print("training ...")
    metrics = dr_grpo_update(
        model=network,
        optimizer=optimizer,
        designs_groups=trajectories,
        config=config,
        device=torch.device(config.training_device),
        logger=None,
        novelty_memory=novelty_memory
    )
    print("dr GRPO update done.")
    metrics["best_gen_obj"] = metrics.get("best_objective", float("-inf"))
    metrics["mean_best_gen_obj"] = metrics.get("mean_reward", float("-inf"))
    metrics.setdefault("loss_level_zero", 0.0)
    metrics.setdefault("loss_level_one", 0.0)
    metrics.setdefault("loss_level_two", 0.0)

    # Build top 20 text artifact
    mol_map = {}
    for group in trajectories:
        for m in group:
            if m.objective is None:
                continue
            if not m.smiles_string:  # Good to add this check
                continue
            if m.smiles_string not in mol_map or mol_map[m.smiles_string]["obj"] < m.objective:
                mol_map[m.smiles_string] = {
                    "smiles": m.smiles_string,
                    "obj": m.objective
                }
    unique_mols = list(mol_map.values())
    unique_mols.sort(key=lambda x: x["obj"], reverse=True)
    top20 = unique_mols[:20]

    if top20:
        mean_top20_obj = sum(entry["obj"] for entry in top20) / len(top20)
        metrics["mean_top_20_obj"] = mean_top20_obj
    else:
        metrics["mean_top_20_obj"] = float("-inf")

    top_20_text_lines = []
    for i, entry in enumerate(top20):
        top_20_text_lines.append(f"{i + 1:02d}: {entry['smiles']}  obj={entry['obj']:.4f}")
    if not top20:
        top_20_text_lines.append("No terminated molecules")

    return metrics, top_20_text_lines

def evaluate(eval_type: str, config: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Uses generation (supervised-style metrics) for evaluation irrespective of RL training.
    If a fragment library is used (Scaffold Decoration), it attempts to load
    the corresponding 'test_scaffolds.txt' to perform zero-shot evaluation.
    Processes large test sets in batches to avoid OOM.
    """
    config = copy.deepcopy(config)
    config.gumbeldore_config["destination_path"] = None

    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )

    test_prompts = None
    use_batched_eval = False
    EVAL_BATCH_SIZE = 500  # Process scaffolds in batches of 500

    # Check if a specific TEST file is defined in config
    if getattr(config, 'evaluation_scaffolds_path', None):
        path = config.evaluation_scaffolds_path
        if os.path.exists(path):
            print(f"[Eval] Loading Test Scaffolds from: {path}")
            with open(path, 'r') as f:
                test_prompts = [line.strip() for line in f if line.strip()]

            # Use only 1% of scaffolds to avoid memory issues
            original_count = len(test_prompts)
            sample_size = max(1, int(original_count * 0.01))  # 1%
            test_prompts = test_prompts[:sample_size]
            print(f"[Eval] Using {sample_size}/{original_count} scaffolds (0.1% sample)")

            # Optional: Subset for speed during training checks
            if eval_type != 'test':
                test_prompts = test_prompts[:32]
                print(f"[Eval] Non-test mode: further limited to {len(test_prompts)} scaffolds")

            # Enable batched evaluation for large test sets
            if len(test_prompts) > EVAL_BATCH_SIZE:
                use_batched_eval = True
                print(f"[Eval] Large test set ({len(test_prompts)} scaffolds). Using batched evaluation.")
        else:
            print(f"[Eval] WARNING: Config path {path} not found.")

    # If no path, check Prodrug mode
    elif config.prodrug_mode:
        test_prompts = config.prodrug_parents_test
        print(f"[Eval] Using {len(test_prompts)} prodrug parent scaffolds for evaluation")

    # If no path, check Prodrug mode
    elif config.prodrug_mode:
        test_prompts = config.prodrug_parents_test

    # Parameters for "Success"
    SUCCESS_THRESHOLD = 0.5

    scaffold_metrics = []
    all_valid_mols = []

    if use_batched_eval and test_prompts is not None:
        # --- BATCHED EVALUATION ---
        num_batches = (len(test_prompts) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
        print(f"[Eval] Processing {len(test_prompts)} scaffolds in {num_batches} batches...")

        for batch_idx in tqdm(range(num_batches), desc="Eval Batches"):
            start_idx = batch_idx * EVAL_BATCH_SIZE
            end_idx = min(start_idx + EVAL_BATCH_SIZE, len(test_prompts))
            batch_prompts = test_prompts[start_idx:end_idx]

            # Generate for this batch
            grouped_trajectories = gumbeldore_dataset.generate_dataset(
                copy.deepcopy(network.get_weights()),
                memory_aggressive=False,
                prompts=batch_prompts,
                return_raw_trajectories=True
            )

            # Process batch results
            for group in grouped_trajectories:
                if not group:
                    continue
                objs = [m.objective for m in group if m.objective is not None]
                valid_mols = [m for m in group if m.objective is not None]
                all_valid_mols.extend(valid_mols)

                if not objs:
                    scaffold_metrics.append({"solved": 0.0, "top1": 0.0, "mean": 0.0})
                    continue

                best_score = max(objs)
                mean_score = np.mean(objs)
                is_solved = 1.0 if best_score > SUCCESS_THRESHOLD else 0.0
                scaffold_metrics.append({"solved": is_solved, "top1": best_score, "mean": mean_score})

            # Clear memory between batches
            del grouped_trajectories
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    else:
        # --- ORIGINAL NON-BATCHED PATH ---
        grouped_trajectories = gumbeldore_dataset.generate_dataset(
            copy.deepcopy(network.get_weights()),
            memory_aggressive=False,
            prompts=test_prompts,
            return_raw_trajectories=True
        )

        for group in grouped_trajectories:
            if not group:
                continue
            objs = [m.objective for m in group if m.objective is not None]
            valid_mols = [m for m in group if m.objective is not None]
            all_valid_mols.extend(valid_mols)

            if not objs:
                scaffold_metrics.append({"solved": 0.0, "top1": 0.0, "mean": 0.0})
                continue

            best_score = max(objs)
            mean_score = np.mean(objs)
            is_solved = 1.0 if best_score > SUCCESS_THRESHOLD else 0.0
            scaffold_metrics.append({"solved": is_solved, "top1": best_score, "mean": mean_score})

    # --- AGGREGATION ---
    if not scaffold_metrics:
        print("[Eval] Warning: No valid molecules generated.")
        return {}, []

    avg_success_rate = np.mean([m["solved"] for m in scaffold_metrics])
    avg_top1_score = np.mean([m["top1"] for m in scaffold_metrics])
    avg_mean_score = np.mean([m["mean"] for m in scaffold_metrics])

    metrics_out = {
        f"{eval_type}_success_rate": avg_success_rate,
        f"{eval_type}_mean_top1_obj": avg_top1_score,
        f"{eval_type}_global_mean_obj": avg_mean_score,
        f"{eval_type}_num_scaffolds_evaluated": len(scaffold_metrics)
    }

    print("=" * 30)
    print(f"EVALUATION REPORT ({eval_type})")
    print(f"Scaffolds Processed: {len(scaffold_metrics)}")
    print(f"Success Rate (> {SUCCESS_THRESHOLD}): {avg_success_rate * 100:.2f}%")
    print(f"Mean Top-1 Score: {avg_top1_score:.4f}")
    print("=" * 30)

    all_valid_mols.sort(key=lambda x: x.objective, reverse=True)
    top_20_objects = all_valid_mols[:20]

    top_20_text_lines = []
    for i, m in enumerate(top_20_objects):
        smi = m.smiles_string if m.smiles_string else "Invalid"
        top_20_text_lines.append(f"{i + 1:02d}: {smi}  obj={m.objective:.4f}")

    return metrics_out, top_20_text_lines




if __name__ == '__main__':
    print(">> Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")

    # We add arguments for the parameters defined in sweep.yaml
    # Use defaults from base config if the script is run without `wandb agent`
    temp_config_for_defaults = MoleculeConfig()  # Load defaults once
    parser.add_argument('--learning_rate', type=float,
                        default=temp_config_for_defaults.optimizer["lr"],
                        help='Optimizer learning rate')
    parser.add_argument('--rl_entropy_beta', type=float,
                        default=temp_config_for_defaults.rl_entropy_beta,
                        help='Entropy bonus coefficient for RL')
    parser.add_argument('--ppo_epochs', type=int,
                        default=temp_config_for_defaults.ppo_epochs,
                        help='Number of PPO epochs per RL update')
    parser.add_argument('--rl_ppo_clip_epsilon', type=float,
                        default=temp_config_for_defaults.rl_ppo_clip_epsilon,
                        help='GRPO clipping epsilon')
    del temp_config_for_defaults  # Clean up temporary config

    args = parser.parse_args()

    if args.config is not None:
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig
    config = MoleculeConfig()

    config.optimizer["lr"] = args.learning_rate
    config.rl_entropy_beta = args.rl_entropy_beta
    config.ppo_epochs = args.ppo_epochs
    config.rl_ppo_clip_epsilon = args.rl_ppo_clip_epsilon

    # --- WANDB INITIALIZATION ---
    if hasattr(config, 'use_wandb') and config.use_wandb:
        # Convert the config object to a dictionary for wandb
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        # For nested dictionaries like 'optimizer', wandb prefers a flat structure
        flat_config = {}
        for k, v in config_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_config[f"{k}.{sub_k}"] = sub_v
            else:
                flat_config[k] = v

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=flat_config
        )

        config.optimizer["lr"] = wandb.config.get('optimizer.lr', config.optimizer["lr"]) # Example
        config.rl_entropy_beta = wandb.config.get('rl_entropy_beta', config.rl_entropy_beta)
        config.ppo_epochs = wandb.config.get('ppo_epochs', config.ppo_epochs)
        config.rl_ppo_clip_epsilon = wandb.config.get('rl_ppo_clip_epsilon', config.rl_ppo_clip_epsilon)

        wandb.config.update({"task": config.objective_type}, allow_val_change=True)
        # wandb.config.update({"task": config.objective_type})  # Log the task separately for easy filtering

    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))

    if ray.is_initialized():
        ray.shutdown()  # In case ray was already running and messing things up

    ray.init(num_gpus=num_gpus, logging_level="info", ignore_reinit_error=True)
    print(ray.available_resources())

    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Top-K (all-time) SMILES archive (purely observational)
    # Lines format: <objective>\t<SMILES>
    # ------------------------------------------------------------------
    TOP_K_OBS = 10
    topk_archive_path = os.path.join(config.results_path, "top10_all_time_smiles.txt")
    topk_smiles_scores = {}  # {smiles: best_objective}


    def update_topk_archive_from_epoch(epoch_top20, rl_mode: bool):
        """
        epoch_top20:
          - RL mode: list[str] lines like '01: SMILES  obj=0.1234'
          - Supervised mode: list with a single dict {smiles: obj, ...}
        """
        if rl_mode:
            for line in epoch_top20:
                if "obj=" not in line:
                    continue
                try:
                    after_rank = line.split(":", 1)[1].strip()
                    tokens = after_rank.split()
                    if not tokens:
                        continue
                    obj_token = None
                    for t in reversed(tokens):
                        if t.startswith("obj="):
                            obj_token = t
                            break
                    if obj_token is None:
                        continue
                    obj_val = float(obj_token.split("obj=")[1])
                    obj_index = tokens.index(obj_token)
                    smiles = " ".join(tokens[:obj_index]).strip()
                    if not smiles:
                        continue
                    prev = topk_smiles_scores.get(smiles)
                    if (prev is None) or (obj_val > prev):
                        topk_smiles_scores[smiles] = obj_val
                except Exception:
                    continue
        else:
            if not epoch_top20:
                return
            d = epoch_top20[0]
            for smiles, obj_val in d.items():
                prev = topk_smiles_scores.get(smiles)
                if (prev is None) or (obj_val > prev):
                    topk_smiles_scores[smiles] = obj_val

        # Truncate to top K by objective
        if len(topk_smiles_scores) > TOP_K_OBS:
            sorted_items = sorted(topk_smiles_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_OBS]
            topk_smiles_scores.clear()
            topk_smiles_scores.update(sorted_items)

        # Persist: objective<TAB>SMILES (descending objective)
        os.makedirs(config.results_path, exist_ok=True)
        with open(topk_archive_path, "w") as f:
            for smiles, score in sorted(topk_smiles_scores.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{score:.6f}\t{smiles}\n")


    # Policy network
    network = MoleculeTransformer(config, config.training_device)
    objective_eval = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)

    # Load checkpoint if needed
    if config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path}")
        checkpoint = torch.load(config.load_checkpoint_from_path)
        print(f"{checkpoint['epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "epochs_trained": 0,
            "validation_metric": float("-inf"),
            "best_validation_metric": float("-inf")
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()

    if config.num_epochs > 0:
        print(f"Starting training for {config.num_epochs} epochs.")

        best_model_weights = checkpoint["best_model_weights"]
        best_validation_metric = checkpoint["best_validation_metric"]

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )
        if checkpoint["optimizer_state"] is not None and config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
        print("Setting up LR scheduler")
        _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (
                checkpoint["epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        start_time_counter = None
        if config.wall_clock_limit is not None:
            print(f"Wall clock limit of training set to {config.wall_clock_limit / 3600} hours")
            start_time_counter = time.perf_counter()

        rl_mode_active = getattr(config, "use_dr_grpo", False)

        if getattr(config, "rl_use_novelty_bonus") and rl_mode_active:
            print("Novelty bonus enabled.")
            novelty_memory = {}
        else:
            novelty_memory = None

        gumbeldore_dset = GumbeldoreDataset(config=config, objective_evaluator=objective_eval)

        for epoch in range(config.num_epochs):
            print("------")
            network_weights = copy.deepcopy(network.get_weights())

            if novelty_memory is not None:
                print(f"Start of Epoch {epoch + 1}: Novelty memory contains {len(novelty_memory)} unique SMILES.")

            if rl_mode_active:
                generated_loggable_dict, top20_text = train_for_one_epoch_rl(
                    epoch, config, network, network_weights, optimizer, objective_eval, gumbeldore_dset,
                    novelty_memory=novelty_memory
                )
                # The last return value (the buffer data) is not needed in the main loop, so we use _
                val_metric = generated_loggable_dict.get("best_gen_obj", float("-inf"))

            else:  # Original Supervised-only mode
                generated_loggable_dict, top20_text = train_for_one_epoch_supervised(
                    epoch, config, network, network_weights, optimizer, objective_eval, best_validation_metric
                )
                val_metric = generated_loggable_dict["best_gen_obj"]

            # Update all-time top-K SMILES archive
            try:
                if rl_mode_active:  # top20_text is a list of strings
                    update_topk_archive_from_epoch(top20_text, rl_mode=True)
                else:  # top20_text is a list containing one dictionary
                    update_topk_archive_from_epoch(top20_text, rl_mode=False)
            except Exception as e:
                print(f"[TopK Archive] Warning: failed to update archive this epoch: {e}")

            checkpoint["epochs_trained"] += 1
            scheduler.step()

            print(f">> Epoch {checkpoint['epochs_trained']}. "
                  f"Best (gen/rl) objective: {val_metric:.4f}")
            if rl_mode_active:
                mean_r = generated_loggable_dict.get('mean_reward', float('nan'))
                policy_l = generated_loggable_dict.get('policy_loss', float('nan'))
                print(f"   RL Stats: Mean Reward={mean_r:.4f}, Policy Loss={policy_l:.6f}")

            logger.log_metrics(generated_loggable_dict, step=epoch)

            # First, update the overall best metric
            if val_metric > best_validation_metric:
                print(">> Got new best model.")
                best_validation_metric = val_metric
                checkpoint["best_model_weights"] = copy.deepcopy(network.get_weights())
                checkpoint["best_validation_metric"] = best_validation_metric
                save_checkpoint(checkpoint, "best_model.pt", config)

            # WandB logging
            if hasattr(config, 'use_wandb') and config.use_wandb:
                wandb_log = {
                    "epoch": checkpoint["epochs_trained"],
                    "best_all_time_obj": best_validation_metric,
                    "best_current_obj": val_metric,
                    'mean_current_obj': generated_loggable_dict.get('mean_best_gen_obj'),
                    'worst_current_obj': generated_loggable_dict.get('worst_gen_obj'),
                    "mean_top_20_all_time_obj": generated_loggable_dict.get("mean_top_20_obj"),
                    "learning_rate": scheduler.get_last_lr()[0]
                }
                # Add specific RL metrics if in RL mode
                if rl_mode_active:
                    # Define the specific list of metrics you want to log from the RL update
                    keys_to_log = [
                        'baseline',
                        'mean_reward',
                        'best_reward',
                        # 'best_objective',
                        'mean_advantage',
                        'std_advantage',
                        'policy_loss',
                        'mean_entropy',
                        'mean_traj_length',
                        'num_trajectories',
                        'mean_top_20_obj'
                        # 'mean_novelty_bonus'
                    ]
                    # Add only the specified metrics to the log
                    for key in keys_to_log:
                        if key in generated_loggable_dict:
                            wandb_log[key] = generated_loggable_dict[key]

                    # This captures 'prodrug/mean_logp_delta', 'prodrug/fraction_cleavable', etc.
                    for key, val in generated_loggable_dict.items():
                        if key.startswith("prodrug/"):
                            wandb_log[key] = val

                wandb.log(wandb_log)

            if rl_mode_active:
                logger.text_artifact(os.path.join(config.results_path, f"epoch_{epoch + 1}_train_top_20_molecules.txt"),
                                     "\n".join(top20_text))
            else:
                logger.text_artifact(os.path.join(config.results_path, f"epoch_{epoch + 1}_train_top_20_molecules.txt"),
                                     top20_text)

            # Update and save the 'last' model checkpoint
            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))
            checkpoint["validation_metric"] = val_metric
            save_checkpoint(checkpoint, "last_model.pt", config)

            if start_time_counter is not None and time.perf_counter() - start_time_counter > config.wall_clock_limit:
                print("Time exceeded. Stopping training.")
                break

    if config.num_epochs == 0:
        print(f"Testing with loaded model.")
    else:
        print(f"Testing with best model.")
        best_ckpt_path = os.path.join(config.results_path, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path)
            network.load_state_dict(checkpoint["model_weights"])
        else:
            print("WARNING: best_model.pt not found; using last model.")

    if checkpoint["model_weights"] is None and config.num_epochs == 0:
        print("WARNING! No training performed and no checkpoint loaded. Evaluating random model.")

    torch.cuda.empty_cache()
    with torch.no_grad():
        test_loggable_dict, test_text_to_save = evaluate('test', config, network, objective_eval)
    print(">> TEST")
    print(test_loggable_dict)
    logger.log_metrics(test_loggable_dict, step=0, step_desc="test")
    print(test_text_to_save)
    logger.text_artifact(os.path.join(config.results_path, "test_top_20_molecules.txt"),
                         test_text_to_save)

    # WanB finish
    if hasattr(config, 'use_wandb') and config.use_wandb:
        wandb.finish()

    print("Finished. Shutting down ray.")
    ray.shutdown()