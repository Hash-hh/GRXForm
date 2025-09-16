import argparse
import copy
import importlib
import os
import time
from typing import List

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


# ---------------- Supervised (original) epoch ---------------- #
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


def train_for_one_epoch_hybrid(epoch: int,
                               config: MoleculeConfig,
                               network: MoleculeTransformer,
                               network_weights: dict,
                               optimizer: torch.optim.Optimizer,
                               objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Performs one hybrid epoch of RL + IL, using a pickle file for the elite buffer.
    """
    # ====== 1. GENERATION & RL UPDATE STEP (Explore) ======
    print(f"[Hybrid Epoch {epoch + 1}] Starting RL Exploration Step...")
    gumbeldore_dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_evaluator)

    generated_trajectories = gumbeldore_dataset.generate_dataset(
        network_weights=network_weights, best_objective=None, memory_aggressive=False
    )

    if not generated_trajectories:
        print("[RL] WARNING: No trajectories generated. Skipping RL and IL updates.")
        return {"skipped": True}, ["No molecules this epoch"], []

    network.train()
    if config.freeze_all_except_final_layer:
        # Set trainable layers for RL update
        for p in network.parameters(): p.requires_grad = False
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    # Remember to return the records from dr_grpo_update
    rl_metrics, rl_records = dr_grpo_update(
        model=network, optimizer=optimizer, designs=generated_trajectories,
        config=config, device=torch.device(config.training_device), logger=None
    )
    print("[RL] DR-GRPO update complete.")

    # ====== 2. UPDATE ELITE BUFFER (on disk) ======
    print(f"[Hybrid Epoch {epoch + 1}] Updating elite buffer pickle...")
    updated_elite_records_as_dicts = update_elite_buffer_pickle(rl_records, config)

    # ====== 3. IMITATION LEARNING UPDATE STEP (Distill) ======
    il_metrics = {}
    if getattr(config, "rl_use_il_distillation", False):
        print(f"[Hybrid Epoch {epoch + 1}] Starting IL Distillation Step...")
        il_metrics = perform_il_update(
            config.gumbeldore_config["destination_path"], network, optimizer, config
        )

    # ====== 4. COMBINE METRICS AND RETURN ======
    combined_metrics = {**rl_metrics, **il_metrics}
    combined_metrics["best_gen_obj"] = combined_metrics.get("best_reward", float("-inf"))

    print(f"[Hybrid] RL_reward_mean={rl_metrics.get('mean_reward', float('nan')):.4f}  "
          f"RL_policy_loss={rl_metrics.get('policy_loss', float('nan')):.6f}  "
          f"IL_distill_loss={il_metrics.get('il_loss', float('nan')):.4f}")

    # Create top 20 text artifact from the elite buffer for logging
    top_20_text_lines = []
    for i, entry in enumerate(updated_elite_records_as_dicts[:20]):
        top_20_text_lines.append(f"{i + 1:02d}: {entry['smiles']}  obj={entry['obj']:.4f}")
    if not top_20_text_lines:
        top_20_text_lines.append("No molecules in elite buffer")

    return combined_metrics, top_20_text_lines, updated_elite_records_as_dicts

# ---------------- RL (Dr. GRPO) epoch ---------------- #
def train_for_one_epoch_rl(epoch: int,
                           config: MoleculeConfig,
                           network: MoleculeTransformer,
                           network_weights: dict,
                           optimizer: torch.optim.Optimizer,
                           objective_evaluator: MoleculeObjectiveEvaluator):
    """
    RL fine-tuning epoch:
      1. Generate trajectories (terminated molecules) with current policy.
      2. Run policy gradient update via dr_grpo_update.
      3. Produce logging artifacts similar in spirit to supervised path.
    """
    print(f"[RL] Generating trajectories (epoch {epoch + 1})...")
    gumbeldore_dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_evaluator)

    # Return raw terminated trajectories (list of MoleculeDesign)
    trajectories = gumbeldore_dataset.generate_dataset(
        network_weights=network_weights,
        best_objective=None,
        memory_aggressive=False
    )

    if len(trajectories) == 0:
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
        designs=trajectories,
        config=config,
        device=torch.device(config.training_device),
        logger=None
    )
    print("dr GRPO update done.")
    metrics["best_gen_obj"] = metrics.get("best_reward", float("-inf"))
    metrics["mean_best_gen_obj"] = metrics.get("mean_reward", float("-inf"))
    metrics.setdefault("loss_level_zero", 0.0)
    metrics.setdefault("loss_level_one", 0.0)
    metrics.setdefault("loss_level_two", 0.0)

    # Build top 20 text artifact
    mol_map = {}
    for m in trajectories:
        if m.objective is None:
            continue
        if m.smiles_string not in mol_map or mol_map[m.smiles_string]["obj"] < m.objective:
            mol_map[m.smiles_string] = {
                "smiles": m.smiles_string,
                "obj": m.objective
            }
    unique_mols = list(mol_map.values())
    unique_mols.sort(key=lambda x: x["obj"], reverse=True)
    top20 = unique_mols[:20]

    top_20_text_lines = []
    for i, entry in enumerate(top20):
        top_20_text_lines.append(f"{i + 1:02d}: {entry['smiles']}  obj={entry['obj']:.4f}")
    if not top20:
        top_20_text_lines.append("No terminated molecules")

    return metrics, top_20_text_lines


# ---------------- Evaluation ---------------- #
def evaluate(eval_type: str, config: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Uses generation (supervised-style metrics) for evaluation irrespective of RL training.
    """
    config = copy.deepcopy(config)
    config.gumbeldore_config["destination_path"] = None

    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )
    metrics = gumbeldore_dataset.generate_dataset(copy.deepcopy(network.get_weights()), memory_aggressive=False)
    top_20_mols = metrics["top_20_molecules"]
    metrics = {
        f"{eval_type}_mean_top_20_obj": metrics["mean_top_20_obj"],
        f"{eval_type}_mean_top_20_sa_score": metrics["mean_top_20_sa_score"],
        f"{eval_type}_best_obj": metrics['best_gen_obj'],
        f"{eval_type}_best_mol_sa_score": metrics['best_gen_sa_score'],
    }
    print("Evaluation done")
    print(f"Eval ({eval_type}) best obj: {metrics[f'{eval_type}_best_obj']:.3f}")
    print(f"Eval ({eval_type}) mean top 20 obj: {metrics[f'{eval_type}_mean_top_20_obj']:.3f}")

    return metrics, top_20_mols


def update_elite_buffer_pickle(new_records: List[TrajectoryRecord], config: MoleculeConfig) -> List[dict]:
    """
    Loads an elite buffer from a pickle file, merges new trajectory records,
    sorts/truncates, and saves it back to the file. Mimics original GraphXForm logic.

    Returns:
        The updated elite buffer as a list of dictionaries, ready for dataset loading.
    """
    destination_path = config.gumbeldore_config["destination_path"]
    capacity = config.gumbeldore_config["num_trajectories_to_keep"]

    # 1. Load existing buffer from disk, if it exists.
    existing_mols_by_smiles = {}
    if os.path.isfile(destination_path):
        try:
            with open(destination_path, "rb") as f:
                # The file stores a list of dictionaries
                existing_mols = pickle.load(f)
                for mol_dict in existing_mols:
                    existing_mols_by_smiles[mol_dict["smiles"]] = mol_dict
        except (EOFError, pickle.UnpicklingError):
            print(f"[Warning] Could not read existing buffer at {destination_path}. Starting fresh.")

    # 2. Convert new TrajectoryRecord objects to the same dict format and merge.
    for r in new_records:
        smiles = r.design.smiles_string
        if smiles is None:
            continue

        # If the new record is better, add/replace it.
        if smiles not in existing_mols_by_smiles or r.reward > existing_mols_by_smiles[smiles]["obj"]:
            existing_mols_by_smiles[smiles] = {
                "start_atom": r.design.initial_atom,
                "action_seq": r.history,
                "smiles": smiles,
                "obj": r.reward
                # "sa_score": r.design.sa_score # You can add this back if needed
            }

    # 3. Sort, truncate, and save back to disk.
    all_records_as_dicts = sorted(
        list(existing_mols_by_smiles.values()),
        key=lambda x: x["obj"],
        reverse=True
    )
    truncated_records = all_records_as_dicts[:capacity]

    with open(destination_path, "wb") as f:
        pickle.dump(truncated_records, f)

    print(f"  Elite buffer updated. Size: {len(truncated_records)}. "
          f"Best obj: {truncated_records[0]['obj']:.4f}" if truncated_records else "Buffer is empty.")

    return truncated_records


def perform_il_update(destination_path: str,
                      network: MoleculeTransformer,
                      optimizer: torch.optim.Optimizer,
                      config: MoleculeConfig):
    """
    Performs one epoch of supervised Imitation Learning (CEM-SIL style)
    by loading data from the specified pickle file.
    """
    if not os.path.exists(destination_path):
        print("[IL] Elite buffer pickle file not found. Skipping IL update step.")
        return {"il_loss": 0.0}

    # Uses existing config parameters as requested
    dataset = RandomMoleculeDataset(
        config,
        path_to_pickle=destination_path,
        batch_size=config.batch_size_training,
        custom_num_batches=config.num_batches_per_epoch
    )

    if len(dataset) == 0:
        print("[IL] Could not create any training batches from elite buffer. Skipping IL update step.")
        return {"il_loss": 0.0}

    # The rest of this function is identical to the previous version
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=config.num_dataloader_workers, pin_memory=True)

    network.train()
    if config.freeze_all_except_final_layer:
        for p in network.parameters():
            p.requires_grad = False
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    total_loss = 0.0
    for data in tqdm(dataloader, desc="IL Update"):
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)

        logits_zero, logits_one, logits_two = network(input_data)

        logits_zero[input_data["feasibility_mask_level_zero"]] = float("-inf")
        logits_one[input_data["feasibility_mask_level_one"]] = float("-inf")
        logits_two[input_data["feasibility_mask_level_two"]] = float("-inf")

        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
        loss_zero = criterion(logits_zero, target_zero)
        loss_one = criterion(logits_one, target_one)
        loss_two = criterion(logits_two, target_two)
        loss = (torch.nan_to_num(loss_zero) +
                config.scale_factor_level_one * torch.nan_to_num(loss_one) +
                config.scale_factor_level_two * torch.nan_to_num(loss_two))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    print(f"[IL] Imitation Learning update complete. Average Loss: {avg_loss:.4f}")
    return {"il_loss": avg_loss}


if __name__ == '__main__':
    print(">> Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    args = parser.parse_args()

    if args.config is not None:
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig
    config = MoleculeConfig()

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
        wandb.config.update({"task": config.objective_type})  # Log the task separately for easy filtering

    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, logging_level="info")
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
    objective_evaluator = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)

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

        for epoch in range(config.num_epochs):
            print("------")
            network_weights = copy.deepcopy(network.get_weights())

            rl_mode_active = getattr(config, "use_dr_grpo", False)

            if rl_mode_active:
                # The hybrid logic is now encapsulated within this single function
                generated_loggable_dict, top20_text, _ = train_for_one_epoch_hybrid(
                    epoch, config, network, network_weights, optimizer, objective_evaluator
                )
                # The last return value (the buffer data) is not needed in the main loop, so we use _
                val_metric = generated_loggable_dict.get("best_gen_obj", float("-inf"))

            else:  # Original Supervised-only mode
                generated_loggable_dict, top20_text = train_for_one_epoch_supervised(
                    epoch, config, network, network_weights, optimizer, objective_evaluator, best_validation_metric
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
            logger.log_metrics(generated_loggable_dict, step=epoch)

            # --- METRIC & CHECKPOINT LOGIC ---
            # First, update the overall best metric
            if val_metric > best_validation_metric:
                print(">> Got new best model.")
                best_validation_metric = val_metric
                checkpoint["best_model_weights"] = copy.deepcopy(network.get_weights())
                checkpoint["best_validation_metric"] = best_validation_metric
                save_checkpoint(checkpoint, "best_model.pt", config)

            # --- WANDB LOGGING (NOW USES UPDATED METRIC) ---
            if hasattr(config, 'use_wandb') and config.use_wandb:
                wandb_log = {
                    "epoch": checkpoint["epochs_trained"],
                    "best_epoch_objective": val_metric,
                    "best_overall_objective": best_validation_metric
                }
                # Add specific RL metrics if in RL mode
                if rl_mode_active:
                    wandb_log["mean_reward"] = generated_loggable_dict.get('mean_reward', float('nan'))
                    wandb_log["policy_loss"] = generated_loggable_dict.get('policy_loss', float('nan'))
                    wandb_log["num_trajectories"] = generated_loggable_dict.get('num_trajectories', 0)
                    wandb_log["mean_trajectory_length"] = generated_loggable_dict.get('mean_traj_length', 0)

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
        test_loggable_dict, test_text_to_save = evaluate('test', config, network, objective_evaluator)
    print(">> TEST")
    print(test_loggable_dict)
    logger.log_metrics(test_loggable_dict, step=0, step_desc="test")
    print(test_text_to_save)
    logger.text_artifact(os.path.join(config.results_path, "test_top_20_molecules.txt"),
                         test_text_to_save)

    # --- WANDB FINISH ---
    if hasattr(config, 'use_wandb') and config.use_wandb:
        wandb.finish()

    print("Finished. Shutting down ray.")
    ray.shutdown()