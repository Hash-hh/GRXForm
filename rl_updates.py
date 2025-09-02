import math
from dataclasses import dataclass
from typing import List, Optional
import torch
from contextlib import nullcontext

from molecule_design import MoleculeDesign
from model.molecule_transformer import MoleculeTransformer


@dataclass
class TrajectoryRecord:
    design: MoleculeDesign
    history: List[int]
    reward: float
    log_prob_sum: Optional[torch.Tensor] = None
    length: Optional[int] = None
    advantage: Optional[float] = None
    replay_failed: bool = False  # placeholder (unused)


# ------------------ Record & Advantage Utilities ------------------ #

def filter_and_build_records(designs: List[MoleculeDesign]) -> (List[TrajectoryRecord], int, int):
    records: List[TrajectoryRecord] = []
    none_dropped = 0
    nonfinite_dropped = 0
    for d in designs:
        if not d.history:
            continue
        # Termination condition: last action==0 and synthesis_done
        if not (d.history and d.history[-1] == 0 and d.synthesis_done):
            continue
        obj = d.objective
        if obj is None:
            none_dropped += 1
            continue
        if not math.isfinite(obj):
            nonfinite_dropped += 1
            continue
        records.append(TrajectoryRecord(design=d, history=list(d.history), reward=float(obj)))
    if none_dropped or nonfinite_dropped:
        print(f"[DR-GRPO] filter: dropped none={none_dropped} nonfinite={nonfinite_dropped} kept={len(records)}")
    return records, none_dropped, nonfinite_dropped


def compute_baseline_and_advantages(records: List[TrajectoryRecord],
                                    normalize: bool = False,
                                    config=None) -> float:
    """
    Computes trajectory-level advantages A_i and stores them in each record.
    If EMA baseline configured, uses baseline BEFORE updating it (standard).
    """
    if not records:
        return 0.0
    rewards = torch.tensor([r.reward for r in records], dtype=torch.float32)
    batch_mean = rewards.mean().item()

    use_ema = bool(getattr(config, "rl_use_ema_baseline", False)) if config is not None else False
    if use_ema:
        alpha = float(getattr(config, "rl_baseline_ema_alpha", 0.9))
        if not hasattr(config, "_ema_baseline"):
            config._ema_baseline = batch_mean
            config._ema_initialized = True
        baseline = float(config._ema_baseline)
        config._ema_baseline = alpha * config._ema_baseline + (1.0 - alpha) * batch_mean
        if config is not None:
            config._last_batch_mean_reward = batch_mean
    else:
        baseline = batch_mean
        if config is not None:
            config._last_batch_mean_reward = batch_mean

    advantages = rewards - baseline
    if normalize:
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std

    for i, r in enumerate(records):
        r.advantage = float(advantages[i].item())
    return baseline


def _fresh_initial_clone(final_design: MoleculeDesign) -> MoleculeDesign:
    # atoms[0] = virtual atom; atoms[1] first real atom (confirmed)
    first_atom_token = final_design.atoms[1]
    if hasattr(first_atom_token, "item"):
        first_atom_token = int(first_atom_token.item())
    return MoleculeDesign(config=final_design.config, initial_atom=first_atom_token)


def _prepare_masked_log_probs(level_logits: torch.Tensor,
                              mask_list,
                              device: torch.device,
                              assert_masks: bool = True):
    """
    Applies infeasible mask (True=infeasible), returns:
      log_probs (log-softmax over full vector with infeasible = -inf),
      infeasible_mask (bool tensor),
      feasible_mask (bool tensor)
    If assert_masks=True: structural checks on shape & at least one feasible action.
    """
    if isinstance(mask_list, torch.Tensor):
        infeasible_mask = mask_list.to(device=device, dtype=torch.bool)
    else:
        infeasible_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
    if assert_masks:
        assert level_logits.shape[0] >= infeasible_mask.shape[0], \
            f"Logits shorter ({level_logits.shape[0]}) than mask ({infeasible_mask.shape[0]})"
    level_logits = level_logits[:infeasible_mask.shape[0]]
    feasible_mask = ~infeasible_mask
    feasible_any = feasible_mask.any().item()
    if assert_masks:
        assert feasible_any, "All actions masked; invalid environment state."
    if not feasible_any:
        # fallback zero logits to avoid NaNs in log_softmax; action feasibility guard later will raise if chosen infeasible
        level_logits = torch.zeros_like(level_logits)
    level_logits = level_logits.masked_fill(infeasible_mask, float("-inf"))
    log_probs = torch.log_softmax(level_logits, dim=-1)
    return log_probs, infeasible_mask, feasible_mask


def _make_autocast_ctx(config):
    if not getattr(config, "use_amp", False):
        return nullcontext(), None
    amp_dtype = getattr(config, "amp_dtype", "bf16").lower()
    if amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:  # bf16
        scaler = None
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return ctx, scaler


# ------------------ Streaming Update (Only Path) ------------------ #

def _streaming_update_normalized(
        model: MoleculeTransformer,
        optimizer: torch.optim.Optimizer,
        records: List[TrajectoryRecord],
        config,
        device: torch.device):
    """
    Streaming replay + backward with optional length & feasible-count normalization.

    If rl_entropy_length_normalize:
        L = -(Σ_t A_i log π(a_t|s_t))/S_total - β * (Σ_t H_norm_t)/S_total
    Else:
        L = -(1/N) Σ_i Σ_t A_i log π(a_t|s_t) - β * (1/N) Σ_t H_norm_t

    H_raw_t = -Σ_{feasible a} p(a) log p(a)
    H_norm_t:
        if feasible_count <= 1 -> 0
        else H_raw_t / log(feasible_count) if rl_entropy_use_feasible_log_scaling else H_raw_t

    Per-level entropy metrics count only states with feasible_count > 1 (since others have 0 entropy).
    """
    assert all(r.advantage is not None for r in records), "Compute advantages first."
    N = len(records)
    if N == 0:
        return {
            "policy_loss_value": 0.0,
            "mean_entropy_norm": 0.0,
            "mean_entropy_raw": 0.0,
            "per_level_entropy_norm": {0: 0.0, 1: 0.0, 2: 0.0},
            "per_level_avg_feasible": {0: 0.0, 1: 0.0, 2: 0.0},
            "total_states": 0
        }

    for r in records:
        r.length = len(r.history)
        r.log_prob_sum = torch.tensor(0.0, device=device)

    S_total = sum(r.length for r in records)
    if S_total == 0:
        return {
            "policy_loss_value": 0.0,
            "mean_entropy_norm": 0.0,
            "mean_entropy_raw": 0.0,
            "per_level_entropy_norm": {0: 0.0, 1: 0.0, 2: 0.0},
            "per_level_avg_feasible": {0: 0.0, 1: 0.0, 2: 0.0},
            "total_states": 0
        }

    length_norm = bool(getattr(config, "rl_entropy_length_normalize", True))
    feasible_log_scaling = bool(getattr(config, "rl_entropy_use_feasible_log_scaling", True))
    entropy_coef = float(getattr(config, "rl_entropy_coef", 0.0) or 0.0)
    assert_masks = bool(getattr(config, "rl_assert_masks", False))
    debug_entropy = bool(getattr(config, "rl_debug_entropy", False))
    debug_print_first = int(getattr(config, "rl_debug_entropy_print_first", 12)) \
        if hasattr(config, "rl_debug_entropy_print_first") else 12

    # Optional one-time reminder when entropy disabled but metrics computed
    if entropy_coef == 0.0 and debug_entropy and not hasattr(config, "_warned_entropy_off"):
        print("[ENTROPY] entropy_coef=0.0: metrics still computed (no gradient contribution).")
        config._warned_entropy_off = True

    denom_policy = S_total if length_norm else N
    denom_entropy = S_total if length_norm else N

    # Float32 accumulators (more stable under AMP)
    sum_adv_logp = 0.0
    sum_entropy_raw = 0.0
    sum_entropy_norm = 0.0

    per_level_entropy_norm_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    per_level_entropy_norm_count = {0: 0, 1: 0, 2: 0}
    per_level_feasible_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    per_level_feasible_count = {0: 0, 1: 0, 2: 0}

    debug_lines_printed = 0

    micro = getattr(config, "rl_replay_microbatch_size", 0) or N
    assert micro > 0, "Microbatch size must be > 0"

    model.eval()  # deterministic logits (dropout=0 anyway)
    autocast_ctx, scaler = _make_autocast_ctx(config)
    optimizer.zero_grad(set_to_none=True)

    for start in range(0, N, micro):
        batch_records = records[start:start + micro]
        clones = [_fresh_initial_clone(r.design) for r in batch_records]
        cursors = [0] * len(batch_records)
        finished = [False] * len(batch_records)

        while True:
            active_local_indices = [
                i for i, (r, cur, fin, cl) in enumerate(zip(batch_records, cursors, finished, clones))
                if (not fin) and (cur < len(r.history)) and (not cl.synthesis_done)
            ]
            if not active_local_indices:
                break

            active_clones = [clones[i] for i in active_local_indices]
            with autocast_ctx:
                head0, head1, head2 = model(MoleculeDesign.list_to_batch(active_clones, device=device))

            # Keep loss_batch in float32 for stability
            loss_batch = torch.zeros((), device=device, dtype=torch.float32)

            for pos_in_active, local_idx in enumerate(active_local_indices):
                rec = batch_records[local_idx]
                clone = clones[local_idx]
                cursor = cursors[local_idx]

                lvl = clone.current_action_level
                if lvl == 0:
                    level_logits = head0[pos_in_active]
                elif lvl == 1:
                    level_logits = head1[pos_in_active]
                else:
                    level_logits = head2[pos_in_active]

                log_probs, infeasible_mask, feasible_mask = _prepare_masked_log_probs(
                    level_logits, clone.current_action_mask, device, assert_masks=assert_masks
                )

                feasible_count = int(feasible_mask.sum().item())

                # Unconditional guard to avoid -inf gradient contamination
                action = rec.history[cursor]
                if action >= log_probs.shape[0]:
                    raise RuntimeError(f"[StreamingReplay] Action {action} out of bounds {log_probs.shape[0]}")
                # UNCONDITIONAL FEASIBLE ACTION GUARD (prevent NaNs even if assert_masks is False)
                if infeasible_mask[action]:
                    raise RuntimeError(f"Chosen action {action} is masked infeasible (lvl={lvl}).")

                chosen_logp = log_probs[action]

                # Detached stats for logging
                rec.log_prob_sum = rec.log_prob_sum + chosen_logp.detach().float()
                sum_adv_logp += rec.advantage * float(chosen_logp.detach().cpu())

                # Policy gradient contribution (cast to float32)
                loss_batch = loss_batch - (rec.advantage / denom_policy) * chosen_logp.float()

                # Always compute entropy metrics (even if entropy_coef==0) for visibility
                if feasible_count > 1:
                    lp_feas = log_probs[feasible_mask]
                    p_feas = lp_feas.exp()
                    H_raw = -(p_feas * lp_feas).sum()  # tensor
                    if feasible_log_scaling:
                        H_norm = H_raw / math.log(feasible_count)
                    else:
                        H_norm = H_raw

                    H_raw_val = float(H_raw.detach().cpu())
                    H_norm_val = float(H_norm.detach().cpu())
                    sum_entropy_raw += H_raw_val
                    sum_entropy_norm += H_norm_val
                    per_level_entropy_norm_sum[lvl] += H_norm_val
                    per_level_entropy_norm_count[lvl] += 1

                    if entropy_coef > 0.0:
                        loss_batch = loss_batch - entropy_coef * (H_norm.float() / denom_entropy)

                    if debug_entropy and debug_lines_printed < debug_print_first:
                        print(f"[ENTROPY-DBG] lvl={lvl} feas={feasible_count} "
                              f"H_raw={H_raw_val:.4f} H_norm={H_norm_val:.4f} "
                              f"action={action} logp={float(chosen_logp.detach().cpu()):.4f}")
                        debug_lines_printed += 1
                else:
                    # Entropy = 0 in this state; still optionally debug
                    if debug_entropy and debug_lines_printed < debug_print_first:
                        print(f"[ENTROPY-DBG] lvl={lvl} feas={feasible_count} H_raw=0.0000 H_norm=0.0000 "
                              f"action={action} logp={float(chosen_logp.detach().cpu()):.4f}")
                        debug_lines_printed += 1

                # Feasible count stats (includes states with K<=1)
                per_level_feasible_sum[lvl] += feasible_count
                per_level_feasible_count[lvl] += 1

                # Advance environment clone
                clone.take_action(action)
                cursors[local_idx] += 1
                if clone.synthesis_done or cursors[local_idx] >= len(rec.history):
                    finished[local_idx] = True

            # Backward micro-step
            if scaler is not None:
                scaler.scale(loss_batch).backward()
            else:
                loss_batch.backward()

    # Optimizer step
    clip_val = getattr(config, "optimizer", {}).get("gradient_clipping", 0) \
        if hasattr(config, "optimizer") else 0
    if scaler is not None:
        if clip_val and clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_val and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

    policy_loss_value = -(sum_adv_logp / denom_policy)

    # Metrics
    mean_entropy_raw = (sum_entropy_raw / S_total) if S_total > 0 else 0.0
    mean_entropy_norm = (sum_entropy_norm / S_total) if S_total > 0 else 0.0

    per_level_entropy_norm_mean = {}
    per_level_feasible_avg = {}
    for lvl in (0, 1, 2):
        if per_level_entropy_norm_count[lvl] > 0:
            per_level_entropy_norm_mean[lvl] = per_level_entropy_norm_sum[lvl] / per_level_entropy_norm_count[lvl]
        else:
            per_level_entropy_norm_mean[lvl] = 0.0
        if per_level_feasible_count[lvl] > 0:
            per_level_feasible_avg[lvl] = per_level_feasible_sum[lvl] / per_level_feasible_count[lvl]
        else:
            per_level_feasible_avg[lvl] = 0.0

    # Store for external use
    config._last_entropy_norm_mean = mean_entropy_norm
    config._last_entropy_raw_mean = mean_entropy_raw
    for lvl in (0, 1, 2):
        setattr(config, f"_last_entropy_norm_mean_level{lvl}", per_level_entropy_norm_mean[lvl])
        setattr(config, f"_last_avg_feasible_count_level{lvl}", per_level_feasible_avg[lvl])

    return {
        "policy_loss_value": float(policy_loss_value),
        "mean_entropy_norm": float(mean_entropy_norm),
        "mean_entropy_raw": float(mean_entropy_raw),
        "per_level_entropy_norm": per_level_entropy_norm_mean,
        "per_level_avg_feasible": per_level_feasible_avg,
        "total_states": S_total
    }


# ------------------ Logging Helper ------------------ #

def compute_policy_loss_for_logging(records: List[TrajectoryRecord], config) -> float:
    """
    Recompute detached policy loss consistent with normalization flag:
      If length-normalized: -(Σ_i A_i Σ_t logp_i,t) / S_total
      Else:                  -(1/N) Σ_i A_i Σ_t logp_i,t
    """
    if not records:
        return 0.0
    length_norm = bool(getattr(config, "rl_entropy_length_normalize", True))
    N = len(records)
    S_total = sum(len(r.history) for r in records)
    denom = S_total if length_norm else N

    adv_tensor = torch.tensor([r.advantage for r in records], dtype=torch.float32)
    weighted_terms = []
    for r, adv in zip(records, adv_tensor):
        if r.log_prob_sum is None:
            continue
        weighted_terms.append(adv * r.log_prob_sum.detach().cpu().float())
    if not weighted_terms:
        return 0.0
    total = torch.stack(weighted_terms).sum().item()
    return float(-(total / denom))


# ------------------ Main Update Entry ------------------ #

def dr_grpo_update(model: MoleculeTransformer,
                   optimizer: torch.optim.Optimizer,
                   designs: List[MoleculeDesign],
                   config,
                   device: torch.device,
                   logger=None):
    records, none_dropped, nonfinite_dropped = filter_and_build_records(designs)
    if not records:
        metrics = {
            "skipped": True,
            "num_trajectories": 0,
            "mean_reward": float("-inf"),
            "best_reward": float("-inf"),
            "policy_loss": 0.0,
            "invalid_dropped": nonfinite_dropped,
            "none_dropped": none_dropped
        }
        if logger:
            logger.info(f"[DR-GRPO] {metrics}")
        return metrics

    normalize_adv = getattr(config, "rl_advantage_normalize", False)
    baseline = compute_baseline_and_advantages(records, normalize=normalize_adv, config=config)

    update_stats = _streaming_update_normalized(model, optimizer, records, config, device)

    rewards = [r.reward for r in records]
    advantages_vals = [r.advantage for r in records]
    mean_reward = sum(rewards) / len(rewards)
    mean_adv = sum(advantages_vals) / len(advantages_vals)
    std_adv = (sum((a - mean_adv) ** 2 for a in advantages_vals) / len(advantages_vals)) ** 0.5 if advantages_vals else 0.0

    entropy_coef = float(getattr(config, "rl_entropy_coef", 0.0) or 0.0)
    policy_loss_logged = update_stats["policy_loss_value"]

    metrics = {
        "baseline": baseline,
        "batch_mean_reward": float(getattr(config, "_last_batch_mean_reward", baseline)),
        "ema_baseline_active": bool(getattr(config, "rl_use_ema_baseline", False)),
        "mean_reward": float(mean_reward),
        "best_reward": float(max(rewards)),
        "mean_advantage": float(mean_adv),
        "std_advantage": float(std_adv),
        "fraction_pos_adv": float(sum(a > 0 for a in advantages_vals) / len(advantages_vals)),
        "policy_loss": float(policy_loss_logged),
        "num_trajectories": len(records),
        "mean_traj_length": float(sum(r.length for r in records) / len(records)) if records else 0.0,
        "invalid_dropped": nonfinite_dropped,
        "none_dropped": none_dropped,
        "amp_enabled": bool(getattr(config, "use_amp", False)),
        "amp_dtype": getattr(config, "amp_dtype", "none"),
        "adv_norm": bool(normalize_adv),
        "entropy_coef": float(entropy_coef),
        "mean_entropy_norm": float(update_stats["mean_entropy_norm"]),
        "mean_entropy_raw": float(update_stats["mean_entropy_raw"]),
        "mean_entropy_norm_level0": float(update_stats["per_level_entropy_norm"][0]),
        "mean_entropy_norm_level1": float(update_stats["per_level_entropy_norm"][1]),
        "mean_entropy_norm_level2": float(update_stats["per_level_entropy_norm"][2]),
        "avg_feasible_count_level0": float(update_stats["per_level_avg_feasible"][0]),
        "avg_feasible_count_level1": float(update_stats["per_level_avg_feasible"][1]),
        "avg_feasible_count_level2": float(update_stats["per_level_avg_feasible"][2]),
        "total_states": int(update_stats["total_states"]),
        "length_normalized": bool(getattr(config, "rl_entropy_length_normalize", True)),
        "feasible_log_scaling": bool(getattr(config, "rl_entropy_use_feasible_log_scaling", True)),
    }
    if logger:
        logger.info(f"[DR-GRPO] {metrics}")
    return metrics