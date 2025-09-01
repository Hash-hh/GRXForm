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
    replay_failed: bool = False  # marks mismatch / failure during replay


def filter_and_build_records(designs: List[MoleculeDesign]) -> (List[TrajectoryRecord], int, int):
    records: List[TrajectoryRecord] = []
    none_dropped = 0
    nonfinite_dropped = 0
    for d in designs:
        if not d.history:
            continue
        if not (d.history[-1] == 0 and d.synthesis_done):
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
                                    normalize: bool = False) -> float:
    if not records:
        return 0.0
    rewards = torch.tensor([r.reward for r in records], dtype=torch.float32)
    baseline = rewards.mean().item()
    advantages = rewards - rewards.mean()
    if normalize:
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std
    for i, r in enumerate(records):
        r.advantage = advantages[i].item()
    return baseline


def _fresh_initial_clone(final_design: MoleculeDesign) -> MoleculeDesign:
    first_atom_token = final_design.atoms[1]
    if hasattr(first_atom_token, "item"):
        first_atom_token = int(first_atom_token.item())
    return MoleculeDesign(config=final_design.config, initial_atom=first_atom_token)


def _prepare_masked_log_probs(level_logits: torch.Tensor,
                              mask_list,
                              device: torch.device,
                              assert_masks: bool = True) -> torch.Tensor:
    if isinstance(mask_list, torch.Tensor):
        infeasible_mask = mask_list.to(device=device, dtype=torch.bool)
    else:
        infeasible_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
    if assert_masks:
        # Basic structural assertions
        assert level_logits.shape[0] >= infeasible_mask.shape[0], \
            f"Logits shorter ({level_logits.shape[0]}) than mask ({infeasible_mask.shape[0]})"
    # Trim logits to mask length
    level_logits = level_logits[:infeasible_mask.shape[0]]
    feasible_any = (~infeasible_mask).any().item()
    if assert_masks:
        assert feasible_any, "All actions masked; invalid environment state."
    if not feasible_any:
        # Fallback: return uniform - will likely get caught elsewhere
        level_logits = torch.zeros_like(level_logits)
    level_logits = level_logits.masked_fill(infeasible_mask, float("-inf"))
    return torch.log_softmax(level_logits, dim=-1)


def sequential_replay_and_compute_log_probs(
        model: MoleculeTransformer,
        records: List[TrajectoryRecord],
        device: torch.device,
        autocast_ctx,
        assert_masks: bool,
        soft_fail: bool,
        debug_verify: bool = False
):
    # Use eval() to disable dropout for deterministic reconstruction
    model.eval()
    for rec in records:
        design_clone = _fresh_initial_clone(rec.design)
        log_prob_sum = None
        steps = 0
        for action in rec.history:
            if design_clone.synthesis_done:
                break
            with autocast_ctx:
                batch = MoleculeDesign.list_to_batch([design_clone], device=device)
                # Forward (dropout disabled)
                logits_per_level = list(model(batch))
                current_level = design_clone.current_action_level
                level_logits = logits_per_level[current_level][0]
                mask_list = design_clone.current_action_mask
                log_probs = _prepare_masked_log_probs(
                    level_logits, mask_list, device, assert_masks=assert_masks
                )

            if action >= log_probs.shape[0]:
                msg = (f"[DR-GRPO][Replay] Action index {action} >= log_probs length "
                       f"{log_probs.shape[0]} (level={design_clone.current_action_level})")
                if soft_fail:
                    print(msg + " -> marking replay_failed")
                    rec.replay_failed = True
                    log_prob_sum = None
                    steps = 0
                    break
                else:
                    raise RuntimeError(msg)

            chosen_logp = log_probs[action].float()
            log_prob_sum = chosen_logp if log_prob_sum is None else (log_prob_sum + chosen_logp)
            steps += 1

            if debug_verify:
                pass  # place for future consistency checks

            design_clone.take_action(action)
            if design_clone.synthesis_done:
                break

        if rec.replay_failed or log_prob_sum is None:
            rec.log_prob_sum = torch.tensor(0.0, device=device, requires_grad=True)
            rec.length = 0
        else:
            rec.log_prob_sum = log_prob_sum
            rec.length = steps


def batched_replay_and_compute_log_probs(
        model: MoleculeTransformer,
        records: List[TrajectoryRecord],
        device: torch.device,
        autocast_ctx,
        assert_masks: bool,
        soft_fail: bool,
        debug_verify: bool = False
):
    model.eval()
    n = len(records)
    if n == 0:
        return

    clones = [_fresh_initial_clone(rec.design) for rec in records]
    cursors = [0] * n
    running_sums = [None] * n
    lengths = [0] * n

    active_indices = [i for i in range(n)]
    while active_indices:
        batch_clones = [clones[i] for i in active_indices]

        with autocast_ctx:
            head0, head1, head2 = model(MoleculeDesign.list_to_batch(batch_clones, device=device))

        to_remove = []
        for local_pos, rec_idx in enumerate(active_indices):
            rec = records[rec_idx]
            if rec.replay_failed:
                to_remove.append(rec_idx)
                continue

            clone = clones[rec_idx]
            cursor = cursors[rec_idx]
            if cursor >= len(rec.history) or clone.synthesis_done:
                to_remove.append(rec_idx)
                continue

            current_level = clone.current_action_level
            if current_level == 0:
                level_logits = head0[local_pos]
            elif current_level == 1:
                level_logits = head1[local_pos]
            else:
                level_logits = head2[local_pos]

            mask_list = clone.current_action_mask
            log_probs = _prepare_masked_log_probs(
                level_logits, mask_list, device, assert_masks=assert_masks
            )
            action = rec.history[cursor]

            if action >= log_probs.shape[0]:
                msg = (f"[DR-GRPO][Replay] Action index {action} >= log_probs length "
                       f"{log_probs.shape[0]} (level={clone.current_action_level})")
                if soft_fail:
                    print(msg + " -> marking replay_failed")
                    rec.replay_failed = True
                    running_sums[rec_idx] = None
                    lengths[rec_idx] = 0
                    to_remove.append(rec_idx)
                    continue
                else:
                    raise RuntimeError(msg)

            chosen_logp = log_probs[action].float()
            running_sums[rec_idx] = chosen_logp if running_sums[rec_idx] is None else (running_sums[rec_idx] + chosen_logp)
            lengths[rec_idx] += 1
            cursors[rec_idx] += 1

            clone.take_action(action)

            if clone.synthesis_done or cursors[rec_idx] >= len(rec.history):
                to_remove.append(rec_idx)

        if to_remove:
            finished_set = set(to_remove)
            active_indices = [i for i in active_indices if i not in finished_set]

    for i, rec in enumerate(records):
        if rec.replay_failed or running_sums[i] is None:
            rec.log_prob_sum = torch.tensor(0.0, device=device, requires_grad=True)
            rec.length = 0
        else:
            rec.log_prob_sum = running_sums[i]
            rec.length = lengths[i]


def compute_policy_loss(records: List[TrajectoryRecord],
                        entropy_coef: float = 0.0) -> torch.Tensor:
    if not records:
        return torch.tensor(0.0)
    log_probs = torch.stack([r.log_prob_sum for r in records])
    advantages = torch.tensor([r.advantage for r in records],
                              dtype=log_probs.dtype,
                              device=log_probs.device)
    loss = -(advantages * log_probs).mean()
    # entropy_coef placeholder (requires per-step distributions)
    return loss


def _make_autocast_ctx(config):
    if not getattr(config, "use_amp", False):
        return nullcontext(), None  # context, scaler
    amp_dtype = getattr(config, "amp_dtype", "bf16").lower()
    if amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:  # bf16 or default
        scaler = None
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return ctx, scaler


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
    assert_masks = getattr(config, "rl_assert_masks", True)
    soft_fail = getattr(config, "rl_soft_replay_failure", False)

    # Initial baseline (will recompute if any replay failures & soft_fail True)
    baseline = compute_baseline_and_advantages(records, normalize=normalize_adv)

    use_batched = getattr(config, "rl_batched_replay", True) and len(records) > 1
    debug_flag = getattr(config, "rl_debug_verify_replay", False)
    autocast_ctx, scaler = _make_autocast_ctx(config)

    if use_batched:
        batched_replay_and_compute_log_probs(
            model, records, device,
            autocast_ctx=autocast_ctx,
            assert_masks=assert_masks,
            soft_fail=soft_fail,
            debug_verify=debug_flag
        )
    else:
        sequential_replay_and_compute_log_probs(
            model, records, device,
            autocast_ctx=autocast_ctx,
            assert_masks=assert_masks,
            soft_fail=soft_fail,
            debug_verify=debug_flag
        )

    # If soft_fail mode and any replay_failed, drop them & recompute baseline/advantages
    if soft_fail:
        failed = [r for r in records if r.replay_failed]
        if failed:
            kept = [r for r in records if not r.replay_failed]
            if kept:
                baseline = compute_baseline_and_advantages(kept, normalize=normalize_adv)
                records = kept
            else:
                metrics = {
                    "skipped": True,
                    "num_trajectories": 0,
                    "mean_reward": float("-inf"),
                    "best_reward": float("-inf"),
                    "policy_loss": 0.0,
                    "invalid_dropped": nonfinite_dropped,
                    "none_dropped": none_dropped,
                    "replay_failed": len(failed)
                }
                if logger:
                    logger.info(f"[DR-GRPO] {metrics}")
                return metrics

    loss = compute_policy_loss(records, entropy_coef=getattr(config, "rl_entropy_coef", 0.0))

    if not torch.isfinite(loss):
        if logger:
            logger.warning("[DR-GRPO] Non-finite loss; skipping step.")
        metrics = {
            "baseline": baseline,
            "mean_reward": float(sum(r.reward for r in records) / len(records)),
            "best_reward": float(max(r.reward for r in records)),
            "mean_advantage": float(sum(r.advantage for r in records) / len(records)),
            "std_advantage": 0.0,
            "fraction_pos_adv": float(sum(r.advantage > 0 for r in records) / len(records)),
            "policy_loss": float("nan"),
            "num_trajectories": len(records),
            "mean_traj_length": float(sum(r.length for r in records) / len(records)),
            "replay_mode": "batched" if use_batched else "sequential",
            "invalid_dropped": nonfinite_dropped,
            "none_dropped": none_dropped,
            "replay_failed": int(sum(r.replay_failed for r in records)),
            "amp_enabled": bool(getattr(config, "use_amp", False)),
            "amp_dtype": getattr(config, "amp_dtype", "none")
        }
        return metrics

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        clip_val = config.optimizer.get("gradient_clipping", 0)
        if clip_val and clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        clip_val = config.optimizer.get("gradient_clipping", 0)
        if clip_val and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

    rewards = [r.reward for r in records]
    advantages = [r.advantage for r in records]
    mean_reward = sum(rewards) / len(rewards)
    mean_adv = sum(advantages) / len(advantages)
    std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5 if advantages else 0.0

    metrics = {
        "baseline": baseline,
        "mean_reward": float(mean_reward),
        "best_reward": float(max(rewards)),
        "mean_advantage": float(mean_adv),
        "std_advantage": float(std_adv),
        "fraction_pos_adv": float(sum(a > 0 for a in advantages) / len(advantages)),
        "policy_loss": float(loss.item()),
        "num_trajectories": len(records),
        "mean_traj_length": float(sum(r.length for r in records) / len(records)) if records else 0.0,
        "replay_mode": "batched" if use_batched else "sequential",
        "invalid_dropped": nonfinite_dropped,
        "none_dropped": none_dropped,
        "replay_failed": int(sum(r.replay_failed for r in records)),
        "amp_enabled": bool(getattr(config, "use_amp", False)),
        "amp_dtype": getattr(config, "amp_dtype", "none"),
        "adv_norm": bool(normalize_adv)
    }
    if logger:
        logger.info(f"[DR-GRPO] {metrics}")
    return metrics