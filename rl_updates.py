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
    replay_failed: bool = False  # (not used in streaming path unless you extend soft-fail)


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
        assert level_logits.shape[0] >= infeasible_mask.shape[0], \
            f"Logits shorter ({level_logits.shape[0]}) than mask ({infeasible_mask.shape[0]})"
    level_logits = level_logits[:infeasible_mask.shape[0]]
    feasible_any = (~infeasible_mask).any().item()
    if assert_masks:
        assert feasible_any, "All actions masked; invalid environment state."
    if not feasible_any:
        level_logits = torch.zeros_like(level_logits)
    level_logits = level_logits.masked_fill(infeasible_mask, float("-inf"))
    return torch.log_softmax(level_logits, dim=-1)


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


# --------- (OLD non-streaming replay functions retained for fallback) ---------
def sequential_replay_and_compute_log_probs(
        model: MoleculeTransformer,
        records: List[TrajectoryRecord],
        device: torch.device,
        autocast_ctx,
        assert_masks: bool,
        debug_verify: bool = False
):
    model.eval()
    for rec in records:
        design_clone = _fresh_initial_clone(rec.design)
        rec.log_prob_sum = torch.tensor(0.0, device=device)
        steps = 0
        for action in rec.history:
            if design_clone.synthesis_done:
                break
            with autocast_ctx:
                batch = MoleculeDesign.list_to_batch([design_clone], device=device)
                head0, head1, head2 = model(batch)
                current_level = design_clone.current_action_level
                if current_level == 0:
                    level_logits = head0[0]
                elif current_level == 1:
                    level_logits = head1[0]
                else:
                    level_logits = head2[0]
                log_probs = _prepare_masked_log_probs(level_logits, design_clone.current_action_mask, device,
                                                      assert_masks=assert_masks)
            if action >= log_probs.shape[0]:
                raise RuntimeError(f"[Replay] Action {action} out of bounds {log_probs.shape[0]}")
            rec.log_prob_sum = rec.log_prob_sum + log_probs[action].detach().float()
            steps += 1
            design_clone.take_action(action)
            if design_clone.synthesis_done:
                break
        rec.length = steps


def batched_replay_and_compute_log_probs(
        model: MoleculeTransformer,
        records: List[TrajectoryRecord],
        device: torch.device,
        autocast_ctx,
        assert_masks: bool,
        debug_verify: bool = False
):
    model.eval()
    n = len(records)
    if n == 0:
        return
    clones = [_fresh_initial_clone(r.design) for r in records]
    cursors = [0] * n
    lengths = [0] * n
    log_sums = [torch.tensor(0.0, device=device) for _ in records]

    active = [i for i in range(n)]
    while active:
        batch_clones = [clones[i] for i in active]
        with autocast_ctx:
            head0, head1, head2 = model(MoleculeDesign.list_to_batch(batch_clones, device=device))
        remove = []
        for local_pos, rec_idx in enumerate(active):
            rec = records[rec_idx]
            clone = clones[rec_idx]
            cursor = cursors[rec_idx]
            if cursor >= len(rec.history) or clone.synthesis_done:
                remove.append(rec_idx)
                continue
            lvl = clone.current_action_level
            if lvl == 0:
                level_logits = head0[local_pos]
            elif lvl == 1:
                level_logits = head1[local_pos]
            else:
                level_logits = head2[local_pos]
            log_probs = _prepare_masked_log_probs(level_logits, clone.current_action_mask, device,
                                                  assert_masks=assert_masks)
            action = rec.history[cursor]
            if action >= log_probs.shape[0]:
                raise RuntimeError(f"[Replay] Action {action} out of bounds {log_probs.shape[0]}")
            log_sums[rec_idx] = log_sums[rec_idx] + log_probs[action].detach().float()
            lengths[rec_idx] += 1
            cursors[rec_idx] += 1
            clone.take_action(action)
            if clone.synthesis_done or cursors[rec_idx] >= len(rec.history):
                remove.append(rec_idx)
        if remove:
            done = set(remove)
            active = [i for i in active if i not in done]

    for i, rec in enumerate(records):
        rec.log_prob_sum = log_sums[i]
        rec.length = lengths[i]


# --------- STREAMING BACKWARD (FIXED) ---------
def streaming_replay_and_backward(model: MoleculeTransformer,
                                  optimizer: torch.optim.Optimizer,
                                  records: List[TrajectoryRecord],
                                  config,
                                  device: torch.device,
                                  autocast_ctx,
                                  scaler):
    """
    Streaming (per-step) backward with microbatching.
    For each replay step (single forward over current active subset) we:
      - accumulate a scalar loss_batch across active trajectories
      - call backward exactly once on that scalar
    This avoids reusing a freed graph while keeping activation memory low.
    """
    model.eval()  # deterministic
    assert all(r.advantage is not None for r in records), "Compute advantages first."
    N = len(records)
    if N == 0:
        return 0.0

    # Init metrics fields
    for rec in records:
        rec.log_prob_sum = torch.tensor(0.0, device=device)
        rec.length = 0

    optimizer.zero_grad(set_to_none=True)

    micro = getattr(config, "rl_replay_microbatch_size", 0) or N
    assert micro > 0, "Microbatch size must be > 0"

    # For logging: accumulate Σ_i A_i Σ_t logp
    sum_adv_logp = 0.0

    # Slice into microbatches of trajectories
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

            # Accumulate per-step loss across active trajectories, then backward once.
            if scaler is not None:
                # Use FP32 accumulator outside scaled region? Simpler to accumulate inside.
                loss_batch = 0.0
            else:
                loss_batch = torch.tensor(0.0, device=device, dtype=head0.dtype)

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

                log_probs = _prepare_masked_log_probs(
                    level_logits, clone.current_action_mask, device,
                    assert_masks=getattr(config, "rl_assert_masks", False)
                )
                action = rec.history[cursor]
                if action >= log_probs.shape[0]:
                    raise RuntimeError(f"[StreamingReplay] Action {action} out of bounds {log_probs.shape[0]}")

                chosen_logp = log_probs[action]
                # Metrics accumulation
                rec.log_prob_sum = rec.log_prob_sum + chosen_logp.detach().float()
                rec.length += 1
                sum_adv_logp += rec.advantage * float(chosen_logp.detach().cpu())

                # Form loss contribution (tensor)
                # Loss overall: -(1/N) * Σ_i A_i Σ_t logp
                contrib = -(rec.advantage / N) * chosen_logp
                if scaler is not None:
                    # Accumulate as Python float; we will wrap in a tensor at backward time
                    loss_batch += float(contrib.detach().cpu())
                else:
                    loss_batch = loss_batch + contrib

                # Advance environment
                clone.take_action(action)
                cursors[local_idx] += 1
                if clone.synthesis_done or cursors[local_idx] >= len(rec.history):
                    finished[local_idx] = True

            # Single backward for this step
            if scaler is not None:
                loss_batch_tensor = torch.tensor(loss_batch, device=device, dtype=head0.dtype, requires_grad=True)
                scaler.scale(loss_batch_tensor).backward()
            else:
                loss_batch.backward()

    # Finish optimizer step
    clip_val = config.optimizer.get("gradient_clipping", 0)
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

    # Compute logged policy loss
    policy_loss_value = -(1.0 / N) * sum_adv_logp
    return policy_loss_value


def compute_policy_loss_for_logging(records: List[TrajectoryRecord]) -> float:
    if not records:
        return 0.0
    with torch.no_grad():
        log_probs = torch.stack([r.log_prob_sum for r in records])
        adv = torch.tensor([r.advantage for r in records], device=log_probs.device, dtype=log_probs.dtype)
        return float(-(adv * log_probs).mean().item())


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
    baseline = compute_baseline_and_advantages(records, normalize=normalize_adv)

    use_streaming = getattr(config, "rl_streaming_backward", False)

    if use_streaming:
        autocast_ctx, scaler = _make_autocast_ctx(config)
        policy_loss_val = streaming_replay_and_backward(
            model, optimizer, records, config, device, autocast_ctx, scaler
        )
        replay_mode = "streaming"
    else:
        # Fallback: old accumulate-then-backward path
        assert_masks = getattr(config, "rl_assert_masks", False)
        autocast_ctx, scaler = _make_autocast_ctx(config)
        use_batched = getattr(config, "rl_batched_replay", True) and len(records) > 1
        if use_batched:
            batched_replay_and_compute_log_probs(model, records, device, autocast_ctx, assert_masks=assert_masks)
        else:
            sequential_replay_and_compute_log_probs(model, records, device, autocast_ctx, assert_masks=assert_masks)

        log_probs = torch.stack([r.log_prob_sum for r in records])
        advantages = torch.tensor([r.advantage for r in records], dtype=log_probs.dtype, device=log_probs.device)
        loss = -(advantages * log_probs).mean()

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
                "amp_enabled": bool(getattr(config, "use_amp", False)),
                "amp_dtype": getattr(config, "amp_dtype", "none"),
                "adv_norm": bool(normalize_adv)
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

        policy_loss_val = float(loss.item())
        replay_mode = "batched" if use_batched else "sequential"

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
        "policy_loss": float(policy_loss_val),
        "num_trajectories": len(records),
        "mean_traj_length": float(sum(r.length for r in records) / len(records)) if records else 0.0,
        "replay_mode": replay_mode,
        "invalid_dropped": nonfinite_dropped,
        "none_dropped": none_dropped,
        "amp_enabled": bool(getattr(config, "use_amp", False)),
        "amp_dtype": getattr(config, "amp_dtype", "none"),
        "adv_norm": bool(normalize_adv)
    }
    if logger:
        logger.info(f"[DR-GRPO] {metrics}")
    return metrics