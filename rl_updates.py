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
                                    normalize: bool = False,
                                    config=None) -> float:
    """
    Computes advantages. If rl_use_ema_baseline enabled, uses the EMA baseline
    value PRIOR to updating it, then updates EMA for next call.
    Returns the baseline that was actually subtracted (for logging).
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


# --------- (OLD fallback replay functions) ---------
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


# --------- STREAMING BACKWARD (with entropy + instrumentation FIXED) ---------
def streaming_replay_and_backward(model: MoleculeTransformer,
                                  optimizer: torch.optim.Optimizer,
                                  records: List[TrajectoryRecord],
                                  config,
                                  device: torch.device,
                                  autocast_ctx,
                                  scaler):
    """
    Streaming (per-step) backward with microbatching.
    Entropy regularization (only over feasible actions to avoid NaNs):
      Loss = -(1/N) Σ_i A_i Σ_t log π(a_{i,t}|s_{i,t}) - (β / N) Σ_{i,t} H_feasible(s_{i,t})
    where H_feasible excludes masked actions.
    """
    model.eval()
    assert all(r.advantage is not None for r in records), "Compute advantages first."
    N = len(records)
    if N == 0:
        return 0.0

    for rec in records:
        rec.log_prob_sum = torch.tensor(0.0, device=device)
        rec.length = 0

    optimizer.zero_grad(set_to_none=True)

    micro = getattr(config, "rl_replay_microbatch_size", 0) or N
    assert micro > 0, "Microbatch size must be > 0"

    entropy_coef = float(getattr(config, "rl_entropy_coef", 0.0) or 0.0)
    debug_entropy = bool(getattr(config, "rl_debug_entropy", False))
    debug_print_first = int(getattr(config, "rl_debug_entropy_print_first", 10))

    # Policy / entropy accumulators
    sum_adv_logp = 0.0
    entropy_total = 0.0
    entropy_step_count = 0  # counts states where entropy computed (feasible_count >=1)

    # Per-level entropy sums & counts
    entropy_level_sums = {0: 0.0, 1: 0.0, 2: 0.0}
    entropy_level_counts = {0: 0, 1: 0, 2: 0}

    # Per-level feasible action stats
    steps_level = {0: 0, 1: 0, 2: 0}
    singleton_steps_level = {0: 0, 1: 0, 2: 0}
    feasible_sum_level = {0: 0, 1: 0, 2: 0}

    # Probability sharpness stats (only for steps with >=2 feasible)
    max_prob_sum_level = {0: 0.0, 1: 0.0, 2: 0.0}
    top2_gap_sum_level = {0: 0.0, 1: 0.0, 2: 0.0}
    multi_action_steps_level = {0: 0, 1: 0, 2: 0}

    debug_lines_printed = 0

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

            loss_batch = torch.zeros((), device=device, dtype=head0.dtype)
            step_entropy_sum = 0.0
            step_entropy_states = 0  # number of feasible states contributing (for gating)
            level_entropy_this_step = []

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

                # Build mask tensor to inspect feasible count
                mask_list = clone.current_action_mask
                if isinstance(mask_list, torch.Tensor):
                    infeasible_mask = mask_list.to(device=device, dtype=torch.bool)
                else:
                    infeasible_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)

                feasible_mask = ~infeasible_mask
                feasible_count = feasible_mask.sum().item()
                steps_level[lvl] += 1
                feasible_sum_level[lvl] += feasible_count
                if feasible_count == 1:
                    singleton_steps_level[lvl] += 1

                log_probs = _prepare_masked_log_probs(
                    level_logits, infeasible_mask, device,
                    assert_masks=getattr(config, "rl_assert_masks", False)
                )

                # Compute entropy ONLY over feasible actions to avoid 0 * -inf = NaN
                if entropy_coef > 0.0 and feasible_count > 0:
                    log_probs_feas = log_probs[feasible_mask]
                    probs_feas = log_probs_feas.exp()
                    ent = -(probs_feas * log_probs_feas).sum()
                    if torch.isfinite(ent):
                        ent_val = float(ent.detach().cpu())
                        step_entropy_sum += ent_val
                        step_entropy_states += 1
                        level_entropy_this_step.append((lvl, ent_val))
                    else:
                        if debug_entropy and debug_lines_printed < debug_print_first:
                            print(f"[ENTROPY-DBG] NaN after masking: lvl={lvl} feasible_count={feasible_count}")
                else:
                    # still need probs for sharpness stats
                    probs_feas = log_probs[feasible_mask].exp()

                # Sharpness stats (only if >=2 feasible)
                if feasible_count >= 2:
                    sorted_probs, _ = torch.sort(probs_feas, descending=True)
                    maxp = float(sorted_probs[0].detach().cpu())
                    second = float(sorted_probs[1].detach().cpu())
                    max_prob_sum_level[lvl] += maxp
                    top2_gap_sum_level[lvl] += (maxp - second)
                    multi_action_steps_level[lvl] += 1

                if debug_entropy and debug_lines_printed < debug_print_first:
                    # For debug display approximate entropy (safe)
                    if feasible_count > 0:
                        with torch.no_grad():
                            ent_dbg = -(probs_feas * log_probs[feasible_mask]).sum().item()
                    else:
                        ent_dbg = 0.0
                    print(f"[ENTROPY-DBG] lvl={lvl} feas={feasible_count} "
                          f"maxp={(probs_feas.max().item() if feasible_count>0 else 0.0):.4f} ent={ent_dbg:.4f}")
                    debug_lines_printed += 1

                action = rec.history[cursor]
                if action >= log_probs.shape[0]:
                    raise RuntimeError(f"[StreamingReplay] Action {action} out of bounds {log_probs.shape[0]}")
                chosen_logp = log_probs[action]

                rec.log_prob_sum = rec.log_prob_sum + chosen_logp.detach().float()
                rec.length += 1
                sum_adv_logp += rec.advantage * float(chosen_logp.detach().cpu())

                contrib = -(rec.advantage / N) * chosen_logp
                loss_batch = loss_batch + contrib

                clone.take_action(action)
                cursors[local_idx] += 1
                if clone.synthesis_done or cursors[local_idx] >= len(rec.history):
                    finished[local_idx] = True

            # Apply aggregated entropy for this forward step
            if entropy_coef > 0.0 and step_entropy_states > 0 and math.isfinite(step_entropy_sum):
                entropy_loss = torch.tensor((-entropy_coef / N) * step_entropy_sum,
                                            device=device, dtype=loss_batch.dtype)
                loss_batch = loss_batch + entropy_loss
                entropy_total += step_entropy_sum
                entropy_step_count += step_entropy_states
                for lvl, ent_val in level_entropy_this_step:
                    entropy_level_sums[lvl] += ent_val
                    entropy_level_counts[lvl] += 1

            if scaler is not None:
                scaler.scale(loss_batch).backward()
            else:
                loss_batch.backward()

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

    policy_loss_value = -(1.0 / N) * sum_adv_logp

    # Store entropy means
    if entropy_step_count > 0:
        config._last_entropy_mean = entropy_total / entropy_step_count
    else:
        config._last_entropy_mean = 0.0
    for lvl in (0, 1, 2):
        if entropy_level_counts[lvl] > 0:
            setattr(config, f"_last_entropy_mean_level{lvl}",
                    entropy_level_sums[lvl] / entropy_level_counts[lvl])
        else:
            setattr(config, f"_last_entropy_mean_level{lvl}", 0.0)

    # Persist debug stats
    config._last_steps_level = steps_level
    config._last_singleton_steps_level = singleton_steps_level
    config._last_feasible_sum_level = feasible_sum_level
    config._last_multi_action_steps_level = multi_action_steps_level
    config._last_max_prob_sum_level = max_prob_sum_level
    config._last_top2_gap_sum_level = top2_gap_sum_level

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
    baseline = compute_baseline_and_advantages(records, normalize=normalize_adv, config=config)

    use_streaming = getattr(config, "rl_streaming_backward", False)

    if use_streaming:
        autocast_ctx, scaler = _make_autocast_ctx(config)
        policy_loss_val = streaming_replay_and_backward(
            model, optimizer, records, config, device, autocast_ctx, scaler
        )
        replay_mode = "streaming"
    else:
        # Entropy instrumentation only implemented for streaming path
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

    entropy_coef = float(getattr(config, "rl_entropy_coef", 0.0))
    mean_step_entropy = float(getattr(config, "_last_entropy_mean", 0.0))
    mean_entropy_level0 = float(getattr(config, "_last_entropy_mean_level0", 0.0))
    mean_entropy_level1 = float(getattr(config, "_last_entropy_mean_level1", 0.0))
    mean_entropy_level2 = float(getattr(config, "_last_entropy_mean_level2", 0.0))

    # Improvement-triggered entropy decay
    if entropy_coef > 0.0:
        best_reward = max(rewards)
        if not hasattr(config, "_entropy_base_best_reward"):
            config._entropy_base_best_reward = best_reward
            config._entropy_decays_done = 0
        delta_target = float(getattr(config, "rl_entropy_improvement_delta", 0.02))
        decay_factor = float(getattr(config, "rl_entropy_decay_factor", 0.7))
        min_coef = float(getattr(config, "rl_entropy_min_coef", 0.001))
        decays_done = int(getattr(config, "_entropy_decays_done", 0))
        improvement = best_reward - config._entropy_base_best_reward
        needed = (decays_done + 1) * delta_target
        if improvement >= needed and entropy_coef > min_coef:
            new_coef = max(min_coef, entropy_coef * decay_factor)
            setattr(config, "rl_entropy_coef", new_coef)
            config._entropy_decays_done = decays_done + 1
            config._entropy_last_decay_at_reward = best_reward
        else:
            new_coef = entropy_coef
    else:
        new_coef = entropy_coef

    # Extract debug stats
    steps_level_dbg = getattr(config, "_last_steps_level", {0: 0, 1: 0, 2: 0})
    singleton_steps_level_dbg = getattr(config, "_last_singleton_steps_level", {0: 0, 1: 0, 2: 0})
    feasible_sum_level_dbg = getattr(config, "_last_feasible_sum_level", {0: 0, 1: 0, 2: 0})
    multi_action_steps_level_dbg = getattr(config, "_last_multi_action_steps_level", {0: 0, 1: 0, 2: 0})
    max_prob_sum_level_dbg = getattr(config, "_last_max_prob_sum_level", {0: 0.0, 1: 0.0, 2: 0.0})
    top2_gap_sum_level_dbg = getattr(config, "_last_top2_gap_sum_level", {0: 0.0, 1: 0.0, 2: 0.0})

    def safe_div(a, b):
        return a / b if b else 0.0

    metrics = {
        "baseline": baseline,
        "batch_mean_reward": float(getattr(config, "_last_batch_mean_reward", baseline)),
        "ema_baseline_active": bool(getattr(config, "rl_use_ema_baseline", False)),
        "mean_reward": float(mean_reward),
        "best_reward": float(max(rewards)),
        "mean_advantage": float(mean_adv),
        "std_advantage": float(std_adv),
        "fraction_pos_adv": float(sum(a > 0 for a in advantages) / len(advantages)),
        "policy_loss": float(policy_loss_val),
        "num_trajectories": len(records),
        "mean_traj_length": float(sum(r.length for r in records) / len(records)) if records else 0.0,
        "replay_mode": replay_mode,
        "invalid_dropped": int(sum(not math.isfinite(r.reward) for r in records)),
        "none_dropped": 0,
        "amp_enabled": bool(getattr(config, "use_amp", False)),
        "amp_dtype": getattr(config, "amp_dtype", "none"),
        "adv_norm": bool(normalize_adv),
        "entropy_coef": float(entropy_coef),
        "entropy_coef_next": float(new_coef),
        "mean_step_entropy": mean_step_entropy,
        "mean_entropy_level0": mean_entropy_level0,
        "mean_entropy_level1": mean_entropy_level1,
        "mean_entropy_level2": mean_entropy_level2,
        "entropy_decays_done": int(getattr(config, "_entropy_decays_done", 0)),
        # Per-level visit & feasibility stats
        "steps_level0": steps_level_dbg.get(0, 0),
        "steps_level1": steps_level_dbg.get(1, 0),
        "steps_level2": steps_level_dbg.get(2, 0),
        "singleton_steps_level0": singleton_steps_level_dbg.get(0, 0),
        "singleton_steps_level1": singleton_steps_level_dbg.get(1, 0),
        "singleton_steps_level2": singleton_steps_level_dbg.get(2, 0),
        "avg_feasible_actions_level0": safe_div(feasible_sum_level_dbg.get(0, 0), steps_level_dbg.get(0, 0)),
        "avg_feasible_actions_level1": safe_div(feasible_sum_level_dbg.get(1, 0), steps_level_dbg.get(1, 0)),
        "avg_feasible_actions_level2": safe_div(feasible_sum_level_dbg.get(2, 0), steps_level_dbg.get(2, 0)),
        "multi_action_steps_level0": multi_action_steps_level_dbg.get(0, 0),
        "multi_action_steps_level1": multi_action_steps_level_dbg.get(1, 0),
        "multi_action_steps_level2": multi_action_steps_level_dbg.get(2, 0),
        "mean_max_prob_level0": safe_div(max_prob_sum_level_dbg.get(0, 0.0), multi_action_steps_level_dbg.get(0, 0)),
        "mean_max_prob_level1": safe_div(max_prob_sum_level_dbg.get(1, 0.0), multi_action_steps_level_dbg.get(1, 0)),
        "mean_max_prob_level2": safe_div(max_prob_sum_level_dbg.get(2, 0.0), multi_action_steps_level_dbg.get(2, 0)),
        "mean_top2_gap_level0": safe_div(top2_gap_sum_level_dbg.get(0, 0.0), multi_action_steps_level_dbg.get(0, 0)),
        "mean_top2_gap_level1": safe_div(top2_gap_sum_level_dbg.get(1, 0.0), multi_action_steps_level_dbg.get(1, 0)),
        "mean_top2_gap_level2": safe_div(top2_gap_sum_level_dbg.get(2, 0.0), multi_action_steps_level_dbg.get(2, 0)),
    }
    if logger:
        logger.info(f"[DR-GRPO] {metrics}")
    return metrics