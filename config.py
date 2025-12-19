import os
import datetime


class MoleculeConfig:
    def __init__(self):
        # Get Run ID (1, 2, 3) for repeated experiments
        run_id = int(os.environ.get("PMO_RUN_ID", 1))

        # Change seed based on run_id to ensure randomness across the 3 runs
        self.seed = 42 + run_id

        # Network and environment
        self.latent_dimension = 512
        self.num_transformer_blocks = 10
        self.num_heads = 16
        self.dropout = 0.
        self.use_rezero_transformer = True

        # Environment options
        self.wall_clock_limit = None  # in seconds. If no limit, set to None
        self.max_num_atoms = 50

        self.atom_vocabulary = {  # Attention! Order matters!
            "C":    {"allowed": True, "atomic_number": 6, "valence": 4},
            "C-":   {"allowed": True, "atomic_number": 6, "valence": 3, "formal_charge": -1},
            "C+":   {"allowed": True, "atomic_number": 6, "valence": 5, "formal_charge": 1},
            "C@":   {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 1},
            "C@@":  {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 2},

            "N":    {"allowed": True, "atomic_number": 7, "valence": 3},
            "N-":   {"allowed": True, "atomic_number": 7, "valence": 2, "formal_charge": -1},
            "N+":   {"allowed": True, "atomic_number": 7, "valence": 4, "formal_charge": 1},

            "O":    {"allowed": True, "atomic_number": 8, "valence": 2},
            "O-":   {"allowed": True, "atomic_number": 8, "valence": 1, "formal_charge": -1},
            "O+":   {"allowed": True, "atomic_number": 8, "valence": 3, "formal_charge": 1},

            "F":    {"allowed": True, "atomic_number": 9, "valence": 1},

            "P":    {"allowed": True, "atomic_number": 15, "valence": 7},
            "P-":   {"allowed": True, "atomic_number": 15, "valence": 6, "formal_charge": -1},
            "P+":   {"allowed": True, "atomic_number": 15, "valence": 8, "formal_charge": 1},

            "S":    {"allowed": True, "atomic_number": 16, "valence": 6},
            "S-":   {"allowed": True, "atomic_number": 16, "valence": 5, "formal_charge": -1},
            "S+":   {"allowed": True, "atomic_number": 16, "valence": 7, "formal_charge": 1},
            "S@":   {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 1},
            "S@@":  {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 2},

            "Cl": {"allowed": True, "atomic_number": 17, "valence": 1},
            "Br": {"allowed": True, "atomic_number": 35, "valence": 1},
            "I": {"allowed": True, "atomic_number": 53, "valence": 1}
        }

        # self.atom_vocabulary = {  # Attention! Order matters!
        #     "C": {"allowed": True, "atomic_number": 6, "valence": 4},
        #     "N": {"allowed": True, "atomic_number": 7, "valence": 3},
        #     "O": {"allowed": True, "atomic_number": 8, "valence": 2}
        # }

        self.start_from_c_chains = True
        self.start_c_chain_max_len = 1
        self.start_from_smiles = None  # Give SMILES and set `start_from_c_chains=False`.
        self.repeat_start_instances = 1
        # Positive value x, where the actual objective with our molecule score will be set to obj = score - x * SA_score
        self.synthetic_accessibility_in_objective_scale = 0
        # Enforce structural constraints (see molecule evaluator)
        self.include_structural_constraints = False

        # Objective molecule predictor
        self.GHGNN_model_path = os.path.join("objective_predictor/GH_GNN_IDAC/models/GHGNN.pth")
        self.GHGNN_hidden_dim = 113
        # --- PMO / TDC Configuration ---
        # Common PMO Tasks:
        # 'drd2', 'gsk3b', 'jnk3', 'qed'
        # 'zaleplon_mpo', 'albuterol_similarity', 'perindopril_mpo', 'sitagliptin_mpo'
        # 'deco_hop', 'scaffold_hop'
        self.objective_type = os.environ.get("PMO_OBJECTIVE", "albuterol_similarity")
        # self.objective_type = "isomers_c7h8n2o2"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        # self.objective_type = "celecoxib_rediscovery"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        # self.objective_type = "median_tadalafil_sildenafil"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        # self.objective_type = "zaleplon_mpo"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        # self.objective_type = "ranolazine_mpo"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        self.num_predictor_workers = 1  # num of parallel workers that operate on a given list of molecules
        # self.num_predictor_workers = 10  # num of parallel workers that operate on a given list of molecules

        target_gpu_id = os.environ.get("TARGET_GPU_ID", "0")

        self.objective_predictor_batch_size = 64
        self.objective_gnn_device = "cpu"  # device on which the GNN should live

        # Loading trained checkpoints to resume training or evaluate
        self.load_checkpoint_from_path = "model/weights.pt"  # If given, model checkpoint is loaded from this path.
        # self.load_checkpoint_from_path = "model/model_z_chembl.pt"  # If given, model checkpoint is loaded from this path.
        # self.load_checkpoint_from_path = None  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = False  # If True, the optimizer state is also loaded.

        # Training
        self.num_dataloader_workers = 1  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = target_gpu_id  # Must be set, as ray can have problems detecting multiple GPUs
        # self.CUDA_VISIBLE_DEVICES = "0"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cuda:" + target_gpu_id  # Device on which to perform the supervised training
        self.num_epochs = 3000  # Number of epochs (i.e., passes through training set) to train
        self.scale_factor_level_one = 1.
        self.scale_factor_level_two = 1.
        self.batch_size_training = 64
        self.num_batches_per_epoch = None  # Can be None, then we just do one pass through generated dataset
        # self.num_batches_per_epoch = 20  # Can be None, then we just do one pass through generated dataset

        # Optimizer
        self.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }
        # self.optimizer = {
        #     "lr": 1e-4,  # learning rate
        #     "weight_decay": 0,
        #     "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
        #     "schedule": {
        #         "decay_lr_every_epochs": 1,
        #         "decay_factor": 1
        #     }
        # }

        # Self-improvement sequence decoding
        self.gumbeldore_config = {
            # Number of trajectories with the highest objective function evaluation to keep for training
            "num_trajectories_to_keep": 5,
            "keep_intermediate_trajectories": False,  # if True, we consider all intermediate, terminable trajectories
            "devices_for_workers": ["cuda:"+target_gpu_id] * 1,
            # "devices_for_workers": ["cuda:0", "cuda:1"],
            "destination_path": "./data/generated_molecules.pickle",
            # "destination_path": None,
            "batch_size_per_worker": 1,  # Keep at one, as we only have three atoms from which we can start
            "batch_size_per_cpu_worker": 1,

            "search_type": "wor",  # "beam_search" | "tasar" | "iid_mc", "wor"
            # "search_type": "tasar",
            "num_samples_per_instance": 128,  # For 'iid_mc': number of IID samples to generate per starting instance
            "sampling_temperature": 1,  # For 'iid_mc': temperature for sampling. >1 is more random.

            "beam_width": 128,
            "replan_steps": 12,
            # "num_rounds": 10,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "num_rounds": 1,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # Only use for gumbeldore_eval=True below, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False,

            # "max_leaves_per_root": 250,  # Max number of leaves to expand per root node in TASAR. 0 = no limit.
            # "leaf_sampling_mode": "stratified",  # "random" | "stratified" | "topk"
            # "stratified_quantiles": (0.10, 0.90),  # (low_q, high_q)
            # "stratified_target_fracs": (0.25, 0.50, 0.25),  # (top, mid, bottom)
            # "stratified_target_fracs": (0.25, 0.50, 0.25),  # (top, mid, bottom)
            # "stratified_allow_shortfall_fill": True
        }

        # Results and logging
        # self.results_path = os.path.join("./results",
        #                                  datetime.datetime.now().strftime(
        #                                      "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights

        self.log_to_file = True

        # --- Dr. GRPO / RL fine-tuning baseline configuration ---

        self.use_dr_grpo = True  # Enable RL fine-tuning (vs pure supervised)

        self.max_size_hof = 1  # Maximum size of the Hall of Fame (HoF)
        self.similarity_threshold_hof = 0.75  # Tanimoto similarity threshold for HoF updates
        self.elite_prompt_prob = 0  # Probability of selecting an elite scaffold from HoF as prompt

        self.use_fragment_library = False  # Master switch for GRPO prompting
        self.fragment_library_path = "data/GDB17.50000000LL.noSR_filtered.txt"
        # self.fragment_library_path = "data/GDB13_Subset_ABCDEFG_filtered.txt"
        # Number of prompts (scaffolds) to sample per epoch
        self.num_prompts_per_epoch = 5

        # K: Number of completions per prompt is already set by:
        # self.gumbeldore_config["num_samples_per_instance"] = ... (for iid_mc)
        # or
        # self.gumbeldore_config["beam_width"] = ... (for wor/tasar)

        self.ppo_epochs = 1  # Number of GRPO iterations per RL update, for now keep 1 for simplicity (REINFORCE with baseline)

        self.rl_ppo_clip_epsilon = 0.2  # PPO clipping parameter

        # self.rl_entropy_beta = 0.0
        # self.rl_entropy_beta = 0.0015
        # self.rl_entropy_beta = 0.001
        # self.rl_entropy_beta = 0.
        # self.rl_entropy_beta = 0.001
        self.rl_entropy_beta = float(os.environ.get("RL_ENTROPY_BETA", "0.0"))

        entropy_str = str(self.rl_entropy_beta).replace(".", "p")

        # Create a unique folder name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.results_path = os.path.join(
            "./results",
            f"{self.objective_type}_ent_{entropy_str}_runID_{run_id}_{timestamp}"
        )

        # --- WandB Logging ---
        self.use_wandb = True  # Master switch for WandB logging
        self.wandb_project = "graphxform-rl-battery-chembl"
        self.wandb_entity = "mbinjavaid-rwth-aachen-university"  # wandb username or team name
        self.wandb_run_name = f"{self.objective_type}_wor_ent_{entropy_str}_runID_{run_id}"

        self.rl_use_novelty_bonus = False  # Master switch to enable/disable novelty
        self.rl_novelty_beta = 0.05  # The coefficient for the novelty bonus

        self.rl_use_il_distillation = False

        # Core RL control
        self.rl_replay_microbatch_size = 32  # Streaming microbatch size (0/None => process all trajectories together)
        # self.rl_replay_microbatch_size = 64  # Streaming microbatch size (0/None => process all trajectories together)

        self.rl_streaming_backward = True  # Use streaming backward pass (vs batched; requires microbatching)

        # Advantage / baseline
        self.rl_advantage_normalize = False  # (Optional) Normalize trajectory advantages; leave False for Dr. GRPO

        # self.rl_global_normalize = False # Globally normalize advantage by dividing by std

        # Trajectory / logging options
        self.rl_store_trajectories_path = None  # Optional: path to pickle recent trajectories
        self.rl_max_group_size = None  # Optional cap on trajectories per update
        self.rl_log_advantages = False  # If True, add detailed advantage stats to logs

        # Structural / safety
        self.rl_assert_masks = False  # Enable strict feasibility & finite log_prob assertions
        self.freeze_all_except_final_layer = False  # If True, only final layer is trainable

        # Mixed precision
        self.use_amp = True
        self.amp_dtype = "bf16"  # "bf16" preferred on modern NVIDIA GPUs; "fp16" if needed
        self.use_amp_inference = True  # Also use autocast during rollout generation