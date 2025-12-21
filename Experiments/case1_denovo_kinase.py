from config import MoleculeConfig as BaseConfig

class MoleculeConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.objective_type = "kinase_mpo"

        # --- TRAINING (Start from C) ---
        self.start_from_c_chains = True  # <--- ON
        self.use_fragment_library = False  # <--- OFF
        self.fragment_library_path = None  # Ignored

        # --- INFERENCE (Test on Scaffolds) ---
        # Even though we trained on C, we force testing on these scaffolds
        self.evaluation_scaffolds_path = "zinc_splits/run_seed_42/test_scaffolds.txt"

        # --- RL METHOD ---
        self.use_dr_grpo = True
        self.use_grpo_grouping = True  # Standard GRPO

        # --- WandB Logging ---
        self.seed = 42
        self.use_wandb = 'auto'  # Master switch for WandB logging
        self.wandb_project = "graphxform-rl-paper"
        self.wandb_entity = "hasham"  # wandb username or team name
        self.wandb_run_name = f"Case1_DeNovo_{self.objective_type}_Seed{self.seed}"


