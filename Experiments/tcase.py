from config import MoleculeConfig as BaseConfig
import platform


class MoleculeConfig(BaseConfig):
    """
    Scenario: TASAR
    """

    def __init__(self):
        super().__init__()

        self.objective_type = "kinase_mpo"

        # --- TRAINING (Start from C) ---
        self.start_from_c_chains = True  # <--- ON
        self.use_fragment_library = False  # <--- OFF
        self.fragment_library_path = None  # Ignored

        # --- INFERENCE (Test on Scaffolds) ---
        seed = 42
        self.evaluation_scaffolds_path = f"scaffold_splitting/zinc_splits_optimized/run_seed_{seed}/test_scaffolds.txt"  # test scaffolds
        self.validation_scaffolds_path = f"scaffold_splitting/zinc_splits_optimized/run_seed_{seed}/val_scaffolds.txt"

        # --- RL METHOD ---
        self.use_dr_grpo = False
        self.use_grpo_grouping = False  # Standard GRPO
        self.use_validation_for_ckpt = True

        self.freeze_all_except_final_layer = True
        self.num_batches_per_epoch = 20

        # sampling parameters
        self.gumbeldore_config["search_type"] = 'tasar'
        self.gumbeldore_config["beam_width"] = 160
        self.num_prompts_per_epoch = 1
        self.fixed_test_beam_width = 1
        self.num_epochs = 500


        # --- WandB Logging ---
        self.use_wandb = 'auto'  # Master switch for WandB logging
        self.wandb_project = "graphxform-rl-paper-v2"
        self.wandb_entity = ""  # wandb username or team name
        self.wandb_run_name = f"TASAR_{self.objective_type}_Seed{self.seed}"

        # Resolve "auto" setting based on OS
        if self.use_wandb == "auto":
            self.use_wandb = platform.system() == "Linux"


