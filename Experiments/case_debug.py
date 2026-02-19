from config import MoleculeConfig as BaseConfig
import platform


class MoleculeConfig(BaseConfig):
    """
    Scenario A: Case 1 (De Novo Training)
    Train on "C", ignore scaffolds. Test on Scaffolds to see if it generalizes.
    """

    def __init__(self):
        super().__init__()

        self.objective_type = "kinase_mpo"

        # --- TRAINING (Start from C) ---
        self.start_from_c_chains = False  # <--- ON
        self.use_fragment_library = True  # <--- OFF
        # self.fragment_library_path = None  # Ignored
        self.max_oracle_calls = 5

        # --- INFERENCE (Test on Scaffolds) ---
        seed = 42
        self.evaluation_scaffolds_path = f"scaffold_splitting/zinc_splits/run_seed_{seed}/test_scaffolds_small.txt"  # test scaffolds
        self.validation_scaffolds_path = f"scaffold_splitting/zinc_splits/run_seed_{seed}/val_scaffolds_small.txt"

        # --- RL METHOD ---
        self.use_dr_grpo = True
        self.use_grpo_grouping = True  # Standard GRPO
        self.use_validation_for_ckpt = True

        self.freeze_all_except_final_layer = True
        self.num_batches_per_epoch = 20

        # sampling parameters
        self.gumbeldore_config["search_type"] = 'wor'
        self.gumbeldore_config["beam_width"] = 1  #320
        self.num_prompts_per_epoch = 1
        self.fixed_test_beam_width = 2  #32
        self.num_epochs = 5  #500


        # --- WandB Logging ---
        self.use_wandb = 'auto'  # Master switch for WandB logging
        self.wandb_project = "graphxform-rl-paper"
        self.wandb_entity = ""  # wandb username or team name
        self.wandb_run_name = f"Case1_DUBUG_{self.objective_type}_Seed{self.seed}"

        # Resolve "auto" setting based on OS
        if self.use_wandb == "auto":
            self.use_wandb = platform.system() == "Linux"


