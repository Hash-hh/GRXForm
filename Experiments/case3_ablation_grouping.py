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
        self.start_from_c_chains = False
        self.use_fragment_library = True
        self.max_oracle_calls = 50000

        # --- INFERENCE (Test on Scaffolds) ---
        seed = 42
        self.fragment_library_path = f"scaffold_splitting/zinc_splits_optimized/run_seed_{seed}/train_scaffolds.txt"
        self.evaluation_scaffolds_path = f"scaffold_splitting/zinc_splits_optimized/run_seed_{seed}/test_scaffolds.txt"  # test scaffolds
        self.validation_scaffolds_path = f"scaffold_splitting/zinc_splits_optimized/run_seed_{seed}/val_scaffolds.txt"

        # --- RL METHOD ---
        self.use_dr_grpo = True
        self.use_grpo_grouping = False  # Standard GRPO

        # sampling parameters
        self.gumbeldore_config["beam_width"] = 16
        self.num_prompts_per_epoch = 10
        self.fixed_test_beam_width = 1
        self.num_epochs = 500


        # --- WandB Logging ---
        self.use_wandb = 'auto'  # Master switch for WandB logging
        self.wandb_project = "graphxform-rl-paper-v2"
        self.wandb_entity = ""  # wandb username or team name
        self.wandb_run_name = f"Case3_Ablation_{self.objective_type}_Seed{self.seed}"

        # Resolve "auto" setting based on OS
        if self.use_wandb == "auto":
            self.use_wandb = platform.system() == "Linux"


