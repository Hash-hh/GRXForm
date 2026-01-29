from config import MoleculeConfig as BaseConfig
import platform


class MoleculeConfig(BaseConfig):
    """
    Scenario A: Case 1 (De Novo Training)
    Train on "C", ignore scaffolds. Test on Scaffolds to see if it generalizes.
    """

    def __init__(self):
        super().__init__()

        self.objective_type = "prodrug_bbb"

        self.prodrug_mode = 'prodrug' in self.objective_type if hasattr(self, 'objective_type') else False
        self.prodrug_parents_train = [
            "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",  # Morphine
            "C(CC(=O)O)CN",  # GABA
            "C1CNCCC1C(=O)O",  # Nipecotic Acid
            "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        ]


        # Testing: Hold out Dopamine and Naltrexone
        self.prodrug_parents_test = [
            "C1=CC(=C(C=C1CCN)O)O",  # Dopamine
            "C1CC1CN2CC[C@]34[C@@H]5C(=O)CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O"  # Naltrexone
        ]

        # BBB Objective Weights
        self.bbb_weight_logp = 1.0
        self.bbb_weight_hdonor = 1.0
        self.bbb_weight_cleavable = 2.0
        self.bbb_weight_qed = 2.0  # Push for drug-like molecules
        self.bbb_weight_mw_penalty = 5.0  # Strong penalty for going over size
        self.bbb_max_mw = 600.0  # Max Daltons

        self.prodrug_parent_smiles = None  # Will be set during training
        self.prodrug_log_components = True  # Log individual components of prodrug objective

        self.include_carbon_prompt = False


        # --- TRAINING (Start from C) ---
        self.start_from_c_chains = False  # <--- ON
        self.use_fragment_library = True  # <--- OFF
        self.max_oracle_calls = 50000

        # --- INFERENCE (Test on Scaffolds) ---
        # seed = 42
        # self.fragment_library_path = f"scaffold_splitting/zinc_splits_optimized/run_seed_{seed}/train_scaffolds.txt"
        self.evaluation_scaffolds_path = None  # test scaffolds
        self.validation_scaffolds_path = None

        # --- RL METHOD ---
        self.use_dr_grpo = True
        self.use_grpo_grouping = False  # Standard GRPO

        # sampling parameters
        self.gumbeldore_config["beam_width"] = 32
        # self.num_prompts_per_epoch = 10
        self.fixed_test_beam_width = 32
        self.num_epochs = 1000


        # --- WandB Logging ---
        self.use_wandb = 'auto'  # Master switch for WandB logging
        self.wandb_project = "graphxform-rl-paper-v2"
        self.wandb_entity = ""  # wandb username or team name
        self.wandb_run_name = f"Case3_Scaffold_{self.objective_type}"

        # Resolve "auto" setting based on OS
        if self.use_wandb == "auto":
            self.use_wandb = platform.system() == "Linux"


