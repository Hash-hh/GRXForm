"""
Logger implementation with Weights & Biases integration for molecular design experiments.
"""

import os
from typing import Dict, Any, Optional
from config import MoleculeConfig

import wandb


class Logger:
    def __init__(self, args, results_path: str, log_to_file: bool = True, wandb_enable: Optional[bool] = None):
        """
        Initialize logger with wandb integration.

        Args:
            args: Command line arguments
            results_path: Path to save results
            log_to_file: Whether to log to file as well
            wandb_enable: Override wandb setting
        """
        self.results_path = results_path
        self.log_to_file = log_to_file

        # Prioritize explicit parameter, then check args
        if wandb_enable is not None:
            self.wandb_enable = wandb_enable
        elif hasattr(args, 'wandb_enable'):
            self.wandb_enable = args.wandb_enable
        elif hasattr(args, 'config') and hasattr(args.config, 'wandb_enable'):
            self.wandb_enable = args.config.wandb_enable
        elif hasattr(args, 'wandb'):
            self.wandb_enable = args.wandb
        else:
            self.wandb_enable = False

        # If wandb is not installed, force disable
        if wandb is None:
            self.wandb_enable = False

        if self.wandb_enable:
            if wandb is not None:
                wandb.init(
                    project="graphxform-molecular-design",
                    name=f"experiment_{os.path.basename(results_path)}",
                    dir=results_path,
                    config=args.__dict__ if hasattr(args, '__dict__') else {},
                    save_code=True
                )
        if self.log_to_file:
            os.makedirs(results_path, exist_ok=True)
            self.log_file = os.path.join(results_path, "log.txt")
            with open(self.log_file, 'w') as f:
                f.write("Experiment log started\n")

    def log_hyperparams(self, config: MoleculeConfig):
        """
        Log hyperparameters to wandb.

        Args:
            config: MoleculeConfig object containing all hyperparameters
        """
        # Convert config to dictionary
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                # Handle nested dictionaries
                if isinstance(value, dict):
                    for k, v in value.items():
                        config_dict[f"{attr}.{k}"] = v
                else:
                    config_dict[attr] = value

        # Update wandb config
        if self.wandb_enable:
            wandb.config.update(config_dict)

        # Log to file if enabled
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write("\n=== HYPERPARAMETERS ===\n")
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")
                f.write("========================\n\n")

    def log_metrics(self, metrics: Dict[str, Any], step: int, step_desc: str = "train"):
        """
        Log metrics to wandb and optionally to file.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number (epoch)
            step_desc: Description of the step (train, val, test)
        """
        # Prepare metrics for wandb
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb_metrics[f"{step_desc}/{key}"] = value
            elif hasattr(value, '__len__') and len(value) == 1:
                # Handle single-element lists/arrays
                wandb_metrics[f"{step_desc}/{key}"] = value[0] if hasattr(value[0], '__float__') else str(value[0])

        # Log to wandb
        if self.wandb_enable:
            wandb.log(wandb_metrics, step=step)

        # Log to file if enabled
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\n=== METRICS - {step_desc.upper()} - Step {step} ===\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("=" * 50 + "\n")

    def text_artifact(self, filepath: str, content: Any):
        """
        Save text artifact and log to wandb.

        Args:
            filepath: Path to save the text file
            content: Content to save (can be dict, list, or string)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to file
        with open(filepath, 'w') as f:
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"{key}: {value}\n")
            elif isinstance(content, list):
                for item in content:
                    f.write(f"{item}\n")
            else:
                f.write(str(content))

        # Log as wandb artifact
        if self.wandb_enable:
            artifact = wandb.Artifact(
                name=os.path.basename(filepath).replace('.txt', ''),
                type='results'
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)

        # Also log to main log file if enabled
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\n=== ARTIFACT SAVED ===\n")
                f.write(f"File: {filepath}\n")
                f.write("=" * 50 + "\n")

    def log_model_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """
        Log model checkpoint as wandb artifact.

        Args:
            checkpoint_path: Path to the checkpoint file
            is_best: Whether this is the best model so far
        """
        if self.wandb_enable and os.path.exists(checkpoint_path):
            artifact_name = "best_model" if is_best else "model_checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type='model'
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    def finish(self):
        """
        Finish the wandb run.
        """
        if self.wandb_enable:
            wandb.finish()
