import os
import inspect
import json
import wandb

from typing import Optional


class Logger:
    def __init__(self, args, results_path: str, log_to_file: bool, config=None):
        self.results_path = results_path
        self.log_to_file = log_to_file
        self.config = config

        self.file_log_path = os.path.join(self.results_path, "log.txt")
        if self.log_to_file:
            os.makedirs(self.results_path, exist_ok=True)

        # Initialize WandB if enabled
        self.use_wandb = config.use_wandb if config and hasattr(config, 'use_wandb') else False
        if self.use_wandb and config:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=self._extract_config_dict(config),
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
            )
            # Disable model checkpointing to save space
            wandb.run.log_code = False

    def _extract_config_dict(self, config_object):
        """Extract config attributes as a dictionary for WandB"""
        attributes = inspect.getmembers(config_object, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attribute_dict = {}

        for key, value in attributes:
            key = key.replace("+", "_plus").replace("@", "_at")
            if key not in ["devices_for_eval_workers"] and len(str(value)) <= 500:
                if not isinstance(value, dict):
                    attribute_dict[key] = value
        return attribute_dict


    def log_hyperparams(self, config_object):
        attributes = inspect.getmembers(config_object, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attribute_dict = {}

        def add_to_attribute_dict(a):
            for key, value in a:
                key = key.replace("+", "_plus")
                key = key.replace("@", "_at")
                if isinstance(value, dict):
                    add_to_attribute_dict([(f"{key}.{k}", v) for k, v in value.items()])
                else:
                    if key not in ["devices_for_eval_workers"] and len(str(value)) <= 500:
                        attribute_dict[key] = value

        add_to_attribute_dict(attributes)

        if self.log_to_file:
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps({"hyperparameters": attribute_dict}))
                f.write("\n")

    def log_metrics(self, metrics: dict, step: Optional[int] = None, step_desc: Optional[str] = "epoch"):
        if self.log_to_file:
            if step is not None:
                metrics[step_desc] = step
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        # Log to WandB
        if self.use_wandb:
            wandb_metrics = metrics.copy()
            if step is not None:
                wandb.log(wandb_metrics, step=step)
            else:
                wandb.log(wandb_metrics)

    def text_artifact(self, dest_path: str, obj):
        with open(dest_path, "w") as f:
            f.write(str(obj))

    def finish(self):
        """Clean up WandB run"""
        if self.use_wandb:
            wandb.finish()

