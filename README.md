```markdown

## Installation

1. Install the dependencies found in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To reproduce the results and tasks described in the paper, we have provided separate configuration files in the `Experiments/` directory. Each task has its own `.py` configuration file.

You can run a specific experiment by pointing `main.py` to the desired configuration module using the `--config` argument.

### Running an Experiment

Run the training script as follows:

```bash
python main.py --config Experiments.<config_name>
```

**Example:** To run the `case3_prodrug_bbb` task:

```bash
python main.py --config Experiments.case3_prodrug_bbb
```

## Output

Results, model checkpoints, and logs will be saved to the `results/` directory by default (unless specified otherwise in the specific config).
