# GRXForm

## Installation

Install all requirements from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Additionally, install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with options corresponding to your hardware.

Create the required directories:

```bash
mkdir -p results data/chembl/pretrain_sequences
```

### GuacaMol

For drug design tasks, [guacamol](https://github.com/BenevolentAI/guacamol/tree/master) is required. Fix the scipy import issue by changing `from scipy import histogram` to `from numpy import histogram` in guacamol's `utils.chemistry`. See this [issue](https://github.com/BenevolentAI/guacamol/issues/33).

## Usage

### Running the Benchmark

Simply run the benchmark script:

```bash
python run_benchmark.py
```

This will automatically run all GuacaMol objectives with their configured entropy values, executing each experiment 3 times.

## Output

Results, model checkpoints, and logs will be saved to the `results/` directory.