# GRXForm (PMO benchmark)

## Installation

Install all requirements from `requirements.txt`:

`pip install -r requirements.txt`

Additionally, install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with options corresponding to your hardware.

Create the required directories:

`mkdir -p results data/chembl/pretrain_sequences`

## Usage

### Running the PMO Benchmark

Simply run the benchmark script:

`python run_benchmark.py`

This will automatically run all objectives with their configured entropy values, executing each experiment 3 times.

## Output

Results, model checkpoints, and logs will be saved to the `results/` directory.