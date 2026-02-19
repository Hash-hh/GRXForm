# GRXForm (PMO benchmark)

## Installation

Install all requirements from `requirements.txt`:

`pip install -r requirements.txt`

Additionally, install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with options corresponding to your hardware.

Create the required directories:

`mkdir -p results data/chembl/pretrain_sequences`

## Pretraining

You can pretrain youself or use the provided pretrained checkpoint.

### Pretrained checkpoint download

Download the pretrained checkpoint [here](https://syncandshare.lrz.de/dl/fiJs7ZHuCFsVskeoab5aZg/graphxform_pretrained.zip).

### Pretrain yourself

To pretrain with the settings in the paper, do the following: 

1. Download the file `chembl_35_chemreps.txt.gz` from this this link: [Download File](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz)
and put the extracted `chembl_35_chemreps.txt` under the `./data/chembl` directory.
2. Run `$ python filter_molecules.py`. This will perform a train/val split with 100k validation molecules, which will be
put under `data/chembl/chembl_{train/valid}.smiles`. Then, depending on the allowed vocabulary (see `filter_molecules.py`),
it filters the smiles for all strings containing the allowed sets of characters and save them under `data/chembl/chembl_{train/valid}_filtered.smiles`.
3. Run `$ python create_pretrain_dataset.py`. This will convert the filtered SMILES to instances of `MoleculeDesign` (the class in `molecule_design.py`, which takes the role of the molecular graph environment).
The pickled molecules are saved in `./data/chembl/pretrain_sequences/chembl_{train/valid}.pickle`.
4. Run `$ python pretrain.py` to perform pretraining of the model. The general config to use (e.g., architecture) is under `config.py`.
In `pretrain.py`, you can specify pretrain-specific options directly at the entrypoint to adjust to your hardware, i.e.:
```python
if __name__ == '__main__':
    pretrain_train_dataset = "./data/chembl/pretrain_sequences/chembl_train.pickle"
    pretrain_val_dataset = "./data/chembl/pretrain_sequences/chembl_valid.pickle"
    pretrain_num_epochs = 1000   # For how many epochs to train
    batch_size = 512  # Minibatch size
    num_batches_per_epoch = 3000   # Number of minibatches per epoch.
    batch_size_validation = 512  # Batch size during validation
    training_device = "cuda:0"  # Device on which to train. Set to "cpu" if no CUDA available.
    num_dataloader_workers = 10  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None   # Path to checkpoint if training should be continued from existing weights.
```
4. The terminal output will show under which subdirectory (named after timestamp) in `./results` the script will save the model checkpoints.

5. After pretraining, the best model checkpoint (lowest validation loss) will be saved as `best_model.pt` in the corresponding subdirectory in `./results`. You can use this checkpoint for fine-tuning on downstream tasks by specifying the path to the checkpoint in `config.py` (in `load_checkpoint_from_path`).

## Usage
### Running the PMO Benchmark

Simply run the benchmark script:
`python run_benchmark.py`

This will automatically run all objectives with their configured entropy values, executing each experiment 3 times.

## Output

Results, model checkpoints, and logs will be saved to the `results/` directory.