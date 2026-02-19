# **GRXForm: Amortized Molecular Optimization via Group Relative Policy Optimization**

GRXForm (Group Relative Xformer) is a framework for amortized molecular optimization. It adapts a pre-trained Graph Transformer model to optimize molecules via sequential atom-and-bond additions. To address the high variance arising from the heterogeneous difficulty of distinct starting structures, GRXForm employs Group Relative Policy Optimization (GRPO) for goal-directed fine-tuning.

This approach normalizes rewards relative to the starting structure, stabilizing the learning process. The resulting policy generalizes to out-of-distribution molecular scaffolds and generates optimized molecules in a single forward pass without requiring inference-time oracle calls or iterative refinement.

## **Installation**

Install the dependencies found in requirements.txt:

pip install \-r requirements.txt

## **Data Preparation**

Before running scaffold-constrained tasks, generate the data splits from the ZINC dataset to evaluate generalization to out-of-distribution scaffolds. The repository includes scripts to handle different clustering and splitting strategies:

\# Standard cluster-based scaffold splits  
python prepare\_zinc\_splits.py

\# Optimized clustering splits  
python smart\_splitting.py

\# Specific subsets and holdout test sets for the Prodrug Transfer task  
python prodrug\_splitting.py

## **Pre-training**

GRXForm utilizes the pre-training framework from the original GraphXForm model. The policy is initialized by pre-training on the ChEMBL database via supervised teacher forcing to impart a prior of chemical plausibility before reinforcement learning fine-tuning.

## **Usage and Configuration**

The framework is controlled via configuration files and command-line arguments. The primary configuration is located in config.py, where you can define the objective task, model hyperparameters, and enable reinforcement learning.

Key variables in config.py:

* objective\_type: Sets the target objective (e.g., "kinase\_mpo", "prodrug\_bbb", "jnk3").  
* use\_dr\_grpo: Set to True to enable GRPO reinforcement learning. Set to False to run supervised fine-tuning.  
* use\_fragment\_library: Master switch for structural conditioning (scaffold-based generation).

### **Running an Experiment**

Run the training script using the default configuration:

python main.py

To run a specific experiment configuration module from an Experiments/ directory, use the \--config argument:

python main.py \--config Experiments.case2\_scaffold\_kinase

You can also override specific RL hyperparameters directly via the command line:

python main.py \--learning\_rate 1e-4 \--rl\_entropy\_beta 0.001 \--ppo\_epochs 1 \--rl\_ppo\_clip\_epsilon 0.2

## **Supported Evaluation Tasks**

The codebase natively supports the objective functions discussed in the paper:

1. **Kinase Scaffold Decoration (Kinase MPO):** Multi-objective optimization targeting GSK3B and JNK3 activity, QED, and Synthetic Accessibility (SA).  
2. **Prodrug Transfer:** Structural transformation targeting blood-brain barrier permeability (optimizing for Delta LogP, HBD masking, cleavability, and QED).  
3. **PMO Benchmark:** Single-objective de-novo generation tasks utilizing the standard PMO/GuacaMol evaluation suite. **Note: To reproduce the PMO benchmark results, please use the grpo\_pmo branch of this repository.**

## **Output**

Results, model checkpoints, and logs are saved to the results/ directory by default. Outputs include:

* best\_model.pt and last\_model.pt weights.  
* Detailed CSV logs generated during validation and testing, containing starting prompts, generated SMILES, objective scores, and constraint satisfaction statuses.  
* Text files logging the top generated molecules per epoch.  
* Weights & Biases logging (if enabled via use\_wandb \= True in the configuration).