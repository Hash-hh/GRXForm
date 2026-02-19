import os
import subprocess
import time

# Dictionary mapping each objective to its specific entropy value
objective_configs = {
    "albuterol_similarity": 0.002,
    "amlodipine_mpo": 0.0,
    "celecoxib_rediscovery": 0.001,
    "deco_hop": 0.0,
    "drd2": 0.0,
    "fexofenadine_mpo": 0.0,
    "gsk3b": 0.001,
    "isomers_c7h8n2o2": 0.002,
    "isomers_c9h10n2o2pf2cl": 0.0,
    "jnk3": 0.0,
    "median1": 0.001,
    "median2": 0.0,
    "mestranol_similarity": 0.001,
    "osimertinib_mpo": 0.0,
    "perindopril_mpo": 0.001,
    "qed": 0.001,
    "ranolazine_mpo": 0.0,
    "scaffold_hop": 0.0,
    "sitagliptin_mpo": 0.0,
    "thiothixene_rediscovery": 0.001,
    "troglitazone_rediscovery": 0.0,
    "zaleplon_mpo": 0.001
}


def run_gpu0():
    # Iterate through the dictionary items (Objective Name, Entropy Value)
    for i, (obj, entropy) in enumerate(objective_configs.items()):

        # Run each experiment 3 times for the assigned entropy
        for run_id in range(1, 4):
            print(f"\n{'=' * 60}")
            print(f"TARGET {i + 1}/{len(objective_configs)}: {obj} | Entropy: {entropy} | Run: {run_id}/3")
            print(f"{'=' * 60}\n")

            # Prepare environment
            env = os.environ.copy()
            env["PMO_OBJECTIVE"] = obj
            env["TARGET_GPU_ID"] = "0"
            env["RL_ENTROPY_BETA"] = str(entropy)
            env["PMO_RUN_ID"] = str(run_id)

            try:
                # Run main.py as a subprocess
                subprocess.run(["python", "main.py"], env=env, check=True)

            except subprocess.CalledProcessError as e:
                print(f"!! ERROR running {obj} (Ent={entropy}, Run={run_id}) !!")
                print(e)
                # Optional: continue to next run
            except KeyboardInterrupt:
                print("\nBenchmark interrupted by user.")
                return

            # Small cooldown
            print(f"\nFinished run {run_id}. Cooling down for 5s...")
            time.sleep(5)


if __name__ == "__main__":
    run_gpu0()