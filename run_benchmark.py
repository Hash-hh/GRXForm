import os
import subprocess
import time

# Full list of PMO objectives
objectives = [
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "zaleplon_mpo"
]


def run_sequential_benchmark():
    for i, obj in enumerate(objectives):
        print(f"\n{'=' * 60}")
        print(f"TARGET {i + 1}/{len(objectives)}: {obj}")
        print(f"{'=' * 60}\n")

        # Prepare environment with the specific objective
        env = os.environ.copy()
        env["PMO_OBJECTIVE"] = obj

        try:
            # Run main.py as a subprocess
            # This ensures a fresh Python instance (and fresh Ray cluster) for each task
            subprocess.run(["python", "main.py"], env=env, check=True)

        except subprocess.CalledProcessError as e:
            print(f"!! ERROR running {obj} !!")
            print(e)
            # Optional: sleep or continue
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
            break

        # Small cooldown between runs to let Ray/GPU clean up
        print(f"\nFinished {obj}. Cooling down for 5s...")
        time.sleep(5)


if __name__ == "__main__":
    run_sequential_benchmark()