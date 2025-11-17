from objective_predictor.Prodrug.bbb_obj import BBBObjective
from rdkit import Chem

pairs = {
    "Morphine -> Heroin": {
        "parent": "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",
        "prodrug": "CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5[C@]2([C@H]1OC5=C(C=C4)OC(=O)C)CCN3C"
    },
    "Naltrexone -> Naltrexone Acetate": {
        "parent": "C1CC1CN2CC[C@]34[C@@H]5C(=O)CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O",
        "prodrug": "CC(=O)OC1=C2C3=C(C[C@@H]4[C@]5([C@]3(CCN4CC6CC6)[C@@H](O2)C(=O)CC5)O)C=C1"
    },
    "Dopamine -> Triacetyl-dopamine": {
        "parent": "C1=CC(=C(C=C1CCN)O)O",
        "prodrug": "CC(=O)NC(CC1=CC(OC(=O)C)=C(OC(=O)C)C=C1)"
    }
}

bbb_objective = BBBObjective(
    weight_logp_delta=1.5,
    weight_hdonor_delta=1.5,
    weight_cleavable=3.0
)

for name, p in pairs.items():
    parent_mol = Chem.MolFromSmiles(p["parent"])
    prodrug_mol = Chem.MolFromSmiles(p["prodrug"])

    if parent_mol is None or prodrug_mol is None:
        print(f"--- FAILED TO PARSE: {name} ---")
        continue

    result = bbb_objective.calculate(prodrug_mol, parent_mol)

    print(f"\n--- {name} ---")
    print(f"Total Reward: {result['total_reward']:.4f}")
    print(f"  Metrics:")
    print(
        f"    LogP Delta: {result['metrics']['logp_delta']:.2f} (Parent: {result['metrics']['logp_parent']:.2f}, Prodrug: {result['metrics']['logp_gen']:.2f})")
    print(
        f"    H-Donor Delta: {result['metrics']['hdonor_delta']} (Parent: {result['metrics']['hdonor_parent']}, Prodrug: {result['metrics']['hdonor_gen']})")
    print(f"    Cleavable Added: {result['metrics']['cleavable_bond_added']}")
