from tdc import Oracle


class KinaseMPOObjective:
    def __init__(self):
        # Initialize the 4 components from TDC
        # These load the specific Random Forest/Regression models used in the literature
        self.gsk3 = Oracle(name='GSK3B')
        self.jnk3 = Oracle(name='JNK3')
        self.qed = Oracle(name='QED')
        self.sa = Oracle(name='SA')

    def score(self, smiles):
        """
        Returns the scalar reward for Reinforcement Learning (Sum of properties).
        Range: [0.0, 1.0]
        """
        if not smiles:
            return 0.0

        # try:
        # 1. Get raw scores
        # GSK3B and JNK3 return probabilities [0, 1]
        gsk_score = self.gsk3(smiles)
        jnk_score = self.jnk3(smiles)

        # QED is [0, 1]
        qed_score = self.qed(smiles)

        # SA is [1, 10] where 1 is best. We normalize to [0, 1] where 1 is best.
        raw_sa = self.sa(smiles)
        sa_norm = (10 - raw_sa) / 9.0

        # 2. Compute RL Reward (Soft Signal)
        # We SUM them to provide a smooth gradient for the agent.
        # If we used hard constraints here, the agent would get 0 reward
        # for 99% of steps and fail to learn (Cold Start problem).
        reward = gsk_score + jnk_score + qed_score + sa_norm
        reward = reward / 4.0 # Normalize to [0, 1]

        return reward
        # except Exception as e:
        #     # If RDKit fails to parse, return 0
        #     return 0.0

    def is_successful(self, smiles):
        """
        Strict Binary Evaluation for the 'Success Rate' metric.
        Matches the definition in RationaleRL (Jin et al., 2020).
        """
        # try:
        gsk = self.gsk3(smiles)
        jnk = self.jnk3(smiles)
        qed = self.qed(smiles)
        sa = self.sa(smiles)

        # The hard thresholds defined in the benchmark paper
        return (gsk >= 0.5) and (jnk >= 0.5) and (qed >= 0.6) and (sa < 4.0)
        # except:
        #     return False