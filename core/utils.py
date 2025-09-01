import numpy as np

def softmax(x: np.array):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def log_scalar_dict(logger, d: dict, step: int):
    for k, v in d.items():
        logger.info(f"[STEP {step}] {k}: {v}")