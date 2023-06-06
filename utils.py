import numpy as np

def select_optimal_action(q_table, stateNumber):
    max_q_value_action = None
    max_q_value = -100000

    if np.any(q_table):
        for actionNumber, action_q_value in enumerate(q_table[stateNumber].tolist()):
            if action_q_value >= max_q_value:
                max_q_value = action_q_value
                max_q_value_action = actionNumber

    return max_q_value_action