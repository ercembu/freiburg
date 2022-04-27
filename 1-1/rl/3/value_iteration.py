import numpy as np


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
      env: OpenAI environment. env.P represents the transition probabilities of the environment.
      theta: Stopping threshold. If the value of all states changes less than theta
        in one iteration we are done.
      discount_factor: lambda time discount factor.

    Returns:
      A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    # TODO: Implement this!
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, ns, r, _ in env.P[s][a]:
                    action_values[a] += prob * (r + discount_factor * V[ns])
            V[s] = np.max(action_values)
            policy[s] = np.eye(env.nA)[np.argmax(action_values)]
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break

    return policy, V
