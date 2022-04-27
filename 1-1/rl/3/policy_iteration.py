import numpy as np


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
      policy: [S, A] shaped matrix representing the policy.
      env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
      theta: We stop evaluation once our value function change is less than theta for all states.
      discount_factor: gamma discount factor.

    Returns:
      Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        # TODO: Implement this!
        delta = 0
        for i in range(env.nS):
            v = 0
            pol = policy[i]

            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[i][a]:
                    v += pol[a] * prob * (reward + discount_factor*V[next_state])

            delta = max(delta, np.abs(v-V[i]))
            V[i] = v
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
      env: The OpenAI envrionment.
      policy_eval_fn: Policy Evaluation function that takes 3 arguments:
        policy, env, discount_factor.
      discount_factor: Lambda discount factor.

    Returns:
      A tuple (policy, V).
      policy is the optimal policy, a matrix of shape [S, A] where each state s
      contains a valid probability distribution over actions.
      V is the value function for the optimal policy.

    """
    V = np.zeros(env.nS)
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # TODO: Implement this!
        policy_stable = True
        V = policy_eval(policy, env, discount_factor)
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            vals = np.zeros([env.nA])
            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    vals[a] += prob * (reward + discount_factor * V[next_state])
            best = np.argmax(vals)

            if old_action != best:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best]

        if policy_stable:
            return policy, V
