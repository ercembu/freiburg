import sys
import numpy as np
from collections import defaultdict, namedtuple
import itertools
from gridworld import GridworldEnv

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
      Q: A dictionary that maps from state -> action-values.
        Each value is a numpy array of length nA (see below)
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(Q[observation] == Q[observation].max()))
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def dyna_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, n=5):
    """
    Dyna-Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.
      n: number of planning steps

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # The model.
    # A nested dictionary that maps state -> (action -> (next state, reward, terminal flag)).
    M = defaultdict(lambda: np.zeros((env.nA, 3)))
    observed_sa = []

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # TODO: Implement this!
        s = env.reset()

        for t in itertools.count():
            a_probs = policy(s)
            a = np.random.choice(np.arange(len(a_probs)), p=a_probs)
            ns, r, d = env.step(a)

            stats.episode_rewards[i_episode] += r
            stats.episode_lengths[i_episode] = t

            Q[s][a] += alpha* (r + discount_factor* Q[ns].max()*(1-d) - Q[s][a])
            M[s][a] = [r, ns, d]

            if (s,a) not in observed_sa:
                observed_sa.append((s,a))
            
            for i in range(n):
                img_s, img_a = observed_sa[np.random.choice(len(observed_sa))]
                img_r, img_ns, img_d = M[img_s][img_a]

                Q[img_s][img_a] += alpha* (img_r + discount_factor* Q[img_ns].max()*(1-img_d) - Q[img_s][img_a])

            if d:
                break
            s = ns

    return Q, stats

if __name__ == "__main__":
    np.random.seed(0)
    env = GridworldEnv()
    Q, stats = dyna_q_learning(env, 10000)

    print("")
    for k, v in Q.items():
        print("%s: %s" % (k, v.tolist()))

