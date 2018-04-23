# RACETRACK
#
# EXERCISE 5.10
#
# Consider driving a race car around a turn like those shown in Figure 5.5. You want to go as fast as possible, but not
# so fast as to run off the track. In our simplified racetrack, the car is at one of a discrete set of grid positions,
# the cells in the diagram. The velocity is also discrete, a number of grid cells moved horizontally and vertically per
# time step. The actions are increments to the velocity components. Each may be changed by +1, -1, or 0 in each step,
# for a total of nine (3 ✕ 3) actions. Both velocity components are restricted to be nonnegative and less than 5, and
# they cannot both be zero except at the starting line. Each episode begins in one of the randomly selected start states
# with both velocity components zero and ends when the car crosses the finish line. The rewards are -1 for each step
# until the car crosses the finish line. If the car hits the track boundary, it is moved back to a random position on
# the starting line, both velocity components are reduced to zero, and the episode continues. Before updating the car's
# location at each time step, check to see if the projected path of the car intersects the track boundary. If it
# intersects the finish line, the episode ends, if it intersects anywhere else, the car is considered to have hit the
# track boundary and is sent back to the starting line. To make the task more challenging, with probability 0.1 at each
# time step the velocity increments are both zero, independently of the intended increments. Apply a Monte Carlo control
#  method to this task to compute the optimal policy from each starting state. Exhibit several trajectories following
# the optimal policy (but turn the noise off for these trajectories).

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from gym import Track


# A wrapper class for parameters of the algorithm
class Params:
    def __init__(self):
        # Discount
        self.gamma = 1

        # Number of episodes
        self.episodes = 100000

        # Number of exhibitions to show at the end
        self.exhibitions = 5

        # Start value for q, could be either 0 or a big negative number, I found however that a big negative number as
        # start q value highly improve performances
        self.start_q_value = -100000

        # Soft policy b type
        # Possible values: ['random', 'epsilon_greedy', 'epsilon_decay', 'deterministic']
        self.b_policy_type = 'epsilon_decay'

        # Probability for exploration of policy b
        if self.b_policy_type == 'epsilon_decay' or self.b_policy_type == 'random':
            # Completely random (no exploitation)
            self.epsilon = 1
        elif self.b_policy_type == 'deterministic':
            # Completely deterministic (no exploration)
            self.epsilon = 0
        else:
            self.epsilon = 0.01

        # Decay rate and minimum value of epsilon, for epsilon decay policy b
        self.epsilon_decay, self.epsilon_min = 0.999, 0.1


# A wrapper class for Off-policy Monte Carlo algorithm
class OffPolicyMonteCarlo:
    def __init__(self, track, params):
        """
        :param track: the track instance
        :param params: the parameters instance
        """
        # Set up the track
        self.track = track

        # Set up parameters
        self.params = params

        # Shape of state-action space
        state_actions_shape = (
                self.track.TRACK_HEIGHT, self.track.TRACK_WIDTH, self.track.MAX_VEL - self.track.MIN_VEL + 1,
                self.track.MAX_VEL - self.track.MIN_VEL + 1, 2 * self.track.MAX_ACC + 1, 2 * self.track.MAX_ACC + 1)

        # Shape of state space
        states_shape = (
                self.track.TRACK_HEIGHT, self.track.TRACK_WIDTH, self.track.MAX_VEL - self.track.MIN_VEL + 1,
                self.track.MAX_VEL - self.track.MIN_VEL + 1)

        # Initial state action pair and C values
        self.Q = np.full(state_actions_shape, params.start_q_value)
        self.C = np.zeros(state_actions_shape)

        # Initial policy, for each state choose a random action from ones allowed in this state
        self.pi = np.empty(states_shape, dtype=object)
        for i in range(states_shape[0]):
            for j in range(states_shape[1]):
                for v_i in range(self.track.MIN_VEL, self.track.MAX_VEL + 1):
                    for v_j in range(self.track.MIN_VEL, self.track.MAX_VEL + 1):
                        self.pi[i, j, v_i, v_j] = random.choice(track.A([i, j, v_i, v_j]))

    def resolve_track(self):
        """
        Resolve the track problem
        """
        i = 0
        while True:
            b = self.pi

            # Generate an episode using soft policy b
            Ss, As, Rs = self.generate_episode(b, self.params.b_policy_type)
            G = 0
            W = 1
            print('Episode n:', i, '\t Step needed: ',len(Ss))
            for t in range(len(Ss) - 1, -1, -1):
                G = self.params.gamma * G + Rs[t]
                self.C[tuple(Ss[t] + As[t])] += W
                self.Q[tuple(Ss[t] + As[t])] += (W / self.C[tuple(Ss[t] + As[t])]) * (G - self.Q[tuple(Ss[t] + As[t])])
                self.pi[tuple(Ss[t])] = random.choice([a for a in self.track.A(Ss[t]) if self.Q[tuple(Ss[t] + a)] ==
                                                       np.max([self.Q[tuple(Ss[t] + a)] for a in self.track.A(Ss[t])])])
                if As[t] != self.pi[tuple(Ss[t])]:
                    break
                W *= 1 / (1 - self.params.epsilon + self.params.epsilon/len(self.track.A(Ss[t])))
            i += 1
            if i > self.params.episodes:
                break

    def generate_episode(self, pi, policy_type, is_exhibition=False, S_0=None):
        """
        Generate an episode
        :param pi: policy to follow
        :param policy_type: type of policy to follow ['random', 'epsilon_greedy', 'epsilon_decay', 'deterministic']
        :param is_exhibition: whether is an exhibition or not
        :param S_0: initial state, fixed only in case of exhibition
        :return: [states visited, action taken, reward observed]
        """
        assert S_0 is not None or not is_exhibition

        # Decay epsilon
        if policy_type == 'epsilon_decay' and self.params.epsilon > self.params.epsilon_min and not is_exhibition:
            self.params.epsilon *= self.params.epsilon_decay

        Ss = []
        As = []
        Rs = []

        # Set the initial state, randomly if not in an exhibition
        if not is_exhibition:
            Ss.append(random.choice([[i, j, 0, 0] for i, j in self.track.get_state_locations(self.track.STATE_START)]))
        else:
            Ss.append(S_0)

        index = 0
        while True:
            if policy_type == 'deterministic':
                # Greedy
                As.append(pi[tuple(Ss[index])])
            elif np.random.binomial(1, self.params.epsilon) == 1 or policy_type == 'random':
                # Random
                As.append(random.choice(self.track.A(Ss[index])))
            else:
                # Greedy
                As.append(pi[tuple(Ss[index])])
            state, reward, done = self.track.take_action(Ss[index], As[index], is_exhibition)

            Rs.append(reward)
            if done:
                break
            else:
                Ss.append(state)
            index += 1
        return Ss, As, Rs

    def generate_exhibitions(self):
        """
        Generate an exhibition
        :return:
        """
        starts = [[i, j, 0, 0] for i, j in self.track.get_state_locations(self.track.STATE_START)]
        random.shuffle(starts)
        for idx, S_0 in enumerate(starts[:self.params.exhibitions]):
            # Generate an episode
            Ss, As, Rs = self.generate_episode(self.pi, 'deterministic', is_exhibition=True, S_0=S_0)

            # Create and plot the animation
            fig = plt.figure()
            ims = []
            for state in Ss:
                im = self.track.print_track(state)
                ims.append([im])
            anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)
            plt.show()


def exercise5_10():
    print('Exercise 5.10')

    # Set up a track instance, possible track types ['type0', 'type1', 'type2', 'type3']
    track = Track('type1')

    # Set up parameters
    params = Params()

    # Set up the algorithm
    off_policy_monte_carlo = OffPolicyMonteCarlo(track, params)

    # Solve the track problem
    off_policy_monte_carlo.resolve_track()

    # Show some exhibition of trajectories following the optimal policy
    off_policy_monte_carlo.generate_exhibitions()

exercise5_10()
