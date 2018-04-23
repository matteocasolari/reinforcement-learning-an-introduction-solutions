# DYNA MAZE
#
# EXERCISE 8.4
#
# The exploration bonus described above actually changes the estimated values of states and actions. Is this necessary?
# Suppose the bonus k*sqrt(tau) was used not in updates, but solely in action selection. That is, suppose the action
# selected was always that for which Q(St, a) + k*sqrt(tau(St, a)) was maximal. Carry out a gridworld experiment that
# tests and illustrates the strengths and weaknesses of this alternate approach.

import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from gym import Maze


# A wrapper class for parameters of the algorithm
class Params:
    def __init__(self):
        # Discount
        self.gamma = 0.95

        # Probability for exploration
        self.epsilon = 0.1

        # Step size
        self.alpha = 0.7

        # Weight for elapsed time
        self.k = 1e-4

        # n-step planning
        self.n = 5

        # Average over several independent runs
        self.runs = 20

        # Algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+', 'Alternative-Dyna-Q+']

        # Step number
        self.steps = 3000

        # Max number of steps of one episode, in sample mode
        self.max_steps = 100

        # Steps after which the maze change from the initial one
        self.changing_steps = 1000


# A wrapper class for Dyna-Q algorithm
class TabularDynaQ:
    def __init__(self, maze, params):
        """
        :param maze: the maze instance
        :param params: the parameters instance
        """
        # Set up the maze
        self.maze = maze

        # Set up parameters
        self.params = params

        # Initial state action pair values
        self.Q = np.zeros((self.maze.MAZE_HEIGHT, self.maze.MAZE_WIDTH, len(self.maze.actions)))

        # Initial model
        self.model = np.empty((self.maze.MAZE_HEIGHT, self.maze.MAZE_WIDTH, len(self.maze.actions)), dtype=list)

        # Initial time values, for Q+ algorithm
        self.time = 0
        self.times = np.zeros((self.maze.MAZE_HEIGHT, self.maze.MAZE_WIDTH, len(self.maze.actions)))

    def choose_action(self, state, method, deterministic=False):
        """
        Choose an action based on epsilon-greedy algorithm or deterministically from state
        :param state: state where the action is taken
        :param method: method implemented for the solution
        :param deterministic: if the action is chosen deterministically with respect to Q (default False)
        :return: chosen action
        """
        # Dyna-Q or Dyna-Q+ methods, standard action selection
        if method == 'Dyna-Q' or method == 'Dyna-Q+':
            if np.random.binomial(1, self.params.epsilon) == 1 and not deterministic:
                return random.choice(self.maze.actions)
            else:
                values = self.Q[state[0], state[1], :]
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])

        # Alternative Dyna-Q+ methods, action selection based on time too
        elif method == 'Alternative-Dyna-Q+':
            if np.random.binomial(1, self.params.epsilon) == 1 and not deterministic:
                return random.choice(self.maze.actions)
            elif not deterministic:
                values = self.Q[state[0], state[1], :] + \
                         self.params.k * np.sqrt(self.time - self.times[state[0], state[1], :])
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])
            else:
                values = self.Q[state[0], state[1], :]
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])

    def resolve_maze(self, method):
        """
        Resolve the maze problem
        :param method: method implemented for the solution
        :return: average cumulative rewards over different runs
        """
        # reset all variables, and set maze to initial one
        steps = 0
        reward_ = 0
        rewards = np.zeros(self.params.steps)
        self.maze.init_maze()

        has_changed = False

        while steps < self.params.steps:
            # Start from a random start state at the beginning of the episode
            state = random.choice(self.maze.get_state_locations(self.maze.START_STATE))
            prev_steps = steps

            while state not in self.maze.get_state_locations(self.maze.GOAL_STATE):
                # Track the steps
                steps += 1

                # Get action
                action = self.choose_action(state, method)

                # Take action
                new_state, reward = self.maze.take_action(state, action)

                # Q-Learning update
                self.Q[state[0], state[1], action] += \
                    self.params.alpha * (reward + self.params.gamma * np.max(self.Q[new_state[0], new_state[1], :])
                                         - self.Q[state[0], state[1], action])

                # Feed the model with experience
                self.model[state[0], state[1], action] = [new_state, reward]

                # Update time table
                self.time += 1
                self.times[state[0], state[1], action] = self.time

                # Sample experience from the model
                for _ in range(self.params.n):
                    # Planning state randomly chosen between previously observed ones
                    p_state = random.choice(
                        [[i, j] for i in np.arange(self.maze.MAZE_HEIGHT) for j in np.arange(self.maze.MAZE_WIDTH)
                         if not all(v is None for v in self.model[i, j, :])])

                    # Dyna-Q or Alternative Dyna-Q+ methods
                    if method == 'Dyna-Q' or method == 'Alternative-Dyna-Q+':
                        # Planning action randomly chosen between previously taken ones in this state
                        p_action = random.choice(
                            [a for a in self.maze.actions if self.model[p_state[0], p_state[1], a] is not None])

                        p_new_state, p_reward = self.model[p_state[0], p_state[1], p_action]

                        # Q-Learning update
                        self.Q[p_state[0], p_state[1], p_action] += self.params.alpha * (
                            p_reward + self.params.gamma * np.max(self.Q[p_new_state[0], p_new_state[1], :])
                            - self.Q[p_state[0], p_state[1], p_action])

                    # Dyna-Q+ method
                    elif method == 'Dyna-Q+':
                        # Planning action randomly chosen between all actions
                        p_action = random.choice(self.maze.actions)

                        # Actions already tried from chosen state
                        if self.model[p_state[0], p_state[1], p_action] is not None:
                            p_new_state, p_reward = self.model[p_state[0], p_state[1], p_action]

                        # Actions that had never been tried before from chosen state
                        else:
                            p_new_state, p_reward = p_state, 0

                        p_reward += self.params.k * np.sqrt(self.time - self.times[p_state[0], p_state[1], p_action])

                        # Q-Learning update
                        self.Q[p_state[0], p_state[1], p_action] += self.params.alpha * (
                            p_reward + self.params.gamma * np.max(self.Q[p_new_state[0], p_new_state[1], :])
                            - self.Q[p_state[0], p_state[1], p_action])

                state = new_state

            # Update rewards vector
            rewards[prev_steps:steps] = reward_
            reward_ += 1
            if steps > self.params.changing_steps and not has_changed:
                # Change the maze
                self.maze.change_maze()
                has_changed = True

        return rewards

    def sample_episode(self, method):
        """
        Sample one episode, generated using the last instance of Q
        :param method: method used to solve the maze
        :return:
        """
        # Start from a random start state at the beginning of the episode
        states = [random.choice(self.maze.get_state_locations(self.maze.START_STATE))]
        steps = 0

        while states[steps] not in self.maze.get_state_locations(self.maze.GOAL_STATE) and steps < self.params.max_steps:
            # Get action
            action = self.choose_action(states[steps], method, deterministic=True)

            # Take action
            new_state, _ = self.maze.take_action(states[steps], action)
            states.append(new_state)

            # Track the steps
            steps += 1

        ims = []

        # Create and plot the animation
        fig = plt.figure()
        plt.title(method)
        for state in states:
            im = self.maze.print_maze(state)
            ims.append([im])
        anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)
        plt.show()


def show_results(results):
    """
    Show results
    :param results: results to show
    """
    for result in results:
        rewards, method = result
        plt.plot(rewards, label=method)
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')


def exercise8_4():
    print('Exercise 8.4')

    # Set up a maze instance, possible maze types ['type0', 'type1', 'type2', 'type3']
    maze = Maze('type1')

    # Set up parameters
    params = Params()

    results = []
    for method in params.methods:
        rewards = np.zeros(params.steps)

        for run in range(params.runs):
            print('Method', method, 'run', run + 1)

            # Set up the algorithm
            tabular_dyna_q = TabularDynaQ(maze, params)

            # Solve the maze
            rewards_ = tabular_dyna_q.resolve_maze(method)

            # Update rewards vector
            rewards += rewards_

        # Show an episode (using the last Dyna-Q instance)
        tabular_dyna_q.sample_episode(method)

        # Average over runs
        rewards /= params.runs

        # Collect results
        results.append((rewards, method))

    # Show results
    show_results(results)
    plt.show()

exercise8_4()
