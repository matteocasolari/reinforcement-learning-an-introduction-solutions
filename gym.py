import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# A wrapper class for a maze, containing all the information about the maze.
class Maze:
    def __init__(self, maze_type):
        """
        :param maze_type: the type of the maze, changing or not, to load
        """
        # All possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # All possible state type
        self.OBSTACLE_STATE = 0
        self.OK_STATE = 1
        self.START_STATE = 2
        self.GOAL_STATE = 3

        # Maze grid
        self.maze = []

        # Maze height
        self.MAZE_HEIGHT = 0

        # Maze width
        self.MAZE_WIDTH = 0

        # Maze states colors
        self.RGB_GREY = (.5, .5, .5)
        self.RGB_GREEN = (.5, 1, 0)
        self.RGB_RED = (1, 0, 0)
        self.RGB_YELLOW = (1, 1, 0)
        self.RGB_BLACK = (0, 0, 0)

        assert maze_type in ['type0', 'type1', 'type2', 'type3'], '\'{:s}\' not a valid name for maze type'.format(
            maze_type)
        script_dir = os.path.dirname(__file__)
        absolute_path = os.path.join(script_dir, 'mazes', maze_type)

        if maze_type in ['type1', 'type2']:
            self.initial_file_name = os.path.join(absolute_path, 'maze_init.csv')
            self.changed_file_name = os.path.join(absolute_path, 'maze_changed.csv')

        elif maze_type in ['type0', 'type3']:
            self.initial_file_name = os.path.join(absolute_path, 'maze.csv')
            self.changed_file_name = None

        self.load_maze_from_csv(self.initial_file_name)

    def take_action(self, state, action):
        """
        :param state: state where the action is taken
        :param action: action to take
        :return: [new state, reward]
        """
        # Current state coordinates
        x, y = state

        # Compute next state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.MAZE_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.MAZE_WIDTH - 1)
        if self.maze[x][y] == self.OBSTACLE_STATE:
            x, y = state

        # Compute the reward
        if self.maze[x][y] == self.GOAL_STATE:
            reward = 1.0
        else:
            reward = 0.0

        return [x, y], reward

    def load_maze_from_csv(self, file_name):
        """
        Load the maze from a csv file
        :param file_name: name of the file
        """
        # Load the maze
        with open(file_name) as csv_file:
            self.maze = [list(map(int, rec)) for rec in csv.reader(csv_file, delimiter=',')]

        # Recompute maze height and width
        self.MAZE_HEIGHT = len(self.maze)
        self.MAZE_WIDTH = len(self.maze[0])

    def change_maze(self):
        """
        Change the maze from initial to modified one if there is one, do nothing otherwise
        """
        if self.changed_file_name is not None:
            self.load_maze_from_csv(self.changed_file_name)

    def get_state_locations(self, state_type):
        """
        :param state_type: state type
        :return: a list containing all states of the maze of the given type
        """
        return [[i, j] for i in np.arange(self.MAZE_HEIGHT) for j in np.arange(self.MAZE_WIDTH) if
                self.maze[i][j] == state_type]

    def print_maze(self, state=None):
        """
        Create an image of the maze
        :param state: state of the agent (default None)
        :return: the image
        """
        maze_rgb = self.maze.copy()
        maze_rgb = [[self.RGB_GREEN if s == self.OBSTACLE_STATE else self.RGB_GREY if s == self.OK_STATE else
                    self.RGB_YELLOW if s == self.START_STATE else self.RGB_RED for s in row] for row in maze_rgb]
        if state is not None:
            x, y = state
            maze_rgb[x][y] = self.RGB_BLACK
        im = plt.imshow(maze_rgb, origin='lower', interpolation='none', animated=True)
        plt.gca().invert_yaxis()
        return im


# A wrapper class for a track, containing all the information about the track.
class Track:
    def __init__(self, track_type):
        """
        :param track_type: the type of the track to load
        """
        # All possible state type
        self.STATE_OUT = 0
        self.STATE_ON = 1
        self.STATE_START = 2
        self.STATE_END = 3

        # Maze grid
        self.track = []

        # Maze height
        self.TRACK_HEIGHT = 0

        # Maze width
        self.TRACK_WIDTH = 0

        # Max and min values of velocity in both directions
        self.MAX_VEL, self.MIN_VEL = 5, -5

        # Max absolute value of acceleration in both directions
        self.MAX_ACC = 1

        # All possible actions
        self.actions = [[a_i, a_j] for a_j in range(-self.MAX_ACC, self.MAX_ACC + 1) for a_i in
                        range(-self.MAX_ACC, self.MAX_ACC + 1)]

        # Probability that action has no effect due to failure
        self.FAILURE_PROB = 0.1

        # Maze states colors
        self.RGB_BROWN = (139 / 255, 69 / 255, 19 / 255)
        self.RGB_GREEN = (.5, 1, 0)
        self.RGB_RED = (1, 0, 0)
        self.RGB_YELLOW = (1, 1, 0)
        self.RGB_BLACK = (0, 0, 0)

        assert track_type in ['type0', 'type1', 'type2', 'type3'], '\'{:s}\' not a valid name for track type'.format(
            track_type)
        script_dir = os.path.dirname(__file__)
        absolute_path = os.path.join(script_dir, 'tracks', track_type, 'track.csv')

        self.load_track_from_csv(absolute_path)

    def take_action(self, state, action, is_exhibition):
        """
        :param state: state where the action is taken
        :param action: action to take
        :return: [new state, reward, done]
        """
        # Current state coordinates
        i, j, v_i, v_j = state
        a_i, a_j = action

        if np.random.binomial(1, self.FAILURE_PROB) == 1 and not is_exhibition:
            a_i, a_j = 0, 0

        p_i = i
        p_j = j
        v_i += a_i
        v_j += a_j
        i -= v_i
        j += v_j

        done = False

        states_visited = [[i_, j_] for i_ in range(min(i, p_i), max(i, p_i) + 1) for j_ in range(min(j, p_j), max(j, p_j) + 1)]
        states_visited_type = [self.track[i_][j_] if 0 <= i_ < self.TRACK_HEIGHT
                          and 0 <= j_ < self.TRACK_WIDTH else self.STATE_OUT for i_, j_ in states_visited]

        if self.STATE_END in states_visited_type:
            done = True
        elif self.STATE_OUT in states_visited_type:
            i, j, v_i, v_j = random.choice([[i, j, 0, 0] for i, j in self.get_state_locations(self.STATE_START)])

        return [i, j, v_i, v_j], -1, done

    def A(self, state):
        actions = []
        # All possible actions
        A = self.actions.copy()
        _, _, v_i, v_j = state

        # Discard actions that would make the speed of car negative or higher than max in at least one direction, or
        # zero in both directions
        for a in A:
            a_i, a_j = a
            if v_i + a_i < self.MIN_VEL or v_i + a_i > self.MAX_VEL:
                continue
            if v_j + a_j < self.MIN_VEL or v_j + a_j > self.MAX_VEL:
                continue
            if v_i + a_i == 0 and v_j + a_j == 0:
                continue
            actions.append(a)
        return actions

    def load_track_from_csv(self, file_name):
        """
        Load the track from a csv file
        :param file_name: name of the file
        """
        # Load the track
        with open(file_name) as csv_file:
            self.track = [list(map(int, rec)) for rec in csv.reader(csv_file, delimiter=',')]

        # Recompute track height and width
        self.TRACK_HEIGHT = len(self.track)
        self.TRACK_WIDTH = len(self.track[0])

    def get_state_locations(self, state_type):
        """
        :param state_type: state type
        :return: a list containing all states of the track of the given type
        """
        return [[i, j] for i in np.arange(self.TRACK_HEIGHT) for j in np.arange(self.TRACK_WIDTH) if
                self.track[i][j] == state_type]

    def print_track(self, state=None):
        tr_rgb = self.track.copy()
        tr_rgb = [[self.RGB_GREEN if s == self.STATE_OUT else self.RGB_BROWN if s == self.STATE_ON else
                  self.RGB_YELLOW if s == self.STATE_START else self.RGB_RED for s in row] for row in tr_rgb]
        if state is not None:
            x, y, _, _ = state
            tr_rgb[x][y] = self.RGB_BLACK
        im = plt.imshow(tr_rgb, origin='lower', interpolation='none', animated=True)
        plt.gca().invert_yaxis()
        return im