import numpy as np

# Define maze (0: free, 1: obstacle, 2: goal)
maze = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 2]
])
states = maze.shape[0] * maze.shape[1]
actions = 4  # up, down, left, right
Q = np.zeros((states, actions))

# Helper functions
def state_to_xy(state):
    return state // maze.shape[1], state % maze.shape[1]

def xy_to_state(x, y):
    return x * maze.shape[1] + y

def get_next_state(state, action):
    x, y = state_to_xy(state)
    if action == 0: x -= 1  # up
    elif action == 1: x += 1  # down
    elif action == 2: y -= 1  # left
    elif action == 3: y += 1  # right
    if 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] != 1:
        return xy_to_state(x, y)
    return state

# Train Q-Learning
episodes = 1000
alpha, gamma, epsilon = 0.1, 0.9, 0.1
for _ in range(episodes):
    state = 0  # start at (0,0)
    while maze[state_to_xy(state)] != 2:
        if np.random.rand() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(Q[state])
        next_state = get_next_state(state, action)
        reward = 1 if maze[state_to_xy(next_state)] == 2 else -0.1
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

print("Q-Table:", Q)