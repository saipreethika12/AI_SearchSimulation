import time
import gymnasium as gym
import numpy as np


def heuristic(node, goal):
    """
    Heuristic function: Manhattan Distance from current position to goal.
    """
    size = 4  # FrozenLake 4x4 grid
    return ((goal % size - node % size)  + (goal // size - node // size))


def is_goal(node, goal):
    return node == goal


def successors(env, node):
    """
    Expands the node by considering all possible actions and returns valid successors.
    """
    valid_moves = []
    for action in range(env.action_space.n):
        transitions = env.unwrapped.P[node][action]
        for prob, new_state, reward, done in transitions:
            if prob > 0:  # Ensure valid transitions
                valid_moves.append((new_state, action))
    return valid_moves


def cost(node, succ):
    """
    Constant step cost for each move.
    """
    return 1


def search(path, g, bound, env, visited, goal, action_path):
    node = path[-1]
    f = g + heuristic(node, goal)
    if f > bound:
        return f
    if is_goal(node, goal):
        return "FOUND"

    min_bound = float('inf')
    for succ, action in successors(env, node):
        if succ not in visited:
            path.append(succ)
            visited.add(succ)
            action_path.append(action)  # Store action taken
            t = search(path, g + cost(node, succ), bound,
                       env, visited, goal, action_path)
            if t == "FOUND":
                return "FOUND"
            if t < min_bound:
                min_bound = t
            path.pop()
            visited.remove(succ)
            action_path.pop()  # Remove failed action
    return min_bound


def ida_star(env, start, goal, time_limit=600):
    bound = heuristic(start, goal)
    path = [start]
    visited = {start}
    action_path = []  # Store the sequence of actions
    start_time = time.time()

    while True:
        if time.time() - start_time > time_limit:
            print("Timeout reached!")
            return "NOT_FOUND (TIMEOUT)", []

        t = search(path, 0, bound, env, visited, goal, action_path)

        if t == "FOUND":
            return path, action_path  # Return both states and actions
        if t == float('inf'):
            return "NOT_FOUND", []
        bound = t


# Initialize environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4",
               is_slippery=False, render_mode="human")

goal_state = 15  # Bottom-right corner of a 4x4 grid

# Run IDA* on Frozen Lake for 5 runs
for i in range(5):
    obs, _ = env.reset()
    path, actions = ida_star(env, obs, goal_state)

    print(f"Run {i+1}: Path -> {path}")

    if path == "NOT_FOUND":
        print("No solution found.")
        continue

    # Replay solution step by step
    obs, _ = env.reset()
    env.render()
    time.sleep(1)

    for action in actions:
        obs, _, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.5)  # Pause to see movement

        if done:
            break  # Stop if goal is reached

env.close()
