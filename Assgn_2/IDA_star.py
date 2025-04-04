import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def heuristic(node, goal):
    size = 4
    return abs(goal % size - node % size) + abs(goal // size - node // size)


def is_goal(node, goal):
    return node == goal


def successors(env, node):
    valid_moves = []
    for action in range(env.action_space.n):
        transitions = env.unwrapped.P[node][action]
        for prob, new_state, reward, done in transitions:
            if prob > 0:
                valid_moves.append((new_state, action))
    return valid_moves


def cost(node, succ):
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
            action_path.append(action)
            t = search(path, g + cost(node, succ), bound, env, visited, goal, action_path)
            if t == "FOUND":
                return "FOUND"
            if t < min_bound:
                min_bound = t
            path.pop()
            visited.remove(succ)
            action_path.pop()
    return min_bound


def ida_star(env, start, goal, time_limit=600):
    bound = heuristic(start, goal)
    path = [start]
    visited = {start}
    action_path = []
    start_time = time.time()

    while True:
        if time.time() - start_time > time_limit:
            return "NOT_FOUND (TIMEOUT)", [], (time.time() - start_time) 

        t = search(path, 0, bound, env, visited, goal, action_path)
        if t == "FOUND":
            return path, action_path, (time.time() - start_time)
        if t == float('inf'):
            return "NOT_FOUND", [], (time.time() - start_time)
        bound = t


# Initialize environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4",
               is_slippery=False, render_mode="human")

goal_state = 15
times = []

for i in range(5):
    obs, _ = env.reset()
    print(f"\nRun {i+1}:")
    path, actions, duration = ida_star(env, obs, goal_state)

    print(f"Time taken: {duration:.2f} ms")
    times.append(duration)

    if path == "NOT_FOUND" or path == "NOT_FOUND (TIMEOUT)":
        print("No solution found.")
        continue

    print(f"Path: {path}")

    obs, _ = env.reset()
    env.render()
    time.sleep(1)

    for action in actions:
        obs, _, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.5)
        if done:
            break

env.close()

# Show average time
if times:
    avg = sum(times) / len(times)
    print(f"\nAverage time to find path: {avg:.2f} ms")

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(times)+1), times, marker='o', linestyle='--', color='blue')
    plt.axhline(y=avg, color='red', linestyle='-', label=f"Average: {avg:.2f} ms")
    plt.title("IDA* Time to Reach Goal (FrozenLake)")
    plt.xlabel("Run Number")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
