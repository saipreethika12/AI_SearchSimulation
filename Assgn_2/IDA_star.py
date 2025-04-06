import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def heuristic(node, goal):
    size = 4  # 4x4 FrozenLake
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


def search(path, g, bound, env, visited, goal, action_path, explored):
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
            explored.append(succ)
            t = search(path, g + cost(node, succ), bound, env, visited, goal, action_path, explored)
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
    explored = [start]
    start_time = time.time()

    while True:
        if time.time() - start_time > time_limit:
            return "NOT_FOUND (TIMEOUT)", [], [], time.time() - start_time

        t = search(path, 0, bound, env, visited, goal, action_path, explored)
        if t == "FOUND":
            return path[:], action_path[:], explored, time.time() - start_time
        if t == float('inf'):
            return "NOT_FOUND", [], explored, time.time() - start_time
        bound = t


def visualize_frozen_lake(env, path, explored_path=None):
    env.reset()

    if explored_path:
        print("Showing all explored states before optimal path:")
        for state in explored_path:
            env.unwrapped.s = state
            env.render()
            time.sleep(0.1)
        time.sleep(1)

    if not path or not isinstance(path, list):
        print("No valid path found.")
        return

    print("Showing optimal path:")
    for state in path:
        env.unwrapped.s = state
        env.render()
        time.sleep(0.5)

    time.sleep(2)
    env.close()


def run_ida_star(runs=5, timeout=600):
    goal_state = 15
    times = []

    for i in range(runs):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
        obs, _ = env.reset()
        print(f"\nRun {i + 1}:")
        path, actions, explored, duration = ida_star(env, obs, goal_state, time_limit=timeout)

        print(f"Time taken: {duration:.2f} seconds")
        times.append(duration)

        if path == "NOT_FOUND" or path == "NOT_FOUND (TIMEOUT)":
            print("No solution found.")
        else:
            print(f"Path: {path}")
            visualize_frozen_lake(env, path, explored)

        env.close()


    if times:
        avg = sum(times) / len(times)
        print(f"\nAverage time to find path: {avg:.2f} seconds")

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(times) + 1), times, marker='o', linestyle='--', color='green')
        plt.axhline(y=avg, color='red', linestyle='-', label=f"Average: {avg:.2f}s")
        plt.title("Time Taken in Each Run (IDA*)")
        plt.xlabel("Run Number")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_ida_star(runs=5, timeout=600)
