import gymnasium as gym
import time
import matplotlib.pyplot as plt


def get_transitions(env):
    transitions = {}
    for state in range(env.observation_space.n):
        transitions[state] = {}
        for action in range(env.action_space.n):
            transitions[state][action] = env.unwrapped.P[state][action]
    return transitions


def depth_first_bnb(env, max_time=600, start_time=None):
    transitions = get_transitions(env)
    start_state, _ = env.reset()
    stack = [(start_state, [start_state], 0)]
    visited = set()

    best_cost = float('inf')
    best_path = []

    while stack and (time.time() - start_time) < max_time:
        state, path, cost = stack.pop()

        if cost >= best_cost:
            continue

        if env.unwrapped.desc.reshape(-1)[state] == b'G':
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]
            continue

        for action in range(env.action_space.n):
            for prob, next_state, reward, done in transitions[state][action]:
                if prob > 0 and next_state not in path and (cost + 1 <= best_cost):  # avoid cycles
                    stack.append((next_state, path + [next_state], cost + 1))

    return best_path, best_cost if best_path else None


def visualize_frozen_lake(env, path):
    env.reset()
    if not path:
        print("No valid path found.")
        return

    for state in path:
        env.unwrapped.s = state
        env.render()
        time.sleep(0.5)

    time.sleep(2)
    env.close()


def run_experiment(runs=5, timeout=600):
    times = []

    for i in range(runs):
        print(f"\nRun {i + 1}:")
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")

        start_time = time.time()
        path, cost = depth_first_bnb(env, max_time=timeout, start_time=start_time)
        duration = time.time() - start_time

        if path:
            print(f"Path: {path}")
            print(f"Cost: {cost}")
            print(f"Time: {duration:.3f} seconds")
            visualize_frozen_lake(env, path)
            times.append(duration)
        else:
            print(f"No path found within {timeout} seconds.")

        env.close()

    return times


def plot_results(times):
    if not times:
        print("No successful runs to plot.")
        return

    avg_time = sum(times) / len(times)
    print(f"\nAverage time to reach goal: {avg_time:.3f} seconds")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(times) + 1), times, marker='o', linestyle='--', color='blue')
    plt.axhline(y=avg_time, color='red', linestyle='-', label=f"Average: {avg_time:.3f}s")
    plt.title("Time Taken to Reach Goal in Each Run (DFBnB)")
    plt.xlabel("Run Number")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_times = run_experiment(runs=5, timeout=600)
    plot_results(run_times)
