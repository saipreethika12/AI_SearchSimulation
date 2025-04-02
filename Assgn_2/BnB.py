import gymnasium as gym
import heapq
import time


def branch_and_bound(env, max_time=600):
    start_time = time.time()
    state, _ = env.reset()
    frontier = [(0, state, [])]  # (cost, current_state, path)
    best_cost = float("inf")
    best_path = None

    while frontier and (time.time() - start_time) < max_time:
        cost, current_state, path = heapq.heappop(frontier)

        if current_state in path:
            continue  # Avoid loops

        path = path + [current_state]

        if env.unwrapped.desc.reshape(-1)[current_state] == b'G':  # Goal state
            if cost < best_cost:
                best_cost = cost
                best_path = path
                print("Goal reached with cost:", best_cost)
            continue

        for action in range(env.action_space.n):
            next_state, reward, done, truncated, _ = env.step(action)
            new_cost = cost + 1  # Uniform cost for each step
            if new_cost < best_cost:
                heapq.heappush(frontier, (new_cost, next_state, path))

            if done:
                env.reset()

    return best_path, best_cost


def visualize_frozen_lake(env, path):
    """ Renders the agent moving through the Frozen Lake """
    env.reset()

    if not path:
        print("No valid path found.")
        return

    for state in path:
        env.env.s = state  # Set the player's position manually
        env.render()
        time.sleep(0.5)  # Add delay for better visibility

    time.sleep(2)  # Pause at goal for clarity
    env.close()  # Close the environment after visualization


def run_experiment(runs=5):
    for _ in range(runs):
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                       is_slippery=True, render_mode="human")

        path, cost = branch_and_bound(env)

        if path is None:
            print("No path found. Skipping visualization.")
        else:
            print("Path:", path, "Cost:", cost)
            visualize_frozen_lake(env, path)

        env.close()  # Ensure the environment is closed properly


if __name__ == "__main__":
    run_experiment()
