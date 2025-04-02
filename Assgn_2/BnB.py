import heapq
import time
import gymnasium as gym
import pygame
import numpy as np


def visualize_path(env, path):
    pygame.init()
    size = env.render().shape[:2]
    window = pygame.display.set_mode((size[1], size[0]))
    pygame.display.set_caption('Frozen Lake Path Visualization')

    clock = pygame.time.Clock()
    for state in path:
        frame = env.render()
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, axes=(1, 0, 2)))
        window.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(1)  # Adjust the speed as needed

    pygame.quit()


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

        if env.unwrapped.desc.reshape(-1)[current_state] == b'G':
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


def run_experiment(runs=5):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                   is_slippery=True, render_mode='rgb_array')
    times = []

    for _ in range(runs):
        start_time = time.time()
        path, cost = branch_and_bound(env)
        times.append(time.time() - start_time)
        print("Run completed in:", times[-1], "seconds with cost:", cost)

        if path:
            visualize_path(env, path)

    env.close()
    return times


if __name__ == "__main__":
    run_experiment()
