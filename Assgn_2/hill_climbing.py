import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
from gym_vrp.envs import TSPEnv

def evaluate_tour(env, tours):
    env.reset()
    done = False
    total_reward = np.zeros(env.batch_size)

    for step in range(len(tours[0])):
        actions = np.array([tour[step] for tour in tours])[:, None]
        _, reward, done, _ = env.step(actions)
        total_reward += reward

    return total_reward

def random_tour(num_nodes):
    return [0] + list(np.random.permutation(range(1, num_nodes)))

def hill_climb(env, max_iter=1000):
    num_nodes = env.num_nodes
    batch_size = env.batch_size

    best_tours = [random_tour(num_nodes) for _ in range(batch_size)]
    best_scores = evaluate_tour(env, best_tours)
    
    evolution = [(best_tours[0].copy(), best_scores[0])]

    for _ in range(max_iter):
        new_tours = []
        for tour in best_tours:
            i, j = np.random.choice(range(1, num_nodes), size=2, replace=False)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_tours.append(new_tour)

        new_scores = evaluate_tour(env, new_tours)

        for i in range(batch_size):
            if new_scores[i] > best_scores[i]:
                best_tours[i] = new_tours[i]
                best_scores[i] = new_scores[i]
                evolution.append((best_tours[0].copy(), best_scores[0]))

    return best_tours, best_scores, evolution
    
import matplotlib.animation as animation


def animate_all_runs(all_frames, env, filename="all_runs_evolution.gif"):
    vrp_graphs = [g.graph for g in env.sampler.graphs]
    pos_list = [nx.get_node_attributes(G, 'coordinates') for G in vrp_graphs]
    
    fig, ax = plt.subplots()

    def update(frame_data):
        tour, score, run_idx, frame_idx, pause, G, pos = frame_data
        ax.clear()
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, ax=ax)
        coords = [pos[node] for node in tour] + [pos[tour[0]]]
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color='green', linewidth=1, marker='o')
        if pause:
            ax.set_title(f"DONE\nBest Score: {-score:.2f}", fontsize=12)
        else:
            ax.set_title(f"Step {frame_idx + 1} | Cost: {-score:.2f}")

    ani = animation.FuncAnimation(fig, update, frames=all_frames, interval=300)
    ani.save(filename, writer='pillow')
    plt.close(fig)
    print(f"Saved combined animation as '{filename}'")



def main():
    num_runs = 5
    times = []
    scores = []

    all_frames = []

    for run in range(num_runs):
        env = TSPEnv(num_nodes=25, batch_size=1, num_draw=1)
        G = env.sampler.graphs[0].graph
        pos = nx.get_node_attributes(G, 'coordinates')

        start = time.time()
        best_tours, best_scores, tour_evolution = hill_climb(env, max_iter=1000)
        duration = time.time() - start
        times.append(duration)
        scores.append(best_scores[0])

        print(f"Run {run + 1}: Time = {duration:.4f}s, Best Score = {best_scores[0]:.2f}")

        for idx, (tour, score) in enumerate(tour_evolution):
            if run==0: all_frames.append((tour, score, run, idx, False, G, pos))

        # Add 10 "pause" frames at end of this run with summary text
        for _ in range(10):
            if run==0: all_frames.append((best_tours[0], best_scores[0], run, idx, True, G, pos))
        if run==0: animate_all_runs(all_frames, env)

    # animate_all_runs(all_frames, env)

    avg_time = np.mean(times)
    print(f"\nAverage time over {num_runs} runs: {avg_time:.4f} seconds")

    # Optional: Plotting time per run
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_runs + 1), times, marker='o', linestyle='-')
    plt.title("Time Taken to Reach Optimum in Each Run")
    plt.xlabel("Run Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
