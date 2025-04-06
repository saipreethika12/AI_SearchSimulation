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

def simulated_annealing(env, max_iter=100, initial_temp=100.0, cooling_rate=0.995):
    num_nodes = env.num_nodes
    batch_size = env.batch_size

    best_tours = [random_tour(num_nodes) for _ in range(batch_size)]
    best_scores = evaluate_tour(env, best_tours)
    current_tours = best_tours.copy()
    current_scores = best_scores.copy()
    temp = initial_temp

    history = [(best_tours[0], best_scores[0])]  # For animation (only 1 tour, since batch_size = 1)

    for _ in range(max_iter):
        new_tours = []
        for tour in current_tours:
            i, j = np.random.choice(range(1, num_nodes), size=2, replace=False)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_tours.append(new_tour)

        new_scores = evaluate_tour(env, new_tours)

        for i in range(batch_size):
            delta = new_scores[i] - current_scores[i]
            if delta > 0 or np.exp(delta / temp) > np.random.rand():
                current_tours[i] = new_tours[i]
                current_scores[i] = new_scores[i]
                if current_scores[i] > best_scores[i]:
                    best_tours[i] = current_tours[i]
                    best_scores[i] = current_scores[i]
                    history.append((best_tours[0][:], best_scores[0]))
        temp *= cooling_rate

        # history.append((best_tours[0][:], best_scores[0]))  # Track best tour at each step

    return best_tours, best_scores, history

    
import matplotlib.animation as animation

def create_combined_animation(all_frames, filename="simulated_annealing_all_runs.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_data):
        G, pos, tour, score, title = frame_data
        ax.clear()
        coords = [pos[node] for node in tour] + [pos[tour[0]]]
        xs, ys = zip(*coords)

        ax.plot(xs, ys, color='green', linewidth=1, marker='o')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        ax.set_title(title)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=all_frames,
        interval=500,
        repeat=False
    )

    ani.save(filename, writer='pillow')
    plt.close(fig)
    print(f"Saved combined animation as '{filename}'")


def main():
    num_runs = 5
    times = []
    scores = []

    all_frames = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1} ---")
        env = TSPEnv(num_nodes=25, batch_size=1, num_draw=1)

        start = time.time()
        best_tours, best_scores, history = simulated_annealing(env, max_iter=1200)
        duration = time.time() - start

        times.append(duration)
        scores.append(best_scores[0])

        print(f"Time = {duration:.4f}s, Best Score = {best_scores[0]:.2f}")

        # Extract graph and positions
        vrp_graph = env.sampler.graphs[0]
        G = getattr(vrp_graph, 'graph', None)
        pos = nx.get_node_attributes(G, 'coordinates')

        # Add history frames
        for i, (tour, score) in enumerate(history):
            if run==0: all_frames.append((G, pos, tour, -score, f"Cost: {-score:.2f}"))

        # Pause at final best tour
        final_tour, final_score = history[-1]
        for _ in range(5):  # Pause for ~1 sec
            if run==0: all_frames.append((G, pos, final_tour, -final_score, f"FINAL | Cost: {-final_score:.2f}"))
        if run==0: create_combined_animation(all_frames, filename="simulated_annealing_all_runs.gif")
        
        
    # Create animation
    # create_combined_animation(all_frames, filename="simulated_annealing_all_runs.gif")

    # Plot execution time per run
    avg_time = np.mean(times)
    print(f"\nAverage time over {num_runs} runs: {avg_time:.4f} seconds")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_runs + 1), times, marker='o', linestyle='-')
    plt.title("Time Taken to Reach Optimum (Simulated Annealing)")
    plt.xlabel("Run Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

