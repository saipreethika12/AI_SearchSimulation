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

def simulated_annealing(env, max_iter=1000, initial_temp=100.0, cooling_rate=0.995):
    num_nodes = env.num_nodes
    batch_size = env.batch_size

    best_tours = [random_tour(num_nodes) for _ in range(batch_size)]
    best_scores = evaluate_tour(env, best_tours)
    current_tours = best_tours.copy()
    current_scores = best_scores.copy()
    temp = initial_temp

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

        temp *= cooling_rate

    return best_tours, best_scores

def visualize_tour(env, tour, graph_index=0):
    vrp_graph = env.sampler.graphs[graph_index]
    G = getattr(vrp_graph, 'graph', None)

    if G is None:
        raise TypeError("Could not extract networkx.Graph from VRPGraph")

    tour = [int(node) for node in tour]
    pos = nx.get_node_attributes(G, 'coordinates')

    if not pos:
        raise ValueError("No 'coordinates' attribute found in node data.")

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)

    coords = [pos[node] for node in tour]
    coords.append(coords[0])  # Return to depot

    xs, ys = zip(*coords)
    plt.plot(xs, ys, color='green', linewidth=2, marker='o')
    plt.title(f"Simulated Annealing Tour for Graph {graph_index + 1}")
    plt.show()

def main():
    num_runs = 5
    times = []
    scores = []
    nodes=[5,10,12,15,20]

    for run in range(num_runs):
        env = TSPEnv(num_nodes=nodes[run], batch_size=1, num_draw=1)
        start = time.time()
        best_tours, best_scores = simulated_annealing(env, max_iter=1000)
        duration = time.time() - start
        times.append(duration)
        scores.append(best_scores[0])

        print(f"Run {run + 1}: Time = {duration:.4f}s, Best Score = {best_scores[0]:.2f}")
        visualize_tour(env, best_tours[0], graph_index=0)

    avg_time = np.mean(times)
    print(f"\nAverage time over {num_runs} runs: {avg_time:.4f} seconds")

    # Plotting time per run
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

