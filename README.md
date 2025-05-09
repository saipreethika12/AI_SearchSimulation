﻿# AI_SearchSimulation
## 📖 Overview

This repository showcases the implementation of popular AI search algorithms in two different environments:

- **FrozenLake Environment**  
  - 🧭 **Branch and Bound (BnB)**  
  - ⭐ **Iterative Deepening A\* (IDA\*)**

- **Travelling Salesman Problem (TSP) Environment**  
  - ⛰️ **Hill Climbing**  
  - ❄️ **Simulated Annealing**

Each algorithm demonstrates different strategies to efficiently explore search spaces and solve optimization or pathfinding problems. The FrozenLake environment highlights decision-making in a grid-based world with stochastic transitions, while the TSP setup explores combinatorial optimization in route planning.

 
## 🚀 Setup Instructions

Follow these steps to clone the repository, install dependencies, and run the simulations.

### 1. Clone the Repository

```bash
git clone https://github.com/saipreethika12/AI_SearchSimulation.git
cd AI_SearchSimulation/Assgn_2
```

### 2. Set Up and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up environment for tsp
```bash
cd VRP-GYM
pip install -e .
cd ..
```

### 5. Run the Simulations

```bash
python BnB.py
python IDA_star.py
python hill_climbing.py
python Simulated_Annealing.py
```
