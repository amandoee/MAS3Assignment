import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# --- 1. CONFIGURATION GENERATOR ---
def generate_config(n_agents, n_apples, grid_size=3):
    # Ensure distinct coordinates
    if n_agents + n_apples > grid_size * grid_size:
        raise ValueError("Grid too small for number of items")
        
    all_coords = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    np.random.shuffle(all_coords)
    
    robot_coords = all_coords[:n_agents]
    apple_coords = all_coords[n_agents:n_agents+n_apples]

    config = {
        'world': {
            'height': grid_size,
            'width': grid_size,
            'step_time': 0.1,
            'sample_time': 0.1,
            'offset': [0, 0, 0],
            'xy_resolution': 0.01,
            'collision_threshold': 0.05
        },
        'robot': [],
        'obstacle': []
    }

    # AGENTS (Blue)
    for i, (x, y) in enumerate(robot_coords):
        robot_entry = {
            'id': i,
            'state': [float(x), float(y), 0.0],
            'kinematics': { 'name': 'omni' },
            'shape': { 'name': 'circle', 'radius': 0.2 },
            'behavior': { 'name': 'dash' },
            'color': [0, 0, 1.0] 
        }
        config['robot'].append(robot_entry)

    # APPLES (Red)
    for i, (x, y) in enumerate(apple_coords):
        obs_entry = {
            'id': i,
            'state': [float(x), float(y), 0.0],
            'shape': { 'name': 'circle', 'radius': 0.2 },
            'color': [1.0, 0, 0], 
            'static': True        
        }
        config['obstacle'].append(obs_entry)

    file_path = os.path.abspath(f'lbf_world_{n_agents}.yaml')
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False)
    
    return file_path

# --- 2. LBF GAME LOGIC ---
class LBF_Wrapper:
    def __init__(self, n_agents, n_apples, grid_size):
        self.n_agents = n_agents
        self.n_apples = n_apples 
        self.grid_size = grid_size
        
        self.agent_levels = [1] * n_agents 
        self.apple_levels = [2] * n_apples # Will be randomized in reset
        
        self.actions = [0, 1, 2, 3, 4, 5] # Wait, Up, Down, Left, Right, Load
        self.reset()

    def reset(self):
        all_coords = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(all_coords)
        
        robot_coords = all_coords[:self.n_agents]
        apple_coords = all_coords[self.n_agents:self.n_agents+self.n_apples]
        
        self.agent_pos = [np.array(c, dtype=float) for c in robot_coords]
        self.apple_pos = [np.array(c, dtype=float) for c in apple_coords]
    
        
        return self.get_state()

    def get_state(self):
        # State representation: tuple of agent coords + tuple of apple coords
        agents_tuple = tuple(tuple(p.astype(int)) for p in self.agent_pos)
        apples_tuple = tuple(tuple(a.astype(int)) if a is not None else (-1, -1) for a in self.apple_pos)
        # Add apple levels to state to make it Markovian regarding "can I pick this?"
        levels_tuple = tuple(self.apple_levels)
        return agents_tuple + apples_tuple + levels_tuple

    def step(self, joint_actions):
        rewards = [0.0] * self.n_agents
        
        # 1. MOVE PHASE
        for i, action in enumerate(joint_actions):
            if action == 5: continue  # No movement for load
            
            old_pos = self.agent_pos[i].copy()
            proposed_pos = old_pos.copy()
            if action == 1: proposed_pos[1] += 1  # Up
            elif action == 2: proposed_pos[1] -= 1  # Down
            elif action == 3: proposed_pos[0] -= 1  # Left
            elif action == 4: proposed_pos[0] += 1  # Right
            
            proposed_pos = np.clip(proposed_pos, 0, self.grid_size-1)
            
            # Check collisions
            collision = False
            for j, other_pos in enumerate(self.agent_pos):
                if i != j and np.array_equal(proposed_pos, other_pos): 
                    collision = True; break
            for apple_loc in self.apple_pos:
                if apple_loc is not None and np.array_equal(proposed_pos, apple_loc): 
                    collision = True; break
            
            if not collision: 
                self.agent_pos[i] = proposed_pos

        # 2. LOAD PHASE
        loading_agents = [i for i, a in enumerate(joint_actions) if a == 5]
        
        for apple_idx, apple_loc in enumerate(self.apple_pos):
            if apple_loc is None: continue
            
            # Check adjacency (Manhattan distance = 1)
            agents_adjacent = []
            for agent_idx in loading_agents:
                pos_diff = np.abs(self.agent_pos[agent_idx] - apple_loc)
                if np.sum(pos_diff) == 1: # Orthogonal adjacency
                    agents_adjacent.append(agent_idx)
            
            if agents_adjacent:
                group_level = sum([self.agent_levels[a] for a in agents_adjacent])
                if group_level >= self.apple_levels[apple_idx]:
                    for a in agents_adjacent:
                        rewards[a] += float(self.apple_levels[apple_idx] * self.agent_levels[a] / group_level)
                    self.apple_pos[apple_idx] = None
                    
        return self.get_state(), rewards

# --- 3. JAL-AM AGENT ---
class JAL_AM_Agent:
    def __init__(self, agent_id, n_agents, action_space):
        self.id = agent_id
        self.n_agents = n_agents
        self.actions = action_space

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.decay = 0.995

        self.Q = {}     # Q[state][joint_action]
        self.Models = {} # Models[state][other_id][action] -> count

    def get_q(self, state, joint_action):
        if state not in self.Q: self.Q[state] = {}
        if joint_action not in self.Q[state]: self.Q[state][joint_action] = 1.0 # Optimistic init
        return self.Q[state][joint_action]

    def update_model(self, state, other_id, action):
        if state not in self.Models: self.Models[state] = {}
        if other_id not in self.Models[state]: self.Models[state][other_id] = {a: 0.1 for a in self.actions}
        self.Models[state][other_id][action] += 1

    def prob(self, state, other_id, action):
        if state not in self.Models or other_id not in self.Models[state]:
            return 1.0 / len(self.actions)
        counts = self.Models[state][other_id]
        total = sum(counts.values())
        return counts[action] / total

    def expected_q(self, state, my_action):
        ev = 0.0
        def recurse(agent_idx, joint, prob):
            nonlocal ev
            if agent_idx == self.n_agents:
                ev += prob * self.get_q(state, tuple(joint))
                return

            if agent_idx == self.id:
                joint.append(my_action)
                recurse(agent_idx + 1, joint, prob)
                joint.pop()
            else:
                for a in self.actions:
                    p = self.prob(state, agent_idx, a)
                    joint.append(a)
                    recurse(agent_idx + 1, joint, prob * p)
                    joint.pop()
        recurse(0, [], 1.0)
        return ev

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        # Greedy w.r.t Expected Q
        best_val = -1e9
        best_actions = []
        for a in self.actions:
            val = self.expected_q(state, a)
            if val > best_val:
                best_val = val
                best_actions = [a]
            elif val == best_val:
                best_actions.append(a)
        return np.random.choice(best_actions)

    def learn(self, state, joint_action, reward, next_state):
        q_curr = self.get_q(state, joint_action)
        # Max over expected Q for next state
        next_best = max(self.expected_q(next_state, a) for a in self.actions)
        target = reward + self.gamma * next_best
        self.Q[state][joint_action] += self.alpha * (target - q_curr)
        
        for other_id, act in enumerate(joint_action):
            if other_id != self.id:
                self.update_model(state, other_id, act) # Use current state for model update

        self.epsilon *= self.decay # Decay epsilon

# --- 4. VISUALIZATION UPDATER ---
def update_dashboard(ax, game, episode_num, step):
    grid_size = game.grid_size
    ax.clear()
    ax.set_title(f"Episode {episode_num} | Step {step}")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, alpha=0.3)
    
    # Draw Apples
    for i, pos in enumerate(game.apple_pos):
        if pos is not None:
            lvl = game.apple_levels[i]
            rect = patches.Rectangle((pos[0]-0.35, pos[1]-0.35), 0.7, 0.7, color='red', alpha=0.6)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], f"L{lvl}", ha='center', va='center', color='white', fontweight='bold')

    # Draw Agents
    for i, pos in enumerate(game.agent_pos):
        circle = patches.Circle((pos[0], pos[1]), 0.3, color='blue', alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"A{i}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    
    plt.pause(0.001)

# --- 5. MAIN EXPERIMENT WITH METRICS ---
def run_metric_evaluation():
    # SETTINGS
    grid_size = 6
    n_apples = 2
    episodes = 300  # Kept low for demonstration speed (increase for real convergence)
    max_steps = 600  # Per user request (can be 300)
    
    # METRIC STORAGE
    results = {}
    
    # Comparison loop to test SCALABILITY (N=2 vs N=3)
    agent_counts = [2, 3, 4] 
    
    for n_agents in agent_counts:
        print(f"\n--- Starting Experiment with {n_agents} Agents ---")
        
        game = LBF_Wrapper(n_agents, n_apples, grid_size)
        agents = [JAL_AM_Agent(i, n_agents, game.actions) for i in range(n_agents)]
        
        # Metrics for this configuration
        rewards_history = []
        success_history = []  # 1 if all apples collected, 0 otherwise
        time_per_step_history = []
        
        # Visualization setup
        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 4))
        
        for ep in range(episodes):
            state = game.reset()
            ep_reward = 0
            step_times = []
            
            for step in range(max_steps):
                t0 = time.time()
                
                # --- AGENT DECISION & STEP ---
                actions = tuple(agent.choose_action(state) for agent in agents)
                next_state, rewards = game.step(actions)
                
                for i, agent in enumerate(agents):
                    agent.learn(state, actions, rewards[i], next_state)
                
                state = next_state
                ep_reward += sum(rewards)
                
                t1 = time.time()
                step_times.append(t1 - t0)
                
                # Visualization (only every 10 eps to save time)
                if ep % 300 == 0:
                    update_dashboard(ax, game, ep, step)

                # Check if all apples collected
                if all(pos is None for pos in game.apple_pos):
                    break
            
            # --- END OF EPISODE METRICS ---
            avg_step_time = np.mean(step_times)
            is_success = 1 if all(pos is None for pos in game.apple_pos) else 0
            
            rewards_history.append(ep_reward)
            success_history.append(is_success)
            time_per_step_history.append(avg_step_time)
            
            print(f"Ep {ep+1}/{episodes} | R: {ep_reward:.1f} | Time/Step: {avg_step_time:.4f}s | Success: {is_success}", end='\r')

        plt.close(fig)
        
        # Store results for this N
        results[n_agents] = {
            'rewards': rewards_history,
            'success': success_history,
            'time': time_per_step_history
        }

    # --- 6. PLOTTING METRICS ---
    print("\n\nGenerating Evaluation Plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # PLOT 1: Cumulative Reward (Moving Average)
    for n, data in results.items():
        window = 5
        smoothed = np.convolve(data['rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=f'{n} Agents')
    ax1.set_title("Cumulative Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True)

    # PLOT 2: Convergence Rate (Success %)
    for n, data in results.items():
        window = 10
        # success rate over moving window
        smoothed = np.convolve(data['success'], np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, label=f'{n} Agents')
    ax2.set_title("Convergence Rate (Success %)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate (Moving Avg)")
    ax2.legend()
    ax2.grid(True)

    # PLOT 3: Scalability (Time per Step)
    avg_times = [np.mean(results[n]['time']) for n in agent_counts]
    ax3.bar([str(n) for n in agent_counts], avg_times, color=['blue', 'orange'])
    ax3.set_title("Scalability: Comp. Complexity")
    ax3.set_xlabel("Number of Agents")
    ax3.set_ylabel("Avg Time per Step (seconds)")
    
    plt.tight_layout()
    plt.show()
    input("Press Enter to close...")

if __name__ == '__main__':
    run_metric_evaluation()