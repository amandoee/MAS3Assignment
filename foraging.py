import numpy as np
import irsim
import yaml
import itertools
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION GENERATOR ---
def generate_config(n_agents, n_apples, grid_size=5):
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

    for i, (x, y) in enumerate(robot_coords):
        robot_entry = {
            'id': i,
            'state': [float(x), float(y), 0.0],
            'kinematics': { 'name': 'omni' },
            'shape': { 'name': 'circle', 'radius': 0.2 },
            'goal': [float(x), float(y), 0.0],
            'behavior': { 'name': 'dash' } 
        }
        config['robot'].append(robot_entry)

    for i, (x, y) in enumerate(apple_coords):
        obs_entry = {
            'id': i,
            'state': [float(x), float(y), 0.0],
            'shape': { 'name': 'circle', 'radius': 0.2 }
        }
        config['obstacle'].append(obs_entry)

    file_path = os.path.abspath('lbf_world.yaml')
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False)
    
    return file_path

# --- 2. LBF GAME LOGIC ---
class LBF_Wrapper:
    def __init__(self, ir_env, n_agents, n_apples, grid_size):
        self.env = ir_env
        self.n_agents = n_agents
        self.n_apples = n_apples 
        self.grid_size = grid_size
        self.agent_levels = [1] * n_agents 
        self.apple_levels = [2] * n_apples 
        
        self.robot_list = getattr(self.env, 'robot_list', [])
        if not self.robot_list and hasattr(self.env, 'robots'):
            self.robot_list = self.env.robots

        self.obstacle_list = getattr(self.env, 'obstacle_list', [])
        if not self.obstacle_list and hasattr(self.env, 'obstacles'):
            self.obstacle_list = self.env.obstacles

        # Actions: 0:None, 1:Up, 2:Down, 3:Left, 4:Right, 5:Load
        self.actions = [0, 1, 2, 3, 4, 5]
        self.reset()

    def reset(self):
        all_coords = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(all_coords)
        
        robot_coords = all_coords[:self.n_agents]
        apple_coords = all_coords[self.n_agents:self.n_agents+self.n_apples]
        
        self.agent_pos = [np.array(c, dtype=float) for c in robot_coords]
        self.apple_pos = [np.array(c, dtype=float) for c in apple_coords]
        
        self._sync_visuals()
        return self.get_state()
    
    def _sync_visuals(self):
        # Update Robots
        for i, r in enumerate(self.robot_list):
            if i < len(self.agent_pos):
                x, y = self.agent_pos[i]
                if isinstance(r.state, np.ndarray):
                    if r.state.ndim == 2:
                        r.state[0, 0] = x
                        r.state[1, 0] = y
                    else:
                        r.state[0] = x
                        r.state[1] = y

        # Update Apples
        for i, o in enumerate(self.obstacle_list):
            if i < len(self.apple_pos):
                if self.apple_pos[i] is not None:
                    x, y = self.apple_pos[i]
                else:
                    x, y = -10.0, -10.0 # Hide

                if isinstance(o.state, np.ndarray):
                    if o.state.ndim == 2:
                        o.state[0, 0] = x
                        o.state[1, 0] = y
                    else:
                        o.state[0] = x
                        o.state[1] = y

    def get_state(self):
        agents_tuple = tuple(tuple(p.astype(int)) for p in self.agent_pos)
        apples_tuple = tuple(tuple(a.astype(int)) if a is not None else (-1, -1) for a in self.apple_pos)
        return agents_tuple + apples_tuple

    def step(self, joint_actions):
        rewards = [0] * self.n_agents
        
        # 1. MOVE PHASE
        for i, action in enumerate(joint_actions):
            if action == 5: continue 
            
            old_pos = self.agent_pos[i].copy()
            proposed_pos = old_pos.copy()
            
            if action == 1: proposed_pos[1] += 1 
            elif action == 2: proposed_pos[1] -= 1 
            elif action == 3: proposed_pos[0] -= 1 
            elif action == 4: proposed_pos[0] += 1 
            
            proposed_pos = np.clip(proposed_pos, 0, self.grid_size-1)
            
            collision = False
            for j, other_pos in enumerate(self.agent_pos):
                if i != j and np.array_equal(proposed_pos, other_pos):
                    collision = True; break
            
            for apple_loc in self.apple_pos:
                if apple_loc is not None and np.array_equal(proposed_pos, apple_loc):
                    collision = True; break
            
            if not collision:
                self.agent_pos[i] = proposed_pos

        self._sync_visuals()

        # 2. LOAD PHASE
        loading_agents = [i for i, a in enumerate(joint_actions) if a == 5]
        
        for apple_idx, apple_loc in enumerate(self.apple_pos):
            if apple_loc is None: continue 
            adjacent_loaders = []
            for agent_idx in loading_agents:
                dist = np.linalg.norm(self.agent_pos[agent_idx] - apple_loc)
                if dist <= 1.1: adjacent_loaders.append(agent_idx)
            
            if adjacent_loaders:
                group_level = sum([self.agent_levels[a] for a in adjacent_loaders])
                if group_level >= self.apple_levels[apple_idx]:
                    for a in adjacent_loaders:
                        rewards[a] += 10 
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
        self.Q = {} # Q[state][joint_action]
        self.Models = {} 
        self.all_joint_actions = list(itertools.product(self.actions, repeat=self.n_agents))

    def get_joint_q(self, state, joint_action):
        if state not in self.Q: self.Q[state] = {}
        if joint_action not in self.Q[state]: self.Q[state][joint_action] = 0.0
        return self.Q[state][joint_action]

    def update_model(self, state, other_agent_id, action):
        if state not in self.Models: self.Models[state] = {}
        if other_agent_id not in self.Models[state]: 
            self.Models[state][other_agent_id] = {a: 0 for a in self.actions}
        self.Models[state][other_agent_id][action] += 1

    def get_predicted_prob(self, state, other_agent_id, action):
        if state not in self.Models or other_agent_id not in self.Models[state]:
            return 1.0 / len(self.actions)
        counts = self.Models[state][other_agent_id]
        total = sum(counts.values())
        if total == 0: return 1.0 / len(self.actions)
        return counts[action] / total

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        best_av = -np.inf
        best_action = 0
        
        for my_act in self.actions:
            av = 0
            for joint in self.all_joint_actions:
                if joint[self.id] != my_act: continue
                prob = 1.0
                for other_id in range(self.n_agents):
                    if other_id == self.id: continue
                    prob *= self.get_predicted_prob(state, other_id, joint[other_id])
                av += self.get_joint_q(state, joint) * prob
            
            if av > best_av:
                best_av = av
                best_action = my_act
        return best_action

    def learn(self, state, joint_action, reward, next_state):
        q_curr = self.get_joint_q(state, joint_action)
        max_next_q = 0.0
        if next_state in self.Q:
            vals = self.Q[next_state].values()
            if vals: max_next_q = max(vals)
        
        target = reward + self.gamma * max_next_q
        self.Q[state][joint_action] += self.alpha * (target - q_curr)
        
        for other_id, act in enumerate(joint_action):
            if other_id != self.id:
                self.update_model(state, other_id, act)

# --- 4. VISUALIZATION HELPER ---
def visualize_values(agents, grid_size, episode_num):
    """
    Plots a heatmap for each agent showing the Maximum Value they associate 
    with being in each grid square (projected from the high-dim Q-table).
    """
    fig, axes = plt.subplots(1, len(agents), figsize=(5 * len(agents), 5))
    if len(agents) == 1: axes = [axes]
    
    for i, agent in enumerate(agents):
        value_map = np.zeros((grid_size, grid_size))
        
        # Iterate over all states the agent has encountered
        for state, q_values in agent.Q.items():
            # state structure: ((ax, ay), (bx, by), (apple1), ...)
            # Extract THIS agent's position from the state
            my_pos = state[i] 
            x, y = int(my_pos[0]), int(my_pos[1])
            
            # Find the max Q-value for this specific state
            if q_values:
                max_q = max(q_values.values())
                # We store the MAX value seen for this position across all contexts
                if max_q > value_map[x, y]:
                    value_map[x, y] = max_q
        
        # Plot
        im = axes[i].imshow(value_map.T, origin='lower', cmap='viridis', interpolation='nearest')
        axes[i].set_title(f"Agent {i} Value Map")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(f"Projected Value Function - Episode {episode_num}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()

# --- 5. MAIN EXECUTION ---
def run_experiment(n_agents=10):
    print(f"--- Generating Config for {n_agents} Agents ---")
    grid_size = 5
    n_apples = 2
    
    yaml_path = generate_config(n_agents, n_apples, grid_size)
    
    try:
        env = irsim.make(yaml_path)
    except Exception as e:
        print(f"CRITICAL ERROR initializing irsim: {e}")
        return

    game = LBF_Wrapper(env, n_agents, n_apples, grid_size)
    agents = [JAL_AM_Agent(i, n_agents, game.actions) for i in range(n_agents)]
    
    episodes = 20
    
    # Enable interactive mode for plotting
    plt.ion()

    for ep in range(episodes):
        state = game.reset()
        total_reward = 0
        
        if hasattr(env, 'reset'): env.reset() 

        print(f"Episode {ep+1} started...", end="\r")
        
        for step in range(50):
            env.step() 
            
            actions = tuple(agent.choose_action(state) for agent in agents)
            next_state, rewards = game.step(actions)
            
            for i, agent in enumerate(agents):
                agent.learn(state, actions, rewards[i], next_state)
            
            state = next_state
            total_reward += sum(rewards)
            
            # Use a slightly faster render speed
            env.render(0.02)
            
            if all(pos is None for pos in game.apple_pos):
                print(f"\n  -> All apples collected in step {step}!")
                break
        
        print(f"\nEpisode {ep+1} End. Total Reward: {total_reward}")
        
        # Visualize Learning every 5 episodes
        if (ep + 1) % 5 == 0:
            print("Visualizing Q-Values...")
            visualize_values(agents, grid_size, ep+1)
            
    print("\nExperiment finished. Press Ctrl+C to exit.")
    env.show() 

if __name__ == '__main__':
    run_experiment(n_agents=3)