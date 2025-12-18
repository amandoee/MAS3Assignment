import numpy as np
import irsim
import yaml
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. CONFIGURATION GENERATOR ---
def generate_config(n_agents, n_apples, grid_size=3):
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
            #'goal': [float(x), float(y), 0.0],
            'behavior': { 'name': 'dash' },
            'color': [0, 0, 1.0] # Blue
        }
        config['robot'].append(robot_entry)

    # APPLES (Red)
    for i, (x, y) in enumerate(apple_coords):
        obs_entry = {
            'id': i,
            'state': [float(x), float(y), 0.0],
            'shape': { 'name': 'circle', 'radius': 0.2 },
            'color': [1.0, 0, 0], # Red
            'static': True        # Ensure apples don't slide
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
        
        # --- GAME SETTINGS ---
        # Agent levels: n, n-1, n-2, ..., 1
        self.agent_levels = list(range(n_agents, 0, -1))
        # Max food level: sum of 3 highest agent levels (ensures any 3 agents can collect any food)
        # Accounts for edge spawn positions where only 3 agents can be adjacent
        top_3_levels = sorted(self.agent_levels, reverse=True)[:3]
        self.max_food_level = sum(top_3_levels)
        # Generate random food levels from 1 to max_food_level
        self.apple_levels = [np.random.randint(1, self.max_food_level + 1) for _ in range(n_apples)] 
        
        self.robot_list = []
        self.obstacle_list = []
        if self.env is not None:
            self.robot_list = getattr(self.env, 'robot_list', [])
            if not self.robot_list and hasattr(self.env, 'robots'):
                self.robot_list = self.env.robots

            self.obstacle_list = getattr(self.env, 'obstacle_list', [])
            if not self.obstacle_list and hasattr(self.env, 'obstacles'):
                self.obstacle_list = self.env.obstacles

        self.actions = [0, 1, 2, 3, 4, 5]
        self.reset()

    def reset(self):
        all_coords = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(all_coords)
        
        robot_coords = all_coords[:self.n_agents]
        apple_coords = all_coords[self.n_agents:self.n_agents+self.n_apples]
        
        self.agent_pos = [np.array(c, dtype=float) for c in robot_coords]
        self.apple_pos = [np.array(c, dtype=float) for c in apple_coords]
        
        # Regenerate random apple levels (from 1 to sum of all agent levels)
        self.apple_levels = [np.random.randint(1, self.max_food_level + 1) for _ in range(self.n_apples)]
        
        self._sync_visuals()
        return self.get_state()
    
    def _sync_visuals(self):
        # Update Robots
        for i, r in enumerate(self.robot_list):
            if i < len(self.agent_pos):
                x, y = self.agent_pos[i]
                if isinstance(r.state, np.ndarray):
                    if r.state.ndim == 2:
                        r.state[0, 0] = x; r.state[1, 0] = y
                    else:
                        r.state[0] = x; r.state[1] = y

        # Update Apples
        for i, o in enumerate(self.obstacle_list):
            if i < len(self.apple_pos):
                if self.apple_pos[i] is not None:
                    x, y = self.apple_pos[i]
                else:
                    x, y = -10.0, -10.0 # Hide
                if isinstance(o.state, np.ndarray):
                    if o.state.ndim == 2:
                        o.state[0, 0] = x; o.state[1, 0] = y
                    else:
                        o.state[0] = x; o.state[1] = y

    def get_state(self):
        agents_tuple = tuple(tuple(p.astype(int)) for p in self.agent_pos)
        apples_tuple = tuple(tuple(a.astype(int)) if a is not None else (-1, -1) for a in self.apple_pos)
        return agents_tuple + apples_tuple

    def step(self, joint_actions):
        rewards = [0.0] * self.n_agents
        
        # 1. MOVE PHASE
        for i, action in enumerate(joint_actions):
            if action == 5: continue  # No movement for load action
            
            old_pos = self.agent_pos[i].copy()
            proposed_pos = old_pos.copy()
            if action == 1: proposed_pos[1] += 1  # Up
            elif action == 2: proposed_pos[1] -= 1  # Down
            elif action == 3: proposed_pos[0] -= 1  # Left
            elif action == 4: proposed_pos[0] += 1  # Right
            elif action == 0: pass  # No movement
            
            proposed_pos = np.clip(proposed_pos, 0, self.grid_size-1)
            
            # Check collisions
            collision = False
            # Agent-Agent collision
            for j, other_pos in enumerate(self.agent_pos):
                if i != j and np.array_equal(proposed_pos, other_pos): 
                    collision = True; break
            # Agent-Apple collision (cannot move onto apple)
            for apple_loc in self.apple_pos:
                if apple_loc is not None and np.array_equal(proposed_pos, apple_loc): 
                    collision = True; break
            
            if not collision: 
                self.agent_pos[i] = proposed_pos

        self._sync_visuals()

        # 2. LOAD PHASE (Standard LBF: agents ADJACENT to food can collect it together)
        # Per official implementation: adjacent means orthogonal only (N/S/E/W), not diagonal
        loading_agents = [i for i, a in enumerate(joint_actions) if a == 5]
        
        for apple_idx, apple_loc in enumerate(self.apple_pos):
            if apple_loc is None: continue
            
            # Find all agents ADJACENT to this food (orthogonal only: distance = 1)
            agents_adjacent = []
            for agent_idx in loading_agents:
                pos_diff = np.abs(self.agent_pos[agent_idx] - apple_loc)
                # Adjacent = exactly 1 step away in one direction (N/S/E/W only)
                if (pos_diff[0] == 1 and pos_diff[1] == 0) or (pos_diff[0] == 0 and pos_diff[1] == 1):
                    agents_adjacent.append(agent_idx)
            
            if agents_adjacent:
                # Check if combined level is sufficient
                group_level = sum([self.agent_levels[a] for a in agents_adjacent])
                if group_level >= self.apple_levels[apple_idx]:
                    # PDF FORMULA: "points equal to the level of the food divided by their contribution"
                    # reward = food_level × (agent_level / total_participating_level)
                    # This ensures agents are rewarded proportionally to their contribution
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

        # Q[state][joint_action]
        self.Q = {}

        # Models[state][other_id][action] -> count
        self.Models = {}

    # ---------- Q helpers ----------
    def get_q(self, state, joint_action):
        if state not in self.Q:
            self.Q[state] = {}
        if joint_action not in self.Q[state]:
            self.Q[state][joint_action] = 0.0
        return self.Q[state][joint_action]

    # ---------- Opponent model ----------
    def update_model(self, state, other_id, action):
        if state not in self.Models:
            self.Models[state] = {}
        if other_id not in self.Models[state]:
            self.Models[state][other_id] = {a: 0 for a in self.actions}
        self.Models[state][other_id][action] += 1

    def prob(self, state, other_id, action):
        if state not in self.Models:
            return 1.0 / len(self.actions)
        if other_id not in self.Models[state]:
            return 1.0 / len(self.actions)

        counts = self.Models[state][other_id]
        total = sum(counts.values())
        if total == 0:
            return 1.0 / len(self.actions)
        return counts[action] / total

    # ---------- Expected value ----------
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

    # ---------- Action selection ----------
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)

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

    # ---------- Learning ----------
    def learn(self, state, joint_action, reward, next_state, terminal=False):
        q_curr = self.get_q(state, joint_action)

        if terminal:
            target = reward
        else:
            next_best = max(self.expected_q(next_state, a) for a in self.actions)
            target = reward + self.gamma * next_best

        self.Q[state][joint_action] += self.alpha * (target - q_curr)

        # Update opponent models using next_state
        for other_id, act in enumerate(joint_action):
            if other_id != self.id:
                self.update_model(next_state, other_id, act)

# --- 4. DASHBOARD VISUALIZATION ---
def update_dashboard(fig, ax, game, episode_num):
    """
    Updates the Matplotlib window with the Game Grid (showing positions and levels).
    """
    grid_size = game.grid_size
    
    ax.clear()
    ax.set_title(f"Level Based Foraging - Episode {episode_num}")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Draw Apples (Red)
    for i, pos in enumerate(game.apple_pos):
        if pos is not None:
            lvl = game.apple_levels[i]
            rect = patches.Rectangle((pos[0]-0.35, pos[1]-0.35), 0.7, 0.7, 
                                     color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], f"F{i}\nL{lvl}", ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=9)

    # Draw Agents (Blue)
    for i, pos in enumerate(game.agent_pos):
        lvl = game.agent_levels[i]
        circle = patches.Circle((pos[0], pos[1]), 0.3, color='blue', alpha=0.8, 
                               edgecolor='darkblue', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"L{lvl}", ha='center', va='center', 
               color='white', fontweight='bold', fontsize=9)
        ax.text(pos[0], pos[1]-0.5, f"A{i}", ha='center', va='top', 
               color='blue', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# --- 5. MAIN ---
def run_experiment(n_agents=3):
    grid_size = 8
    n_apples = 3
    
    yaml_path = generate_config(n_agents, n_apples, grid_size)
    
    game = LBF_Wrapper(None, n_agents, n_apples, grid_size)
    agents = [JAL_AM_Agent(i, n_agents, game.actions) for i in range(n_agents)]
    
    episodes = 500
    
    # Setup Matplotlib Dashboard (Game State Only)
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    for ep in range(episodes):
        state = game.reset()
        
        # Curriculum learning: gradually increase food difficulty
        # Early episodes: easier food (level 1), later: harder food (up to max)
        # if ep < 10:
        #     game.apple_levels = [1] * game.n_apples  # Level 1: weakest agent can collect alone
        # else:
        game.apple_levels = [np.random.randint(1, game.max_food_level + 1) for _ in range(game.n_apples)]
        
        game._sync_visuals()  # Ensure game state is synchronized
        episode_reward = 0
        foods_collected = 0

        print(f"Episode {ep+1} started...", end="\r")
        
        for step in range(30):
            actions = tuple(agent.choose_action(state) for agent in agents)
            next_state, rewards = game.step(actions)
            
            for i, agent in enumerate(agents):
                agent.learn(state, actions, rewards[i], next_state)
            
            state = next_state
            episode_reward += sum(rewards)
            
            # Count collected food
            if any(pos is None for pos in game.apple_pos):
                foods_collected = sum(1 for pos in game.apple_pos if pos is None)
            
            # --- RENDER ONLY MATPLOTLIB VIEW (Game State Only) ---
            update_dashboard(fig, ax, game, ep+1) # Matplotlib View (Game State Only)
            
            if all(pos is None for pos in game.apple_pos):
                print(f"\n  -> All apples collected in step {step}!")
                break
        
        print(f"Episode {ep+1} End. Total Reward: {episode_reward:.2f} | Food Collected: {foods_collected}")
            
    print("\nExperiment finished.") 

if __name__ == '__main__':
    run_experiment(n_agents=2)