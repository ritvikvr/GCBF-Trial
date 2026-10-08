import argparse
import os
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data

from gcbf.trainer.utils import set_seed, read_settings
from gcbf.env import make_env
from gcbf.algo import make_algo

def main():
    parser = argparse.ArgumentParser(description="Visualize GCBF-PyTorch simulation in 3D")
    parser.add_argument('--path', type=str, default='./pretrained/SimpleDrone', help='Path to model logs/pretrained directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific model checkpoint path')
    parser.add_argument('--env', type=str, default='SimpleDrone', help='Environment to run')
    parser.add_argument('--algo', type=str, default='gcbf', help='Algorithm to use')
    parser.add_argument('-n', '--num-agents', type=int, default=8, help='Number of agents (drones)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-video', type=str, default=None, help='Path to save the output video (e.g., gcbf_simulation_3d.mp4)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load settings from pretrained model path
    try:
        settings = read_settings(args.path)
    except FileNotFoundError:
        print(f"Warning: settings.yaml not found in {args.path}. Using defaults.")
        settings = {'algo': args.algo, 'num_agents': args.num_agents, 'env': args.env}

    # Make environment
    env = make_env(
        env=settings.get('env', args.env),
        num_agents=args.num_agents,
        device=device,
        max_neighbors=12 if settings.get('algo', args.algo) == 'macbf' else None
    )
    env.reset()

    # Build the learned controller
    algo = make_algo(
        algo=settings.get('algo', args.algo),
        env=env,
        num_agents=args.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=env.action_dim,
        device=device,
        hyperparams=settings.get('hyper_params', None)
    )

    # Load the pretrained model
    model_path = os.path.join(args.path, 'models')
    if args.checkpoint:
        algo.load(args.checkpoint)
    else:
        controller_names = [i for i in os.listdir(model_path) if 'step' in i]
        controller_ids = sorted([int(i.split('step_')[1].split('.')[0]) for i in controller_names])
        latest_checkpoint = os.path.join(model_path, f'step_{controller_ids[-1]}')
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        algo.load(latest_checkpoint)

    # Run simulation and collect data
    print("Running simulation episode...")
    data = env.data
    trajectory = []
    
    # SimpleDrone environment variables in 3D
    goals = env._goal[:, :3].cpu().numpy()
    if hasattr(env, '_obs'):
        obs_pos = env._obs[:, :3].cpu().numpy()
    else:
        obs_pos = np.array([])
        
    t = 0
    safe = True
    min_dist_overall = float('inf')
    
    while True:
        data.update(Data(u_ref=env.u_ref(data)))
        action = algo.apply(data, rand=0)
        next_data, reward, done, info = env.step(action)
        
        state_diff = data.states.unsqueeze(1) - data.states.unsqueeze(0)
        pos_diff = state_diff[data.agent_mask, :, :3]
        dist = pos_diff.norm(dim=2)
        dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=device) * 1000
        min_dist = torch.min(dist).item()
        min_dist_overall = min(min_dist_overall, min_dist)
        
        if info.get('safe', 1.0) < 1.0:
            safe = False
            
        # Store state for visualization in 3D
        pos = data.pos.cpu().numpy()[:, :3]
        agent_mask = data.agent_mask.cpu().numpy()
        agent_pos = pos[agent_mask]
        
        trajectory.append({
            'pos': agent_pos,
            'min_dist': min_dist,
            'info': info
        })
        
        data = next_data
        t += 1
        
        if done:
            break
            
    reach = trajectory[-1]['info'].get('reach', torch.zeros(args.num_agents))
    success_rate = reach.sum().item() / args.num_agents
    
    steps_to_goal = np.full(args.num_agents, t)
    for step_idx, step_data in enumerate(trajectory):
        curr_reach = step_data['info'].get('reach', torch.zeros(args.num_agents)).cpu().numpy()
        for a_idx in range(args.num_agents):
            if a_idx < len(curr_reach) and curr_reach[a_idx] and steps_to_goal[a_idx] == t:
                steps_to_goal[a_idx] = step_idx
                
    reached_agents = reach.sum().item()
    timeout_agents = args.num_agents - reached_agents
    avg_steps = np.mean([steps_to_goal[i] for i in range(args.num_agents) if reach[i]]) if reached_agents > 0 else 0
    safety_violations = sum(1 for step_data in trajectory if step_data['min_dist'] < 0.20)

    print(f"Episode finished in {t} steps.")
    print(f"Collision Free: {safe} | Goal Reached Rate: {success_rate:.2f} | Min Dist: {min_dist_overall:.3f}")
    print(f"Number of agents reached: {int(reached_agents)}")
    print(f"Number of agents timed out: {int(timeout_agents)}")
    print(f"Average steps to goal: {avg_steps:.2f}")
    print(f"Number of safety violations (steps < 0.20): {safety_violations}")

    print("Generating 3D Animation...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    if len(obs_pos) > 0:
        ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2], c='gray', s=20, label='Obstacles', marker='o')
        
    # Plot goals
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c='green', s=150, marker='*', label='Goals')
    
    try:
        colors = plt.colormaps['tab10'].resampled(args.num_agents)
    except Exception:
        colors = plt.cm.get_cmap('tab10', args.num_agents)
    
    scatters = []
    trails = []
    texts = []
    for i in range(args.num_agents):
        color = colors(i) if callable(colors) else colors.colors[i % len(colors.colors)]
        scat = ax.scatter([], [], [], c=[color], s=250, marker='o', edgecolors='black', zorder=5)
        scatters.append(scat)
        trail, = ax.plot([], [], [], c=color, alpha=0.5, linestyle='--')
        trails.append(trail)
        
        # Goal labels
        ax.text(goals[i, 0], goals[i, 1], goals[i, 2] + 0.08, f"G{i}", fontsize=10, color=color, ha='center', weight='bold')

        # Agent ID labels
        txt = ax.text(0, 0, 0, f"{i}", fontsize=9, ha='center', va='center', zorder=10, color='white', weight='bold')
        texts.append(txt)

    # Setup axis limits
    ax.set_xlim(env._xyz_min[0], env._xyz_max[0])
    ax.set_ylim(env._xyz_min[1], env._xyz_max[1])
    ax.set_zlim(env._xyz_min[2] if env._xyz_min.shape[0] > 2 else 0.0, env._xyz_max[2] if env._xyz_max.shape[0] > 2 else 2.0)
    ax.set_title("GCBF SimpleDrone 3D Visualization", fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    
    # Status texts (2D on top of 3D plot)
    time_text = fig.text(0.05, 0.95, '', fontsize=10, weight='bold')
    status_text = fig.text(0.05, 0.92, '', fontsize=10)
    
    def init():
        for scat in scatters:
            scat._offsets3d = ([], [], [])
        for trail in trails:
            trail.set_data([], [])
            trail.set_3d_properties([])
        for text in texts:
            text.set_position((0, 0))
            text.set_3d_properties(0, 'z')
            text.set_text("")
        time_text.set_text('')
        status_text.set_text('')
        return scatters + trails + texts + [time_text, status_text]
        
    def update(frame):
        step_data = trajectory[frame]
        current_pos = step_data['pos']
        info = step_data['info']
        
        for i in range(args.num_agents):
            scatters[i]._offsets3d = (current_pos[i:i+1, 0], current_pos[i:i+1, 1], current_pos[i:i+1, 2])
            
            # Update Trail
            history = np.array([tr['pos'] for tr in trajectory[:frame+1]])[:, i, :]
            trails[i].set_data(history[:, 0], history[:, 1])
            trails[i].set_3d_properties(history[:, 2])
            
            # Update text position
            texts[i].set_position((current_pos[i, 0], current_pos[i, 1]))
            texts[i].set_3d_properties(current_pos[i, 2], 'z')
            
        time_text.set_text(f'Timestep: {frame} / {len(trajectory)-1}')
        
        reach_rate = info.get('reach', torch.zeros(1)).sum().item() / args.num_agents
        safe_rate = info.get('safe', 1.0)
        
        status_text.set_text(f"Agents: {args.num_agents} | Min Dist: {step_data['min_dist']:.3f}\n"
                             f"Safe: {safe_rate==1.0} | Reach Rate: {reach_rate:.2f}")
                             
        if safe_rate < 1.0:
            status_text.set_color('red')
        elif reach_rate == 1.0:
            status_text.set_color('green')
        else:
            status_text.set_color('black')
            
        return scatters + trails + texts + [time_text, status_text]

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, blit=False, interval=50)

    if args.save_video:
        print(f"Saving video to {args.save_video}...")
        ani.save(args.save_video, writer='ffmpeg', fps=20)
        print("Video saved.")
    else:
        plt.show()

if __name__ == "__main__":
    main()
