from collections import deque
import torch
from torch import optim
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
from GAIL.models import GAILDiscriminator
from GAIL.training_utils import TransitionDataset, TransitionDataset2, adversarial_imitation_update, compute_advantages, \
    indicate_absorbing, ppo_update, flatten_list_dicts, convert_traj
import numpy as np

def train_g(args, agent, env):

    # Load expert trajectories
    # with open(args.expert_trajectories_path, 'rb') as handle:
    #     expert_trajectories = pickle.load(handle)
    with open(args.expert_trajectories_path, 'rb') as handle:
        expert_trajectories = np.load(handle, allow_pickle=True)
    expert_trajectories = TransitionDataset(expert_trajectories.tolist())
    # print(expert_trajectories)

    # Set up actor-critic model optimiser
    agent_optimiser = optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    # Set up discriminator
    discriminator = GAILDiscriminator((9) + (1 if args.absorbing else 0),
                                      4, args.hidden_size, state_only=args.state_only)
    discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=args.learning_rate)

    # Metrics
    writer = SummaryWriter()

    # Init
    state = env.reset(seed=0)
    episode_return = 0
    episodes = 0
    trajectories = []
    policy_trajectory_replay_buffer = deque(maxlen=args.imitation_replay_size)

    # Start training
    for step in range(args.steps):
        # print(np.shape(state[0]['rgb']))
        # print("Step = ", step)
        # print(state[0]['rgb'].shape)
        # ffsd
        # Collect set of trajectories by running policy π in the environment
        if(len(state) == 2):
            n_state = state[0]['rgb']
            # print(n_state)
        else:
            # print(state)
            # print("length == ", len(state))
            n_state = state['rgb']
            # print(n_state)
        policy, value = agent(state)
        # print(policy)
        action = policy.sample()
        # print(np.shape(action))
        log_prob_action, entropy = policy.log_prob(action), policy.entropy()
        # print(entropy)
        # print(log_prob_action.shape, action.shape)
        next_state, reward, terminal, _, _ = env.step(action)
        episode_return += reward
        trajectories.append(dict(states=n_state, actions=action, rewards=torch.tensor([reward], dtype=torch.float32),
                                 infos=torch.tensor([terminal], dtype=torch.float32),
                                 log_prob_actions=log_prob_action, old_log_prob_actions=log_prob_action,
                                 values=value, entropies=entropy))
        # next_state = torch.from_numpy(next_state['rgb'])
        state = next_state

        # If end episode
        if terminal:
            # Store metrics
            writer.add_scalar("Reward", episode_return, episodes)
            print('episode: {}, total step: {}, last_episode_reward: {}'.format(episodes+1, step+1, episode_return))

            # Reset the environment
            state, episode_return = env.reset(seed=0), 0
            # print(np.shape(state))
            if len(trajectories) >= args.batch_size:
                policy_trajectories = flatten_list_dicts(trajectories)
                trajectories = []  # Clear the set of trajectories

                # Use a replay buffer of previous trajectories to prevent overfitting to current policy
                policy_trajectory_replay_buffer.append(policy_trajectories)
                policy_trajectory_replays = flatten_list_dicts(policy_trajectory_replay_buffer)
                # print(np.shape(policy_trajectory_replays))
                # Train discriminator and predict rewards
                for _ in tqdm(range(args.imitation_epochs), leave=False):
                    adversarial_imitation_update(discriminator, expert_trajectories,
                                                 list(policy_trajectory_replays),
                                                 discriminator_optimiser, args.imitation_batch_size,
                                                 args.absorbing, args.r1_reg_coeff)
                states = policy_trajectories['states']
                actions = policy_trajectories['actions']
                # print(np.shape(policy_trajectories['states'][1:]))
                # print(torch.from_numpy(next_state['rgb']).unsqueeze(0))
                policy_traj = convert_traj(policy_trajectories['states'][1:])

                next_states = torch.cat([policy_traj, torch.from_numpy(next_state['rgb']).unsqueeze(0)])
                terminals = policy_trajectories['infos']

                if args.absorbing:
                    states, actions, next_states = indicate_absorbing(states, actions,
                                                                      policy_trajectories['infos'], next_states)
                with torch.no_grad():
                    policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)

                # Compute rewards-to-go R and generalised advantage estimates ψ based on the current value function V
                print("Computing Advantages....")
                compute_advantages(policy_trajectories, agent(state)[1], args.discount, args.trace_decay)
                print("Done computing Advantages....")
                # Normalise advantages
                policy_trajectories['advantages'] = (policy_trajectories['advantages'] - policy_trajectories[
                    'advantages'].mean()) / (policy_trajectories['advantages'].std() + 1e-8)
                
                policy_trajectories['states'] = convert_traj(policy_trajectories['states'])
                policy_trajectories['actions'] = convert_traj(policy_trajectories['actions'])
                policy_trajectories['log_prob_actions'] = convert_traj(policy_trajectories['log_prob_actions'])
                policy_trajectories['entropies'] = convert_traj(policy_trajectories['entropies'])
                policy_trajectories['values'] = convert_traj(policy_trajectories['values'])
                policy_trajectories['old_log_prob_actions'] = convert_traj(policy_trajectories['old_log_prob_actions'])
                policy_trajectories['rewards_to_go'] = convert_traj(policy_trajectories['rewards_to_go'])
                print("Done converting trajectories....")

                # Perform PPO updates using the rewards given by the discriminator
                for epoch in tqdm(range(args.ppo_epochs), leave=False):
                    ppo_update(agent, policy_trajectories, agent_optimiser, args.ppo_clip, epoch, args.value_loss_coeff,
                               args.entropy_loss_coeff)
            episodes += 1

    writer.flush()
    writer.close()

    return agent
