import argparse
import torch
from GAIL.environments import CartPoleEnv
# from evaluation import evaluate_agent
from GAIL.models import ActorCritic
from GAIL.gail import train_g


# def get_args():
#     parser = argparse.ArgumentParser(description='IL')
#     parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
#     parser.add_argument('--steps', type=int, default=200000, metavar='T', help='Number of environment steps')
#     parser.add_argument('--hidden-size', type=int, default=32, metavar='H', help='Hidden size')
#     parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount')
#     parser.add_argument('--trace-decay', type=float, default=0.95, metavar='λ', help='GAE trace decay')
#     parser.add_argument('--ppo-clip', type=float, default=0.2, metavar='ε', help='PPO clip ratio')
#     parser.add_argument('--ppo-epochs', type=int, default=4, metavar='K', help='PPO epochs')
#     parser.add_argument('--value-loss-coeff', type=float, default=0.5, metavar='c1', help='Value loss coefficient')
#     parser.add_argument('--entropy-loss-coeff', type=float, default=0, metavar='c2',
#                         help='Entropy regularisation coefficient')
#     parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
#     parser.add_argument('--batch-size', type=int, default=2048, metavar='B', help='Minibatch size')
#     parser.add_argument('--max-grad-norm', type=float, default=1, metavar='N', help='Maximum gradient L2 norm')
#     parser.add_argument('--state-only', action='store_true', default=False, help='State-only imitation learning')
#     parser.add_argument('--absorbing', action='store_true', default=False, help='Indicate absorbing states')
#     parser.add_argument('--imitation-epochs', type=int, default=5, metavar='IE', help='Imitation learning epochs')
#     parser.add_argument('--imitation-batch-size', type=int, default=128, metavar='IB',
#                         help='Imitation learning minibatch size')
#     parser.add_argument('--imitation-replay-size', type=int, default=4, metavar='IRS',
#                         help='Imitation learning trajectory replay size')
#     parser.add_argument('--r1-reg-coeff', type=float, default=1, metavar='γ', help='R1 gradient regularisation coefficient')
#     parser.add_argument('--weights_path', type=str, default='models/GAIL/cart_pole.pth')
#     parser.add_argument('--evaluation_episodes', type=int, default=100)
#     parser.add_argument('--expert_trajectories_path', type=str, default='PPO/trajectories/cart_pole.pickle')

#     parser.add_argument('--train', action='store_true', default=False)
#     parser.add_argument('--load', action='store_true', default=False)
#     parser.add_argument('--eval', action='store_true', default=False)
#     parser.add_argument('--save', action='store_true', default=False)
#     return parser.parse_args()


# # python main.py --train --save
# # python main.py --load --eval
# if __name__ == "__main__":
#     args = get_args()
#     # Set up environment
#     env = CartPoleEnv()
#     # env.reset()
#     env.reset(seed=args.seed)
#     torch.manual_seed(args.seed)
#     # Set up actor-critic model
#     agent = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.hidden_size)

#     if args.load:
#         agent = torch.load(args.weights_path)
#         agent.eval()
#     if args.train:
#         agent = train(args, agent, env)
#     if args.eval:
#         # evaluate_agent(agent, env, args.evaluation_episodes)
#         pass
#     if args.save:
#         assert args.weights_path is not None
#         torch.save(agent, args.weights_path)
#         print("model saved to", args.weights_path)
#     env.close()


##########################################################################################################################################


import os
import numpy as np 
import pickle


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
from pathlib import Path

# Required to load inference module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import warnings
from datetime import datetime
from itertools import cycle, islice
from typing import Any, Dict

import gymnasium as gym

# Load inference module to register agent
import inference
import stable_baselines3 as sb3lib
import torch as th
import yaml
from contrib_policy import network
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from eval_policy import evaluate_policy
from torchinfo import summary
from train.env import make_env
from train.utils import ObjDict

from smarts.zoo import registry
from smarts.zoo.agent_spec import AgentSpec

print("\n")
print(f"Torch cuda is available: {th.cuda.is_available()}")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Torch device: {device}")
print("\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

# pytype: disable=attribute-error


def load_config():
    """Load config file."""
    parent_dir = Path(__file__).resolve().parent
    config_file = yaml.safe_load((parent_dir / "config.yaml").read_text())
    config = ObjDict(config_file["smarts"])
    return config


def main(args: argparse.Namespace):
    parent_dir = Path(__file__).resolve().parent
    config = load_config()

    # Load env config.
    config.mode = args.mode
    config.head = args.head
    config.expert_trajectories_path = args.expert_trajectories_path
    config.learning_rate = args.learning_rate
    config.absorbing = args.absorbing
    config.hidden_size = args.hidden_size
    config.state_only = args.state_only
    config.imitation_replay_size =  args.imitation_replay_size
    config.steps = args.steps
    config.batch_size = args.batch_size
    config.imitation_epochs = args.imitation_epochs
    config.imitation_batch_size = args.imitation_batch_size
    config.r1_reg_coeff = args.r1_reg_coeff
    config.discount = args.discount
    config.trace_decay = args.trace_decay
    config.ppo_epochs = args.ppo_epochs
    config.ppo_clip = args.ppo_clip
    config.value_loss_coeff = args.value_loss_coeff
    config.entropy_loss_coeff = args.entropy_loss_coeff

    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        logdir = parent_dir / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = logdir
    print("\nLogdir:", logdir, "\n")

    # Setup model.
    if config.mode == "evaluate":
        # Begin evaluation.
        config.model = args.model
        print("\nModel:", config.model, "\n")
    elif config.mode == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f"Expected 'train' or 'evaluate', but got {config.mode}.")

    # Make agent specification
    agent_spec = registry.make(locator=config.agent_locator)

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    for scenario in config.scenarios:
        scenario_path = str(Path(__file__).resolve().parents[3] / scenario)
        envs_train[f"{scenario}"] = make_env(
            env_id=config.env_id,
            scenario=scenario_path,
            agent_spec=agent_spec,
            config=config,
            seed=config.seed,
        )
        envs_eval[f"{scenario}"] = make_env(
            env_id=config.env_id,
            scenario=scenario_path,
            agent_spec=agent_spec,
            config=config,
            seed=config.seed,
        )

    # Run training or evaluation.
    if config.mode == "train":
        train(
            envs_train=envs_train,
            envs_eval=envs_eval,
            config=config,
            agent_spec=agent_spec,
        )
    else:
        evaluate(envs=envs_eval, config=config, agent_spec=agent_spec)

    # Close all environments
    for env in envs_train.values():
        env.close()
    for env in envs_eval.values():
        env.close()


def train(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
    agent_spec: AgentSpec,
):
    print("\nStart training.\n")
    save_dir = config.logdir / "train"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=config.logdir / "checkpoint",
        name_prefix="PPO",
    )

    scenarios_iter = islice(cycle(config.scenarios), config.epochs)
    for index, scen in enumerate(scenarios_iter):
        env_train = envs_train[scen]
        env_eval = envs_eval[scen]
        state = env_train.reset(seed=0)
        agent = agent_spec.build_agent()
        # print(agent)

        print(f"\nTraining on {scen}.\n")

        # Set up actor-critic model
        # print(np.shape(state[1]['env_obs'].top_down_rgb.data))
        # # agent = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.hidden_size)
        # dfgdfgdsf

        #############################################################
        ### GAIL training 
        #############################################################

        agent = ActorCritic(np.shape(state[0]['rgb']), 4, args.hidden_size)

        # if index == 0:
        #     model = sb3lib.PPO(
        #         env=env_train,
        #         tensorboard_log=config.logdir / "tensorboard",
        #         verbose=1,
        #         **network.combined_extractor(config),
        #     )
        # else:
        #     model = sb3lib.PPO.load(save_dir / "intermediate")

        # print(model.policy.state_dict())

        # eval_callback = EvalCallback(
        #     env_eval,
        #     best_model_save_path=config.logdir / "eval",
        #     n_eval_episodes=3,
        #     eval_freq=config.eval_freq,
        #     deterministic=True,
        #     render=False,
        #     verbose=1,
        # )
        # model.set_env(env_train)
        # model.learn(
        #     total_timesteps=config.train_steps,
        #     callback=[checkpoint_callback, eval_callback],
        #     reset_num_timesteps=False,
        # )
        model = train_g(args, agent, env_train)
        model.save(save_dir / "intermediate")


    print("Finished training.")

    # Save trained model.
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model.save(save_dir / ("model_" + time))
    print("\nSaved trained model.\n")

    # Print model summary
    crop = agent_spec.agent_params["crop"]
    top_down_rgb = agent_spec.interface.top_down_rgb
    h = top_down_rgb.height - crop[2] - crop[3]
    w = top_down_rgb.width - crop[0] - crop[1]
    td = {"rgb": th.zeros(1, 9, h, w)}
    summary(model.policy, input_data=[td], depth=5)


def evaluate(
    envs: Dict[str, gym.Env],
    config: Dict[str, Any],
    agent_spec: AgentSpec,
):
    print("\nEvaluate policy.\n")
    device = th.device("cpu")
    model = sb3lib.PPO.load(config.model, print_system_info=True, device=device)

    # Print model summary
    crop = agent_spec.agent_params["crop"]
    top_down_rgb = agent_spec.interface.top_down_rgb
    h = top_down_rgb.height - crop[2] - crop[3]
    w = top_down_rgb.width - crop[0] - crop[1]
    td = {"rgb": th.zeros(1, 9, h, w)}
    summary(model.policy, input_data=[td], depth=5)

    for env_name, env_eval in envs.items():
        print(f"\nEvaluating env {env_name}.")
        mean_reward, std_reward, traj = evaluate_policy(
            model, env_eval, n_eval_episodes=config.eval_eps, deterministic=True
        )
        print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")
        # print(len(traj))
        with open('traj.pickle', 'wb') as handle:
            pickle.dump(traj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nFinished evaluating.\n")

if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path for saving logs.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head", help="Display the simulation in Envision.", action="store_true"
    )
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--steps', type=int, default=200000, metavar='T', help='Number of environment steps')
    parser.add_argument('--hidden-size', type=int, default=384, metavar='H', help='Hidden size')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount')
    parser.add_argument('--trace-decay', type=float, default=0.95, metavar='λ', help='GAE trace decay')
    parser.add_argument('--ppo-clip', type=float, default=0.2, metavar='ε', help='PPO clip ratio')
    parser.add_argument('--ppo-epochs', type=int, default=4, metavar='K', help='PPO epochs')
    parser.add_argument('--value-loss-coeff', type=float, default=0.5, metavar='c1', help='Value loss coefficient')
    parser.add_argument('--entropy-loss-coeff', type=float, default=0, metavar='c2',
                        help='Entropy regularisation coefficient')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, metavar='B', help='Minibatch size')
    parser.add_argument('--max-grad-norm', type=float, default=1, metavar='N', help='Maximum gradient L2 norm')
    parser.add_argument('--state-only', action='store_true', default=True, help='State-only imitation learning')
    parser.add_argument('--absorbing', action='store_true', default=False, help='Indicate absorbing states')
    parser.add_argument('--imitation-epochs', type=int, default=5, metavar='IE', help='Imitation learning epochs')
    parser.add_argument('--imitation-batch-size', type=int, default=64, metavar='IB',
                        help='Imitation learning minibatch size')
    parser.add_argument('--imitation-replay-size', type=int, default=4, metavar='IRS',
                        help='Imitation learning trajectory replay size')
    parser.add_argument('--r1-reg-coeff', type=float, default=1, metavar='γ', help='R1 gradient regularisation coefficient')
    parser.add_argument('--weights_path', type=str, default='')
    parser.add_argument('--evaluation_episodes', type=int, default=100)
    parser.add_argument('--expert_trajectories_path', type=str, default='')

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)
