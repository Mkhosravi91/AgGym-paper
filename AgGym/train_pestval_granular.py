from sqlite3 import Timestamp
from tokenize import Double
from modules import env_modules_pestval_granular
from utils import general_utils as gu
from configobj import ConfigObj
import pfrl
from pfrl.agents import dqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple
from typing import List
import torch
from torch import nn
import gym
import numpy as np
from einops import rearrange
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import time
import os
import random
import argparse
import pdb
# pdb.set_trace()

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--seed', type=int, default=123, help='Description for foo argument')
args = parser.parse_args()

class QFunction(nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = nn.Linear(obs_size, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, obs_size*4)

    def forward(self, x):
        # h = rearrange(x, "b h w -> b (h w)")
        h = nn.functional.relu(self.l1(x))
        h = nn.functional.relu(self.l2(h))
        h = self.l3(h)
        # h_reshaped = h.view(-1, 10, 10, n_actions)
        # action_indices = h_reshaped.argmax(dim=-1).view(-1, 10*10)
        
        # Return Q-values for the best actions for each cell
        # return pfrl.action_value.DiscreteActionValue(action_indices)
        print(h.shape)
        return pfrl.action_value.DiscreteActionValue(h)
        print(pfrl.action_value.DiscreteActionValue(h))

class CustomQNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(CustomQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Ensure the desired output shape
        self.fc3 = nn.Linear(64, obs_size*n_actions)

    def forward(self, x):
        x = nn.Tanh()(self.fc1(x))
        x = nn.Tanh()(self.fc2(x))
        x = self.fc3(x)
        return x
class CustomDoubleDQN(pfrl.agents.DoubleDQN):
    
    def act(self, obs):
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            q_values = self.model(obs_tensor)

            # q_values = self.model(np.array(obs, dtype=np.float32))
            # Reshape to (5, 5, 4)
            reshaped_q_values = q_values.reshape(obs_size, 4)
            # Take the argmax over the last dimension for each cell
            action = super().act(obs)
            action = np.argmax(reshaped_q_values, axis=1)
            # Flatten back to a single vector for final action
            # action = action_indices.flatten()
            return action
    
    def _compute_target_values(self, exp_batch):
        # The rest of this method remains largely unchanged, but ensure that when you compute Q-values
        # from your model, you handle the reshaping properly
          # Implement based on the original function
        return super()._compute_target_values(exp_batch)




# class CustomPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs, 
#                                             net_arch=[64, 64, obs_size],  # This defines the architecture of the Policy & Value networks
#                                             activation_fn=nn.Tanh)

#     def _build_mlp_extractor(self) -> None:
#         self.mlp_extractor = CustomMLP(self.features_dim, self.net_arch, self.activation_fn)

# class CustomMLP(nn.Module):
#     def __init__(self, feature_dim: int, net_arch: list, activation_fn: nn.Module):
#         super(CustomMLP, self).__init__()
#         self.net_arch = net_arch
#         self.activation_fn = activation_fn
#         self.feature_dim = feature_dim
    
#         # Setup policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(self.feature_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, n_actions*obs_size),  # Adjust as needed
#             nn.Softmax(dim=-1)
#         )
        
#         # Setup value network
#         self.value_net = nn.Sequential(
#             nn.Linear(self.feature_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1),  # Adjust as needed
#         )
        
#         # Initialize weights
#         def ortho_init(layer, gain):
#             nn.init.orthogonal_(layer.weight, gain=gain)
#             nn.init.zeros_(layer.bias)
        
#         ortho_init(self.policy_net[0], gain=1)
#         ortho_init(self.policy_net[2], gain=1)
#         ortho_init(self.policy_net[4], gain=1e-2)
#         ortho_init(self.value_net[0], gain=1)
#         ortho_init(self.value_net[2], gain=1)
#         ortho_init(self.value_net[4], gain=1e-2)
    

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         return self.policy_net(obs), self.value_net(obs)
class CustomMLPExtractor(nn.Module):
    def __init__(self, obs_size: int, net_arch: List[int]):
        super(CustomMLPExtractor, self).__init__()
        
        # Assuming observation_space is flat and gives the number of features
        input_dim = obs_size
        
        # Shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # These dimensions are essential for Stable Baselines3 to understand the size of the output
        # from the shared layers
        self.latent_dim_pi = 64  # This is the size of the output from your actor's MLP
        self.latent_dim_vf = 64  # This is the size of the output from your critic's MLP

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shared layers processing
        shared_latents = self.shared_net(observations)
        return shared_latents, shared_latents  # For both actor and critic
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMLPExtractor(obs_size, self.net_arch)


# training_rewards=np.zeros((10, 10000))
# for x in range(10):

class GridPolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(GridPolicyNetwork, self).__init__()
        
        self.fc1 = lecun_init(nn.Linear(obs_size, 128))
        self.fc2 = lecun_init(nn.Linear(128, 64))
        self.fc3 = lecun_init(nn.Linear(64, 32))
        
        # self.action_head = lecun_init(nn.Linear(32, 4*obs_size), 1e-2)
        self.action_head2 = lecun_init(nn.Linear(32, 284))
        self.action_head = pfrl.policies.SoftmaxCategoricalHead()
        self.value_head = lecun_init(nn.Linear(32, obs_size))
        
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x2 = nn.ReLU()(self.action_head2(x))
        x2 = x2.view(-1, n_actions, 4)
        # action_probabilities = torch.stack([action_head(row) for row in action_logits])
        # action_logits = self.action_head(x2).view(-1, n_actions, 4) # reshaping it to get logits for each cell
        action_logits = self.action_head(x2)
        value = self.value_head(x)
        
#     policy = pfrl.nn.Branched(   nn.Sequential(
#         model.action_head,
#         action_distrib
#     ),
#     model.value_head
# )
        
        return action_logits, value
    

cwd = Path.cwd()
now = datetime.now()
now_str = now.strftime("%d-%m-%y_%H-%M-%S")
(cwd / 'newresults' / now_str).mkdir(parents=True, exist_ok=True)
(cwd / 'newresults' / now_str / 'data').mkdir(parents=True, exist_ok=True)
(cwd / 'newresults' / now_str / 'eval').mkdir(parents=True, exist_ok=True)
(cwd / 'newresults' / now_str / 'agent_weights').mkdir(parents=True, exist_ok=True)

gu.seed_everything(args.seed)
config = ConfigObj('training_config.ini')
config['general']['result_path'] = str(cwd / 'newresults' / now_str)
logging.basicConfig(filename=config['general']['result_path'] + '/train.log', 
                    level=eval(f"logging.{config['general']['level']}"), 
                    filemode='w', 
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info("Training")
logging.info(f"Seed: {args.seed}")
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
branch = branch.decode('utf-8').replace('\n','')
commit = subprocess.check_output(["git", "rev-parse", "HEAD"])
commit = commit.decode('utf-8').replace('\n','')
logging.info(f"Github Branch: {branch}")
logging.info(f"Github Commit: {commit}")
gu.print_to_logs(config)

env = env_modules_pestval_granular.cartesian_grid()
gu.set_as_attr(env, config)
env.init()
# pdb.set_trace()
env.rl_init()

obs_size = env.state_space
# n_actions = env.action_space.n
n_actions = len(env.action_space.nvec)
print(f"nactions:{n_actions}")
env.gpu = int(env.gpu)
print(f"Model: {env.model}, GPU: {env.gpu}")

def random_vector_action():
    action = np.random.choice(4, obs_size)
    print(f"Random action shape: {action.shape}")
    return action
# def random_vector_action():
#     # 10x10 grid with an action between 0 and 3 for each cell
#     return np.random.randint(0, 4, (10, 10))
# def random_vector_action():
#     # Assuming n_actions for each of the 100 cells
#     return np.random.randint(n_actions, size=(10, 10)).flatten()
if env.model == 'ddqn':
    q_func = CustomQNetwork(obs_size, n_actions)
    optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
    gamma = 0.9


    # q_func = QFunction(obs_size, n_actions)

    # optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
    # gamma = 0.99

    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=0.95, end_epsilon=0.1, decay_steps=350000, random_action_func=random_vector_action)

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    gpu = env.gpu
    agent = CustomDoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=20000,
    update_interval=130,
    target_update_interval=1300,
    phi=phi,
    gpu=gpu,
)
    # agent = pfrl.agents.DoubleDQN(
    #     q_func,
    #     optimizer,
    #     replay_buffer,
    #     gamma,
    #     explorer,
    #     replay_start_size=20000,
    #     update_interval=130,
    #     target_update_interval=1300,
    #     phi=phi,
    #     gpu=gpu,
    # )
elif env.model == 'ppo':
    
    # agent = PPO(CustomPolicy, env, policy_kwargs={'net_arch': [64, 64]}, verbose=1)
    agent = PPO(policy = "MlpPolicy",env =  env, verbose=1)
    agent.learn(total_timesteps=115000)
    # agent = PPO(CustomPolicy, env, verbose=1)

    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer
    # model = GridPolicyNetwork(obs_size, 4)
    
    # policy = pfrl.nn.Branched(
    # nn.Sequential(
    #     model.action_head,
    #     pfrl.policies.SoftmaxCategoricalHead()
    # ),
#     model.value_head
# )
    
    # model = nn.Sequential(
    #         lecun_init(nn.Linear(obs_size, 128)),
    #         nn.ReLU(),
    #         lecun_init(nn.Linear(128, 64)),
    #         nn.ReLU(),
    #         lecun_init(nn.Linear(64, 32)),
    #         nn.ReLU(),
    #         pfrl.nn.Branched(
    #             nn.Sequential(
    #                 lecun_init(nn.Linear(32, n_actions), 1e-2),
    #                 pfrl.policies.SoftmaxCategoricalHead(),
    #             ),
    #             lecun_init(nn.Linear(32, 1)),
    #         ),
    #     )
    # ###original
    # opt = torch.optim.RMSprop(model.parameters(),lr=0.0007477847290354089)
    # phi = lambda x: x.astype(np.float32, copy=False)
    # agent = pfrl.agents.PPO(
    #     model,
    #     opt,
    #     gpu=env.gpu,
    #     phi=phi,
    #     update_interval=4000,
    #     minibatch_size=512,
    #     epochs=4,
    #     clip_eps=0.2837622690867982,
    #     clip_eps_vf=None,
    #     standardize_advantages=True,
    #     entropy_coef=0.2,
    #     recurrent=False,
    #     max_grad_norm=0.5,
    # )
elif env.model == 'trpo':
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions*25),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 25),
    )
    
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1e-2)
    
    vf_opt = torch.optim.Adam(vf.parameters())
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = pfrl.agents.TRPO(
        policy=policy,
        vf=vf,
        phi=phi,
        vf_optimizer=vf_opt,
        obs_normalizer=None,
        gpu=env.gpu,
        update_interval=5000,
        max_kl=0.01,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=0.995,
        lambd=0.97,
        vf_epochs=10,
        entropy_coef=0,
    )
elif env.model == 'sac':
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
        pfrl.policies.SoftmaxCategoricalHead(),
        # print(n_actions)
    )

    critic1 = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
    )
    critic2 = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
    )
    critic1taregt = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
    )
    critic2target = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
    )
    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer
    
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    lecun_init(policy[0], gain=1)
    lecun_init(policy[2], gain=1)
    lecun_init(policy[4], gain=1e-2)
    lecun_init(critic1[0], gain=1)
    lecun_init(critic1[2], gain=1)
    lecun_init(critic1[4], gain=1e-2)
    lecun_init(critic2[0], gain=1)
    lecun_init(critic2[2], gain=1)
    lecun_init(critic2[4], gain=1e-2)
    critic1_opt = torch.optim.Adam(critic1.parameters())
    critic2_opt = torch.optim.Adam(critic2.parameters())
    policy_opt = torch.optim.Adam(policy.parameters())
    phi = lambda x: x.astype(np.float32, copy=False)
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
    gamma = 0.99
    agent = pfrl.agents.SoftActorCritic(
        policy=policy,
        q_func1=critic1,
        q_func2=critic2,
        # target_q_func1=critic1taregt,
        # target_q_func2=critic2target,
        policy_optimizer=policy_opt,
        q_func1_optimizer=critic1_opt,
        q_func2_optimizer=critic2_opt,
        gamma=gamma,
        replay_buffer=replay_buffer
        # temperature_holder=None,
        # temperature_optimizer=None

    )
#
if env.model == 'ppo':
    average_value = []
    average_entropy = []
    average_value_loss = []
    average_policy_loss = []
    n_updates = []
    explained_variance = []
#    


max_episode = config["training"].as_int("max_episode")
best_agent_reward = -99999999
total_step = 0
training_R = []
for ep in range(max_episode):
    obs, info = env.reset()
    if env.sim_mode == 'growthseason':
        end_condition = env.gs_end
        # end_condition = 5
    elif env.sim_mode == 'survival':
        end_condition = env.max_timestep
    # training_R_ep=[]
    i = 0
    R = 0
    agent_list = []
    env_list = []
    print("episod")
    while True:
        agent_start = time.time()
        print(f"step#={env.timestep}")
        action = agent.act(obs.astype(np.float32))
        # action, _ = agent.predict(obs.astype(np.float32))
        print(f"action:{action}")
        print(action.shape)
        # action = agent.act(obs.item())
        agent_end = time.time()
        agent_list.append(agent_end - agent_start)
        env_start = time.time()
        # obs, reward, done = env.step(action)
        obs, reward, done, truncated, info= env.step(action)
        # print(env.action_list)
        env_end = time.time()
        env_list.append(env_end - env_start)
        R += reward
        reset = env.timestep == end_condition
        print(type(obs[0]), type(reward))
        # pdb.set_trace()
        # agent.observe(obs.astype(np.float32), reward.astype(np.float32), done, reset)
        logging.debug(f"Step: {i}, Action: {action} Reward: {reward}, Done: {done}, {env.gs_end}")
        i += 1
        total_step += 1
        if done or reset:
            break
    logging.debug(f"Episode {ep}, Reward: {R}, Done: {done}, Total Step: {total_step}")
    logging.debug(f"Agent | Total {sum(agent_list)}, Average {np.round(np.average(agent_list), 6)}, Max: {np.round(max(agent_list), 6)}, Min: {np.round(min(agent_list), 6)} ")
    logging.debug(f"Env | Total {sum(env_list)}, Average {np.round(np.average(env_list), 6)}, Max: {np.round(max(env_list), 6)}, Min: {np.round(min(env_list), 6)} ")

    #
    if env.model == 'ppo':
        average_value.append(agent.get_statistics()[0][1])
        average_entropy.append(agent.get_statistics()[1][1])
        average_value_loss.append(agent.get_statistics()[2][1])
        average_policy_loss.append(agent.get_statistics()[3][1])
        n_updates.append(agent.get_statistics()[4][1])
        explained_variance.append(agent.get_statistics()[5][1])
    #
    
    training_R.append(R)
    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {R}, Done: {done}")
        logging.info(f"Episode {ep}, Reward: {R}, Done: {done}")
    if ep % 500 == 0:
        print(f"Agent Statistics: {agent.get_statistics()}")
        logging.info(f"Agent Statistics: {agent.get_statistics()}")

    if ep % 1000 == 0 or ep == max_episode - 1:
        print("eval")
        eval_reward = []
        with agent.eval_mode():
            for eval_ep in range(10):
                obs, info = env.reset()
                R = 0
                while True:
                    action = agent.act(obs.astype(np.float32))
                    # obs, reward, done = env.step(action)
                    obs, reward, done, truncated, info= env.step(action)
                    R += reward
                    reset = env.timestep == end_condition
                    # reset = env.timestep == 5
                    agent.observe(obs.astype(np.float32), reward, done, reset)
                    if done or reset:
                        break
                eval_reward.append(R)
                print(f"Evaluation Episode: {eval_ep}, Reward: {R}")
                logging.info(f"Evaluation Episode: {eval_ep}, Reward: {R}")
        average = np.average(eval_reward)
        print(f"Average Eval Performance: {average}")
        logging.info(f"Average Eval Performance: {average}")
        if average > best_agent_reward:
            print(f"Current agent weights is better ({average} > {best_agent_reward}), saving weights")
            logging.info(f"Current agent weights is better ({average} > {best_agent_reward}), saving weights")
            best_agent_reward = average
            agent.save(str(Path(env.result_path) / 'agent_weights' / f'agent_{ep}'))
        else:
            print(f"Previous agent weights is better, ignoring weight saving.")
            logging.info(f"Previous agent weights is better, ignoring weight saving.")       
# training_rewards[x, :]=training_R
np.save(cwd / 'newresults' / now_str / 'data' / 'training_reward', training_R)
    
# training_R_2 = np.mean(training_rewards, axis=0)
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(9,6))
sns.lineplot(x=np.arange(len(training_R)), y=training_R)
ma_av=moving_average(training_R, 50)
# ma=np.zeros((10, 10000-49))
# for i in range(10):
#     ma[i,:] = moving_average(training_rewards[i, :], 50)
#     sns.lineplot(x=np.arange(len(ma[i])), y=ma[i,:])
# MA=np.mean(ma, axis=0)
# sns.lineplot(x=np.arange(len(MA)), y=MA, linewidth=7.0)
sns.lineplot(x=np.arange(len(ma_av)), y=ma_av)
plt.savefig(cwd / 'newresults' / now_str / 'training.png', dpi=300)
plt.close()

#
if env.model == 'ppo':
    ax_dict = plt.figure(num=1, figsize=(12,12), constrained_layout=True, clear=True).subplot_mosaic(
        """
        ABC
        DEF
        """)
    sns.lineplot(x=np.arange(len(average_value)), y=average_value, ax=ax_dict["A"])
    ma = moving_average(average_value, 50)
    sns.lineplot(x=np.arange(len(ma)), y=ma, ax=ax_dict["A"])
    ax_dict["A"].set_title("Average Value")

    sns.lineplot(x=np.arange(len(average_entropy)), y=average_entropy, ax=ax_dict["B"])
    ma = moving_average(average_entropy, 50)
    sns.lineplot(x=np.arange(len(ma)), y=ma, ax=ax_dict["B"])
    ax_dict["B"].set_title("Average Entropy")

    sns.lineplot(x=np.arange(len(average_value_loss)), y=average_value_loss, ax=ax_dict["C"])
    ma = moving_average(average_value_loss, 50)
    sns.lineplot(x=np.arange(len(ma)), y=ma, ax=ax_dict["C"])
    ax_dict["C"].set_title("Average Value Loss")

    sns.lineplot(x=np.arange(len(average_policy_loss)), y=average_policy_loss, ax=ax_dict["D"])
    ma = moving_average(average_policy_loss, 50)
    sns.lineplot(x=np.arange(len(ma)), y=ma, ax=ax_dict["D"])
    ax_dict["D"].set_title("Average Policy Loss")

    sns.lineplot(x=np.arange(len(n_updates)), y=n_updates, ax=ax_dict["E"])
    ma = moving_average(n_updates, 50)
    sns.lineplot(x=np.arange(len(ma)), y=ma, ax=ax_dict["E"])
    ax_dict["E"].set_title("N Updates")

    sns.lineplot(x=np.arange(len(explained_variance)), y=explained_variance, ax=ax_dict["F"])
    ma = moving_average(explained_variance, 50)
    sns.lineplot(x=np.arange(len(ma)), y=ma, ax=ax_dict["F"])
    ax_dict["F"].set_title("Explained Variance")

    plt.savefig(cwd / 'newresults' / now_str / 'ppo_stats.png', dpi=300)
    plt.close()
#
def G(severity):
    # if 0=<severity<=0.22:
    # gyield=1/(1+np.exp((severity-0.24)/1))
    gyield=1/(1+np.exp(-(severity-0.26)))
    # else:
    return gyield

agent_list = os.listdir(Path(env.result_path) / 'agent_weights')
max_ep = 0
for i in agent_list:
    cur_ep = int(i.split('_')[1])
    if cur_ep > max_ep:
        max_ep = cur_ep

print(f"Loading best agent {max_ep}")
# breakpoint()
agent.load(str(Path(env.result_path) / 'agent_weights' / f'agent_{max_ep}'))
env.mode = 'eval'
env.best_agent = max_ep
eval_reward = []
print('eval_mode****************************************')
with agent.eval_mode():
    for ep in range(1):
        obs, info = env.reset()
        R = 0
        while True:
            print(f"step start#={env.timestep}")
            print(f"threatlist_beforepest={len(env.threat.infect_list)}")
            action = agent.act(obs.astype(np.float32))
            print(action)
            obs, reward, done, truncated, info = env.step(action)
            R += reward
            agent.observe(obs.astype(np.float32), reward, done, done)
            print(f"step end#={env.timestep}")
            if done == True:
                # dead and infected yield
                print(f"dead_count={env.dead_counts[-1]}")
                print(f"infect_count={env.infect_counts[-1]}")
                healthy_plants=env.state_space-(env.dead_counts[-1]+env.infect_counts[-1])
                # total_yield=healthy_plants+np.sum(np.array(env.Degraded_list))+env.infect_counts[-1]
                # total_yield=healthy_plants+np.sum(np.array(env.Degraded_list))+(env.infect_counts[-1])*0.6
                # total_yield=healthy_plants+np.sum(np.array(env.Degraded_list))+(env.infectdeg_list[-1])
                total_yield=healthy_plants+np.sum(np.array(env.Degraded_list))
                print("total_yield={}".format(total_yield))
                print("Degraded_list={}".format(np.sum(np.array(env.Degraded_list))))
                break
        eval_reward.append(R)
        print(f"Evaluation Episode: {ep}, Reward: {R}")
        logging.info(f"Evaluation Episode: {eval_ep}, Reward: {R}")
        
average = np.average(eval_reward)
print(f"Average Eval Performance: {average}")
logging.info(f"Average Eval Performance: q{average}")