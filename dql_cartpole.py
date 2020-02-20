import math
import random
from collections import namedtuple
from itertools import count

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

# Define replay buffer
Transition = namedtuple('Transition', ('s','a','r','s_prime'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, s, a, r, s_prime):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # make space if needed
        self.buffer[self.position] = Transition(s,a,r,s_prime)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        return Transition(*zip(*sample))
    def __len__(self):
        return len(self.buffer)

# Define Deep Q Network
class DQN(nn.Module):
    def __init__(self, h, w, outputs, kernel_size=5, stride=2):
        super(DQN, self).__init__()
        # Define NN architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size, stride)
        self.bn3 = nn.BatchNorm2d(32)
        # Util to compute the CNN out size
        def cnn_out_size(size):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        conv_w = cnn_out_size(cnn_out_size(cnn_out_size(w)))
        conv_h = cnn_out_size(cnn_out_size(cnn_out_size(h)))
        # Output layer
        self.out = nn.Linear(conv_w * conv_h * 32, outputs)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.out(x.view(x.size(0), -1))

# Define a feature transformer
class FeatureTransformer:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.__resize = T.Compose([
            T.ToPILImage(),
            T.Resize(40, interpolation=Image.CUBIC),
            T.ToTensor()])
    def __get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)
    def get_screen(self):
        # Make PyTorch happy (C,H,W)
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Remove top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        # Remove the edges, so that we have a square image centered on a cart
        view_width = int(screen_width * 0.6)
        cart_location = self.__get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(
                cart_location - view_width // 2, cart_location + view_width // 2)
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.__resize(screen).unsqueeze(0).to(self.device)

# Define the agent
class DQLAgent:
    def __init__(self, 
                 env,
                 device,
                 batch_size=128,
                 gamma=0.999,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200,
                 replay_buffer=ReplayBuffer(10000)):
        # Init attributes
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.replay_buffer = replay_buffer
        self.device = device
        # Init steps counter
        self.steps_done = 0
        # Figure out screen height and width
        self.env.reset() # need to get a screen
        self.ft = FeatureTransformer(env, device)
        screen = self.ft.get_screen()
        _, _, screen_height, screen_width = screen.shape
        # Figure out number of actions
        self.n_actions = env.action_space.n
        # Init main and target networks
        self.main_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        # Init optimizer
        self.optimizer = optim.RMSprop(self.main_net.parameters())
    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.main_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device, dtype=torch.long)
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = self.replay_buffer.sample(self.batch_size)
        is_not_final = torch.tensor(
            tuple(map(lambda s: s is not None, minibatch.s_prime)), 
            device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in minibatch.s_prime if s is not None])
        state_batch = torch.cat(minibatch.s)
        action_batch = torch.cat(minibatch.a)
        reward_batch = torch.cat(minibatch.r)
        # Compute Q(s, a), gather selects the actions
        state_action_values = self.main_net(state_batch).gather(1, action_batch)
        # Compute V(s')
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[is_not_final] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute G
        exp_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, exp_state_action_values.unsqueeze(1))
        # Train for this batch
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def update_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

def play_one(env, agent, device):
    # Init
    ft = agent.ft # feature transformer
    env.reset()
    last_screen = ft.get_screen()
    current_screen = ft.get_screen()
    state = current_screen - last_screen

    # Play!
    for t in count():
        # Select and perform an action
        action = agent.select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        # Observe new state
        last_screen = current_screen
        current_screen = ft.get_screen()
        if not done:
            state_prime = current_screen - last_screen
        else:
            state_prime = None
        # Store trainsition in replay buffer
        agent.replay_buffer.push(state, action, reward, state_prime)
        # Move to the next state
        state = state_prime
        # Train
        agent.train()
        # Exit if done
        if done:
            return t + 1
    

if __name__ == '__main__':
    
    # Init
    N=400
    TARGET_UPDATE=10
    device=torch.device("cpu")
    env = gym.make('CartPole-v0').unwrapped
    agent = DQLAgent(env, device)
    episode_durations = np.empty(N)

    # Main loop
    for i in range(N):
        # Play episode
        episode_durations[i] = play_one(env, agent, device)        
        # Update target network
        if i % TARGET_UPDATE == 0:
            agent.update_target_net()
        # Print episode stats
        running_avg = episode_durations[max(0, i-100):(i+1)].mean()
        print('episode:', i+1, 'duration:', episode_durations[i], 'run avg:', running_avg)

    # Plot durations
    plt.plot(episode_durations)
    # Plot running avg
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = episode_durations[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.savefig('plot/dql_cartpole.png')

    # Save video for 1 episode
    wrp = gym.wrappers.Monitor(env, "video/dql_cartpole", force=True)
    play_one(wrp, agent, device)
    env.close()
    wrp.close()