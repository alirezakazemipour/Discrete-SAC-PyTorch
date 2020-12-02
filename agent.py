import numpy as np
from model import PolicyNetwork, QValueNetwork
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn import functional as F


class SAC:
    def __init__(self, state_shape, n_actions, memory_size, batch_size, gamma, alpha, lr, action_bounds, reward_scale):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.action_bounds = action_bounds
        self.reward_scale = reward_scale
        self.memory = Memory(memory_size=self.memory_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_network = PolicyNetwork(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.q_value_network1 = QValueNetwork(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.q_value_network2 = QValueNetwork(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.q_value_target_network1 = QValueNetwork(state_shape=self.state_shape,
                                                     n_actions=self.n_actions).to(self.device)
        self.q_value_target_network2 = QValueNetwork(state_shape=self.state_shape,
                                                     n_actions=self.n_actions).to(self.device)

        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()

        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

        self.entropy_target = 0.98 * (-np.log(1 / self.n_actions))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)
        self.alpha_opt = Adam([self.log_alpha], lr=self.lr)

        self.update_counter = 0

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, reward, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)

        return states, rewards, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states = self.unpack(batch)

            # Calculating the Q-Value target
            with torch.no_grad():
                dist = self.policy_network.sample_or_likelihood(next_states)
                next_actions = dist.sample()
                next_log_probs = dist.log_prob(next_actions)
                next_q1 = self.q_value_target_network1(next_states).gather(1, next_actions)
                next_q2 = self.q_value_target_network2(next_states).gather(1, next_actions)
                next_q = torch.min(next_q1, next_q2)
                target_q = self.reward_scale * rewards + \
                           self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)

            q1 = self.q_value_network1(states).gather(1, actions)
            q2 = self.q_value_network2(states).gather(1, actions)
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)

            # Calculating the Policy target
            dist = self.policy_network(states)
            log_probs = dist.log_prob(actions)
            # with torch.no_grad():
            q1 = self.q_value_network1(states).gather(1, actions)
            q2 = self.q_value_network2(states).gather(1, actions)
            q = torch.min(q1, q2)

            policy_loss = ((self.alpha * log_probs) - q).mean()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            alpha_loss = -(self.log_alpha * (log_probs + self.entropy_target).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.update_counter += 1

            self.alpha = self.log_alpha.exp()

            if self.update_counter % 8000 == 0:
                self.hard_update_target_network(self.q_value_network1, self.q_value_target_network1)
                self.q_value_target_network1.eval()
                self.hard_update_target_network(self.q_value_network2, self.q_value_target_network2)
                self.q_value_target_network2.eval()

            return alpha_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def hard_update_target_network(local_network, target_network):
        target_network.load_state_dict(local_network.state_dict())

    def save_weights(self):
        torch.save(self.policy_network.state_dict(), "./weights.pth")

    def load_weights(self):
        self.policy_network.load_state_dict(torch.load("./weights.pth"))

    def set_to_eval_mode(self):
        self.policy_network.eval()
