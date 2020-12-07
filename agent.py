import numpy as np
from model import PolicyNetwork, QValueNetwork
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn import functional as F


class SAC:
    def __init__(self, **config):
        self.config = config
        self.state_shape = self.config["state_shape"]
        self.n_actions = self.config["n_actions"]
        self.lr = self.config["lr"]
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.memory = Memory(memory_size=self.config["mem_size"])

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
        self.alpha = self.log_alpha.exp()

        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)
        self.alpha_opt = Adam([self.log_alpha], lr=self.lr)

        self.update_counter = 0

    def store(self, state, action, reward, next_state, done):
        state = from_numpy(state).byte().to("cpu")
        reward = torch.CharTensor([reward])
        action = torch.ByteTensor([action]).to('cpu')
        next_state = from_numpy(next_state).byte().to('cpu')
        done = torch.BoolTensor([done])
        self.memory.add(state, reward, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, *self.state_shape)
        actions = torch.cat(batch.action).view((-1, 1)).long().to(self.device)
        rewards = torch.cat(batch.reward).view((-1, 1)).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, *self.state_shape)
        dones = torch.cat(batch.done).view((-1, 1)).to(self.device)

        return states, rewards, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states = self.unpack(batch)

            # Calculating the Q-Value target
            with torch.no_grad():
                _, next_probs = self.policy_network(next_states)
                next_log_probs = torch.log(next_probs)
                next_q1 = self.q_value_target_network1(next_states)
                next_q2 = self.q_value_target_network2(next_states)
                next_q = torch.min(next_q1, next_q2)
                next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
                target_q = rewards + self.gamma * (~dones) * next_v

            q1 = self.q_value_network1(states).gather(1, actions)
            q2 = self.q_value_network2(states).gather(1, actions)
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)

            # Calculating the Policy target
            _, probs = self.policy_network(states)
            log_probs = torch.log(probs)
            with torch.no_grad():
                q1 = self.q_value_network1(states)
                q2 = self.q_value_network2(states)
                q = torch.min(q1, q2)

            policy_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).mean()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            log_probs = (probs * log_probs).sum(-1)
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.update_counter += 1

            self.alpha = self.log_alpha.exp()

            if self.update_counter % self.config["fixed_network_update_freq"] == 0:
                self.hard_update_target_network()

            return alpha_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def choose_action(self, states, do_greedy=False):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).byte().to(self.device)
        with torch.no_grad():
            dist, p = self.policy_network(states)
            if do_greedy:
                action = p.argmax(-1)
            else:
                action = dist.sample()
        return action.detach().cpu().numpy()[0]

    def hard_update_target_network(self):
        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()
        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

    def set_to_eval_mode(self):
        self.policy_network.eval()
