import gym
from agent import SAC
import time
import psutil
from play import Play

ENV_NAME = "Ant-v2"
test_env = gym.make(ENV_NAME)

n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

MAX_EPISODES = 2000
max_steps = test_env._max_episode_steps
memory_size = 1e+6
batch_size = 64
gamma = 0.99
alpha = 0.2  # !!!!
lr = 3e-4
reward_scale = 1

to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
global_running_reward = 0


def log(episode, start_time, episode_reward, alpha_loss, q_loss, policy_loss, memory_length):
    global global_running_reward
    if episode == 0:
        global_running_reward = episode_reward
    else:
        global_running_reward = 0.99 * global_running_reward + 0.01 * episode_reward

    ram = psutil.virtual_memory()

    if episode % 100 == 0:
        print(f"EP:{episode}| "
              f"EP_r:{episode_reward:3.3f}| "
              f"EP_running_reward:{global_running_reward:3.3f}| "
              f"Alpha_Loss:{alpha_loss:3.3f}| "
              f"Q-Value_Loss:{q_loss:3.3f}| "
              f"Policy_Loss:{policy_loss:3.3f}| "
              f"Memory_length:{memory_length}| "
              f"Duration:{time.time() - start_time:3.3f}| "
              f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")


if __name__ == "__main__":
    print(f"Number of states:{n_states}\n"
          f"Number of actions:{n_actions}\n"
          f"Action boundaries:{action_bounds}")

    env = gym.make(ENV_NAME)
    agent = SAC(n_states=n_states,
                n_actions=n_actions,
                memory_size=memory_size,
                batch_size=batch_size,
                gamma=gamma,
                alpha=alpha,
                lr=lr,
                action_bounds=action_bounds,
                reward_scale=reward_scale)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = 0

        start_time = time.time()
        for step in range(1, 1 + max_steps):
            # if episode < 50:
            #     action = env.action_space.sample()
            # else:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, reward, done, action, next_state)
            if step % 4 == 0:
                alpha_loss, q_loss, policy_loss = agent.train()
            # if episode % 300 == 0:
            #     agent.save_weights()
            if done:
                break
            episode_reward += reward
            state = next_state
        log(episode, start_time, episode_reward, alpha_loss, q_loss, policy_loss, len(agent.memory))

    agent.save_weights()
    player = Play(env, agent)
    player.evaluate()
