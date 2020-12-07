import torch
from torch import device
import time
from utils import *


class Play:
    def __init__(self, env, agent, params, max_episode=4):
        self.env = env
        self.params = params
        self.env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        stacked_states = np.zeros(shape=self.params["state_shape"], dtype=np.uint8)
        total_reward = 0
        print("--------Play mode--------")
        for _ in range(self.max_episode):
            done = 0
            state = self.env.reset()
            episode_reward = 0
            stacked_states = stack_states(stacked_states, state, True)

            while not done:
                stacked_frames_copy = stacked_states.copy()
                action = self.agent.choose_action(stacked_frames_copy, do_greedy=False)
                next_state, r, done, _ = self.env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                self.env.render()
                time.sleep(0.01)
                episode_reward += r
            total_reward += episode_reward

        print("Total episode reward:", total_reward / self.max_episode)
        self.env.close()
        cv2.destroyAllWindows()
