from Brain.agent import SAC
import time
from Common.play import Play
from Common.utils import *
from Common.logger import Logger
from Common.config import get_params


def intro_env():
    for e in range(5):
        test_env.reset()
        d = False
        ep_r = 0
        while not d:
            a = test_env.env.action_space.sample()
            _, r, d, info = test_env.step(a)
            ep_r += r
            test_env.env.render()
            time.sleep(0.005)
            print(f"reward: {np.sign(r)}")
            print(info)
            if d:
                break
        print("episode reward: ", ep_r)
    test_env.close()
    exit(0)


if __name__ == "__main__":
    params = get_params()

    test_env = make_atari(params["env_name"], episodic_life=False)
    params.update({"n_actions": test_env.action_space.n})

    print(f"Number of actions: {params['n_actions']}")

    if params["do_intro_env"]:
        intro_env()

    env = make_atari(params["env_name"], episodic_life=False)

    agent = SAC(**params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode = logger.load_weights()
            agent.hard_update_target_network()
            agent.alpha = agent.log_alpha.exp()
            min_episode = episode
            print("Keep training from previous run.")

        else:
            min_episode = 0
            print("Train from scratch.")

        stacked_states = np.zeros(shape=params["state_shape"], dtype=np.uint8)
        state = env.reset()
        stacked_states = stack_states(stacked_states, state, True)
        episode_reward = 0
        alpha_loss, q_loss, policy_loss = 0, 0, 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, params["max_steps"] + 1):
            if step < params['initial_random_steps']:
                stacked_states_copy = stacked_states.copy()
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                reward = np.sign(reward)
                agent.store(stacked_states_copy, action, reward, stacked_states, done)
                if done:
                    state = env.reset()
                    stacked_states = stack_states(stacked_states, state, True)
            else:
                stacked_states_copy = stacked_states.copy()
                action = agent.choose_action(stacked_states_copy)
                next_state, reward, done, _ = env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                reward = np.sign(reward)
                agent.store(stacked_states_copy, action, reward, stacked_states, done)
                episode_reward += reward
                state = next_state

                if step % params["train_period"] == 0:
                    alpha_loss, q_loss, policy_loss = agent.train()

                if done:
                    logger.off()
                    logger.log(episode, episode_reward, alpha_loss, q_loss, policy_loss , step)

                    episode += 1
                    obs = env.reset()
                    state = stack_states(state, obs, True)
                    episode_reward = 0
                    episode_loss = 0
                    logger.on()

    logger.load_weights()
    player = Play(env, agent, params)
    player.evaluate()
