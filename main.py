from agent import SAC
import time
from play import Play
from utils import *
from logger import Logger
from config import get_params


def intro_env():
    for e in range(5):
        test_env.reset()
        for _ in range(test_env._max_episode_steps):
            a = test_env.env.action_space.sample()
            _, r, d, info = test_env.step(a)
            test_env.env.render()
            time.sleep(0.005)
            print(f"reward: {r}")
            print(info)
            if d:
                break
    test_env.close()
    exit(0)


if __name__ == "__main__":
    params = get_params()

    test_env = make_atari(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})

    print(f"Number of actions: {params['n_actions']}")

    if params["do_intro_env"]:
        intro_env()

    env = make_atari(params["env_name"])
    env.seed(int(time.time()))

    agent = SAC(**params)
    logger = Logger(agent, **params)

    if not params["train_from_scratch"]:
        chekpoint = logger.load_weights()
        agent.policy_network.load_state_dict(chekpoint["policy_network_state_dict"])
        agent.hard_update_target_network()
        min_episode = chekpoint["episode"]

        print("Keep training from previous run.")
    else:
        min_episode = 0
        print("Train from scratch.")

    if params["do_train"]:

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

    player = Play(env, agent)
    player.evaluate()
