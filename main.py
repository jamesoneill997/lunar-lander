import gym
from model import Agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_min=0.01, input_dims=[8], learning_rate=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        #env.render()
        score = 0
        done = False
        obsv = env.reset()
        while not done:
            action = agent.choose_action(obsv)
            new_obsv, reward, done, info = env.step(action)
            score+=reward
            agent.store_transition(obsv, action, reward, new_obsv, done)
            agent.learn()
            obsv = new_obsv
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode %.2f' % i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

plt.plot(scores)
plt.ylabel('Scores')
plt.show()