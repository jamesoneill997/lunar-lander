import gym
from model import Agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_min=0.01, input_dims=[8], learning_rate=0.001)
    scores, eps_history = [], []
    n_games = 2000
    averages = []

    #episode loop
    for i in range(n_games):
        obsv = env.reset()
        score = 0
        done = False
        while not done:
            #env.render() #render env, comment to halt
            action = agent.choose_action(obsv) #choose action based on current env state
            new_obsv, reward, done, info = env.step(action) #step forward
            score+=reward #set score
            agent.store_transition(obsv, action, reward, new_obsv, done)
            agent.learn()
            obsv = new_obsv
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        #take sample of average score every 10 episodes
        if i % 10 == 0:
            averages.append(avg_score)

        print('Episode %.0f\n' % i, 'Score %.2f\n' % score, 'Average_score (last 100 games) %.2f\n' % avg_score, 'Epsilon %.2f\n' % agent.epsilon, '-------------\n')

#end of game loop, kill env process and plot results
env.close()
plt.plot(averages)
plt.ylabel('Average Score:')
plt.xlabel('Time (Episodes):')
plt.show()