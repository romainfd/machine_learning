import gym
env = gym.make('CartPole-v0')

for i_episode in range(10):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        # action = env.action_space.sample()  # gets an action randomly
        action = 1  # or 0
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("State space dimension is:", env.observation_space.shape[0])
print("State upper bounds:", env.observation_space.high)
print("State lower bounds:", env.observation_space.low)
print("Number of actions is:", env.action_space.n)
