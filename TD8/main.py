import gym
import numpy as np
env = gym.make('FrozenLake-v0')


#################################################################################
#  Use these variables to change the RL algorithm and the exploration strategy  #
#################################################################################
# Reinforcement Learning algorithm (SARSA or qlearning)
qlearning = True
# Exploration strategy (epsilon-greedy or softmax)
softMax = True
#################################################################################


nb_episodes = 2000
alpha = 0.4
gamma = 0.999
epsilon = 0
tau = 0.003
q_table = np.ones((16, 4))  # 16 = 4x4 grid; 4 = [left, down, right, up]
results = []

def getAction(env, q_table, observation):
    if softMax:
        #rand = np.random.uniform(0, 1) * np.sum(np.exp(q_table[observation] / tau))
        #cumulate = 0
        #i = 0
        #while cumulate < rand and i < len(q_table[observation]):
        #    cumulate += np.exp(q_table[observation][i] / tau)
        #    i += 1
        #return i

        # we use np.choice instead
        elements = [0, 1, 2, 3]
        probabilities = np.array([np.exp(q_table[observation][i] / tau) for i in range(len(q_table[observation]))])
        probabilities /= np.sum(np.exp(q_table[observation] / tau))
        return np.random.choice(elements, 1, p=probabilities)[0]
    else:
        if np.random.uniform(0, 1) > epsilon:
            # we take the best one seen so far
            act = np.argmax(q_table[observation])
        else:
            print("Random ", end='')
            act = env.action_space.sample()  # gets an action randomly
        print("Action: {}".format(act))
        return act


for i_episode in range(nb_episodes):
    observation = env.reset()
    action = getAction(env, q_table, observation)

    for t in range(200):
        env.render()

        # we do the first step
        print("Observation: {}".format(observation))  # from 0 to 15 (0 1 2 3 on first row)
        observation_2, reward, done, info = env.step(action)

        # then we do the second step VIRTUALLY
        action_2 = getAction(env, q_table, observation_2)

        # we update Q
        error = reward - q_table[observation, action]
        if not done:
            if qlearning:
                error += gamma * np.max(q_table[observation_2])
            else:
                error += gamma * q_table[observation_2, action_2]


        q_table[observation, action] += alpha * error

        if done:
            results.append(observation_2)
            print("Episode finished after {} timesteps".format(t+1))
            break

        # we set observation to the next step
        observation = observation_2
        action = action_2

print("State space dimension is:", env.observation_space.n)
#print("State upper bounds:", env.observation_space.high)
#print("State lower bounds:", env.observation_space.low)
print("Number of actions is:", env.action_space.n)
window = 100
unique, counts = np.unique(results[-window:], return_counts=True)
if unique[-1] == 15:
    print("Number of successes = {} out of the {} last ones".format(counts[-1], window))
else:
    print("No success")
