import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

num_of_experiments = 2000
num_of_arms = 10
num_of_timesteps = 10000
variance = 0 # Variance taken 0 to assign the same value of 0 to all hands
mean = 0
variance_per_arm = 1

def run(arms,e,stepSize = None):
    reward_val = [0]*num_of_arms # Stores estimates of each arm, gets updated after each timestep
    rewards = [] # Ordering of reward gotten on each specific step
    actions = [] # Ordering of actions selected through the timesteps
    arm_select_count = [0] * num_of_arms

    for timestep in range(num_of_timesteps):

        # Action selection subroutine -> Find argmax from reward_val with 1-epsilon probability, find random action with epsilon probability
        greedy_choice = np.argmax(reward_val) # With probability (1-epsilon)
        explore_choices = list(range(num_of_arms))
        explore_choices.remove(greedy_choice) # With probability epsilon
        arm_choices = explore_choices + [greedy_choice]

        pmf = [e/num_of_arms]*(num_of_arms-1) + [1-e+(e/num_of_arms)]
        cur_action = np.random.choice(arm_choices, p = pmf)
        actions.append(cur_action)
        # print(cur_action)
        # Reward value based on action selected subroutine
 
        cur_reward = np.random.normal(arms[cur_action][0],arms[cur_action][1]**(1/2))
        rewards.append(cur_reward)

        # reward_val update subroutine -> update arm_select_count using N(A) <- N(A) + 1 and reward_val using Q(A) <- Q(A) + (1/N(A))*[R-Q(A)]
        arm_select_count[cur_action] += 1
        if not stepSize :
            reward_val[cur_action] += (1/arm_select_count[cur_action])*(cur_reward-reward_val[cur_action])
        else :
            reward_val[cur_action] += stepSize*(cur_reward-reward_val[cur_action])

        #In each step we also need to update the q*(a) for all arms, mean 0 and std dev 0.01
        for arm in arms:
            arm[0] += np.random.normal(0,0.01)
    
    return rewards, actions

# e holds the epsilon value for that particular 2000-test experiment
def experiment(e, stepSize = None):
    print("Epsilon = ",e)
    rewards = [0]*num_of_timesteps #Will hold total reward value at each timestep across all runs
    optimal = [0]*num_of_timesteps #Will hold optimal action % at each timestep

    for i in range(num_of_experiments):
        print("#"+str(i+1), end='\r')
        arms = [[np.random.normal(mean,variance**(1/2)), variance_per_arm] for j in range(num_of_arms)] # First variable holds Expected value, second holds variance 
        exp_rewards, exp_actions = run(arms, e, stepSize)
        exp_opt_arm = 0 # Arbitrarily taking 0th arm as the optimal arm 
        exp_opt_val = arms[0][0]
        for j in range(1,num_of_arms):
            if(exp_opt_val<arms[j][0]):
                exp_opt_val = arms[j][0]
                exp_opt_arm = j
        for timestep in range(num_of_timesteps):
            rewards[timestep] += exp_rewards[timestep]
            if exp_actions[timestep] == exp_opt_arm:
                optimal[timestep] += 1
        
    for i in range(num_of_timesteps): # This for loop just averages all the values that have been added so far, over all the experiments
        rewards[i] /= num_of_experiments
        optimal[i] = 100*optimal[i]/num_of_experiments
    
    return rewards, optimal
                

if __name__ == "__main__":
    epsilon = 0.1

    rewards, optimal = experiment(epsilon)
    rewards2, optimal2 = experiment(epsilon,0.1)

    plt.figure()
    plt.plot(rewards, label = "e = 0.1, with step size sample mean")
    plt.plot(rewards2, label = "e = 0.1, with step size 0.1")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.figure()
    plt.plot(optimal, label = "e = 0.1, with step size sample mean")
    plt.plot(optimal2, label = "e = 0.1, with step size 0.1")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

    plt.show()
