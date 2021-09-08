import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

num_of_experiments = 2000
num_of_arms = 10
num_of_timesteps = 1000
variance = 1
mean = 0
variance_per_arm = 1

def run(arms,e,c=None):
    reward_val = [0]*num_of_arms # Stores estimates of each arm, gets updated after each timestep
    rewards = [] # Ordering of reward gotten on each specific step
    arm_select_count = [0] * num_of_arms

    for timestep in range(1,num_of_timesteps+1):

        cur_action = 0
        # Action selection subroutine -> Find argmax from reward_val with 1-epsilon probability, find random action with epsilon probability
        if not c:
            greedy_choice = np.argmax(reward_val) # With probability (1-epsilon)
            explore_choices = list(range(num_of_arms))
            explore_choices.remove(greedy_choice) # With probability epsilon
            arm_choices = explore_choices + [greedy_choice]
            pmf = [e/num_of_arms]*(num_of_arms-1) + [1-e+(e/num_of_arms)]
            cur_action = np.random.choice(arm_choices, p = pmf)

        # Upper confidence bound implemented with given value of c
        else:
            maxchoice = -1
            maxval = -10000000
            flag = False #Handles for case with 0 arm select count
            for i in range(num_of_arms):
                if arm_select_count[i]==0:
                    cur_action = i
                    flag = True
                else:
                    if reward_val[i]+c*((math.log(timestep)/arm_select_count[i])**(1/2))>maxval:
                        maxval = reward_val[i] + c*((math.log(timestep)/arm_select_count[i])**(1/2))
                        maxchoice = i
            if not flag:
                cur_action = maxchoice
            
        # Reward value based on action selected subroutine
        cur_reward = np.random.normal(arms[cur_action][0],arms[cur_action][1]**(1/2))
        rewards.append(cur_reward)

        # reward_val update subroutine -> update arm_select_count using N(A) <- N(A) + 1 and reward_val using Q(A) <- Q(A) + (1/N(A))*[R-Q(A)]
        arm_select_count[cur_action] += 1
        reward_val[cur_action] += (1/arm_select_count[cur_action])*(cur_reward-reward_val[cur_action])
    
    return rewards

# e holds the epsilon value for that particular 2000-test experiment
def experiment(e,c = None):
    print("Epsilon = ",e)
    rewards = [0]*num_of_timesteps #Will hold total reward value at each timestep across all runs

    for i in range(num_of_experiments):
        print("#"+str(i+1), end='\r')
        arms = [[np.random.normal(mean,variance**(1/2)), variance_per_arm] for j in range(num_of_arms)] # First variable holds Expected value, second holds variance 
        exp_rewards = run(arms,e,c)
    
        for timestep in range(num_of_timesteps):
            rewards[timestep] += exp_rewards[timestep]
        
    for i in range(num_of_timesteps): # This for loop just averages all the values that have been added so far, over all the experiments
        rewards[i] /= num_of_experiments
    
    return rewards
                

if __name__ == "__main__":
    epsilon = 0.1
    rewards = experiment(epsilon)
    print("\rDone")
    rewards2 = experiment(epsilon,2)
    # Only average reward to be shown, for graph with eps greedy with e = 0.1 and 

    plt.figure()
    plt.plot(rewards, label = "e = 0.1")
    plt.plot(rewards2, label = "UCB with c = 2")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.show()
