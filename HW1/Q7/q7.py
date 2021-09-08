import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

num_of_experiments = 2000
num_of_arms = 10
num_of_timesteps = 1000
variance = 1
mean = 4
variance_per_arm = 1

def run(arms,alpha,baseline):
    reward_val = [0] * num_of_arms # Stores estimates of each arm, gets updated after each timestep
    actions = [] # Ordering of actions selected through the timesteps
    arm_select_count = [0] * num_of_arms
    total_reward = 0

    for timestep in range(1,num_of_timesteps+1):
        # Create Softmax Distribution
        softmax = []
        for i in reward_val:
            softmax.append(math.exp(i))
        total = sum(softmax)
        for i in range(num_of_arms):
            softmax[i] /= total
        # Choose action
        arm_choices = list(range(num_of_arms))
        cur_action = np.random.choice(arm_choices, p = softmax)
        actions.append(cur_action)

        # Reward value based on action selected subroutine 
        cur_reward = np.random.normal(arms[cur_action][0],arms[cur_action][1]**(1/2))
        total_reward += cur_reward

        # Action preference updation subroutine according to eqn 2.12 (based on baseline or not)
        if not baseline:
            for i in range(num_of_arms):
                if i==cur_action:
                    reward_val[i] += alpha*(cur_reward - total_reward/timestep)*(1 - softmax[i])
                else: 
                    reward_val[i] -= alpha*(cur_reward - total_reward/timestep)*softmax[i]
        else:
            for i in range(num_of_arms):
                if i==cur_action:
                    reward_val[i] += alpha*(cur_reward)*(1 - softmax[i])
                else: 
                    reward_val[i] -= alpha*(cur_reward)*softmax[i]
    
    return actions

# e holds the epsilon value for that particular 2000-test experiment
def experiment(alpha,baseline):
    print("Alpha = ", alpha, "Baseline = ", baseline)
    optimal = [0]*num_of_timesteps #Will hold optimal action % at each timestep

    for i in range(num_of_experiments):
        print("#"+str(i+1), end='\r')
        arms = [[np.random.normal(mean,variance**(1/2)), variance_per_arm] for j in range(num_of_arms)] # First variable holds Expected value, second holds variance 
        exp_actions = run(arms,alpha,baseline)
        exp_opt_arm = 0 # Arbitrarily taking 0th arm as the optimal arm 
        exp_opt_val = arms[0][0]
        for j in range(1,num_of_arms):
            if(exp_opt_val<arms[j][0]):
                exp_opt_val = arms[j][0]
                exp_opt_arm = j
        for timestep in range(num_of_timesteps):
            if exp_actions[timestep] == exp_opt_arm:
                optimal[timestep] += 1
        
    for i in range(num_of_timesteps): # This for loop just averages all the values that have been added so far, over all the experiments
        optimal[i] = 100*optimal[i]/num_of_experiments
    
    return optimal
                

if __name__ == "__main__":
    
    optimal = experiment(0.1,True)
    optimal2 = experiment(0.1,False)
    optimal3 = experiment(0.4,True)
    optimal4 = experiment(0.4,False)

    plt.figure()
    plt.plot(optimal, label = "alpha = 0.1, basline")
    plt.plot(optimal2, label = "alpha = 0.1, non-basline")
    plt.plot(optimal3, label = "alpha = 0.4, basline")
    plt.plot(optimal4, label = "alpha = 0.4, non-basline")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

    plt.show()
