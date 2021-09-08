import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

num_of_experiments = 2000
num_of_arms = 10
num_of_timesteps = 1000
variance = 1
mean = 0
variance_per_arm = 4

def run(arms,e):
    reward_val = [0]*num_of_arms # Stores estimates of each arm, gets updated after each timestep
    rewards = [] # Ordering of reward gotten on each specific step
    actions = [] # Ordering of actions selected through the timesteps
    errors = [] # Error (between current estimated reward_val and actual expected value) in 2-d array, each row stores all arms value
    arm_select_count = [0] * num_of_arms

    for timestep in range(num_of_timesteps):
        # Error finding subroutine
        cur_error = []
        for i in range(num_of_arms):
            cur_estimate = reward_val[i]
            actual_expected = arms[i][0]
            cur_error.append(abs(cur_estimate-actual_expected)) #Difference between actual expected value assigned to Gaussian and the current estimate
        errors.append(cur_error)

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
        reward_val[cur_action] += (1/arm_select_count[cur_action])*(cur_reward-reward_val[cur_action])
    
    return rewards, actions, errors

# e holds the epsilon value for that particular 2000-test experiment
def experiment(e):
    print("Epsilon = ",e)
    rewards = [0]*num_of_timesteps #Will hold total reward value at each timestep across all runs
    optimal = [0]*num_of_timesteps #Will hold optimal action % at each timestep
    errors = []
    for i in range(num_of_timesteps):
        errors.append([0]*num_of_arms)

    for i in range(num_of_experiments):
        print("#"+str(i+1), end='\r')
        arms = [[np.random.normal(mean,variance**(1/2)), variance_per_arm] for j in range(num_of_arms)] # First variable holds Expected value, second holds variance 
        exp_rewards, exp_actions, exp_errors = run(arms,e)
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
            for arm in range(num_of_arms):
                errors[timestep][arm] += exp_errors[timestep][arm]
        
    for i in range(num_of_timesteps): # This for loop just averages all the values that have been added so far, over all the experiments
        rewards[i] /= num_of_experiments
        optimal[i] = 100*optimal[i]/num_of_experiments
        for j in range(num_of_arms):
            errors[i][j] /= num_of_experiments
    
    return rewards, optimal, errors
                

if __name__ == "__main__":
    epsilon = [0,0.1,0.01]
    rewards = {} # Will hold the value of average reward for each (epsilon value,timestep)
    optimal = {} # Will hold the value of optimal action % for each (epsilon value, timestep)
    error = {} # Will hold the average absolute error in the estimate for each arm (epsilon value, timestep, arm #)

    for e in epsilon:
        rew, opt, err = experiment(e)
        rewards[e] = rew
        optimal[e] = opt
        error[e] = err
        print("\rDone")
    
    plt.figure()
    for e in epsilon:
        plt.plot(rewards[e], label = "e = "+str(e))
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.figure()
    for e in epsilon:
        plt.plot(optimal[e], label = "e = "+str(e))
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

    # 3 10 line graphs for each epsilon denoting error-
    for e in epsilon:
        plt.figure()
        error_transpose = np.array(error[e])
        error_transpose = np.transpose(error_transpose)
        for i in range(num_of_arms):
            plt.plot(error_transpose[i], label = "Arm# "+ str(i))
        title = "Epsilon = "+str(e)
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel("Average Absolute Error")
        plt.legend()

    plt.show()
