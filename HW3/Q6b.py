from random import randint
import matplotlib.pyplot as plt
import numpy as np

def episode():
    states = []
    rewards = []

    cur_state = 3 #C states is given by 3

    while cur_state!=6 and cur_state!=0:
        states.append(cur_state)
        move = randint(1,2)
        if move == 1:
            next_state = cur_state - 1
        else:
            next_state = cur_state + 1
        cur_reward = 0
        if next_state == 6:
            cur_reward = 1
        rewards.append(cur_reward)
        cur_state = next_state
    
    return states, rewards

def init_statevals():
    state_value = np.zeros(7)
    for i in range(1,6):
        state_value[i] = 0.5
    return state_value

def rmse(state_value):
    true = [1/6,2/6,3/6,4/6,5/6]
    error = 0
    for i in range(5):
        error += (true[i]-state_value[i])**2
    error /= 5
    error = error ** (1/2)
    return error

def run_td(num_eps,alpha):
    errors = np.zeros(num_eps)
    for run in range(num_eps):
        state_value = init_statevals()
        for e in range(num_eps):
            states, rewards = episode()
            for i in range(len(states)):
                s = states[i]
                r = rewards[i]
                nsv = 0 #next state value
                if i<len(states)-1:
                    ns = states[i+1]
                    nsv = state_value[ns]
                state_value[s] += alpha * (r + nsv - state_value[s]) #TD[0] update eqn
            errors[e] += rmse(state_value[1:6])

    for i in range(num_eps):
        errors[i] /= num_eps

    return errors

def run_mc(num_eps,alpha):
    errors = np.zeros(num_eps)
    for run in range(num_eps):
        state_value = init_statevals()
        for e in range(num_eps):
            states, rewards = episode()
            G = 0
            for i in range(1,len(states)+1):
                t = len(states)-i
                s = states[t]
                r = rewards[t]
                G += r
                state_value[s] += alpha*(G-state_value[s])
            errors[e] += rmse(state_value[1:6])

    for i in range(num_eps):
        errors[i] /= num_eps
    
    return errors

plt.figure()
plt.xlabel("Number of Episodes")
plt.ylabel("Averaged RMSE")
plt.plot(run_td(100,0.15),label="TD, alpha=0.15")
plt.plot(run_td(100,0.1),label="TD, alpha=0.1")
plt.plot(run_td(100,0.05),label="TD, alpha=0.05")
plt.plot(run_mc(100,0.01),label="MC, alpha=0.01")
plt.plot(run_mc(100,0.02),label="MC, alpha=0.02")
plt.plot(run_mc(100,0.03),label="MC, alpha=0.03")
plt.plot(run_mc(100,0.04),label="MC, alpha=0.04")
plt.legend()
plt.show()