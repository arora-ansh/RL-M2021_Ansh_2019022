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

#Initial State Value
def init_statevals():
    state_value = np.zeros(7)
    for i in range(1,6):
        state_value[i] = 0.5
    return state_value

alpha = 0.1

def run(num_eps):
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
    return state_value

print(episode())

plt.figure()
plt.xticks([0 ,1, 2, 3, 4], ['A', 'B', 'C', 'D', 'E'])
plt.plot(run(0)[1:6],label="0")
plt.plot(run(1)[1:6],label="1")
plt.plot(run(10)[1:6],label="10")
plt.plot(run(100)[1:6],label="100")
plt.plot([1/6,2/6,3/6,4/6,5/6],label = "True Values")
plt.xlabel("State")
plt.ylabel("Estimates")
plt.grid()
plt.title("Estimated values of states")
plt.legend()
plt.show()