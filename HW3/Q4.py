import gym
import numpy as np
import matplotlib.pyplot as plt

def f1(first_visit):
    env = gym.make("Blackjack-v1")

    usable_ace_state_value = {}
    non_usable_ace_state_value = {}

    #Initialize policy

    def policy(state):
        if state[0]==20 or state[0]==21:
            return 0 #Stick
        else:
            return 1 #Hit

    #Initialize V(s) which will estimate state value function
    state_value = {}
    state_count = {}

    episodes = 10000 #Number of episodes
    for e in range(episodes):
        print(f"Game # {e+1} \t", end = "\r")

        #Generate Episode
        states = []
        actions = []
        rewards = [None] #Since R0 doesnt exist
        cur_state = env.reset() #(12, 7, False) form where 12 gives sum of player's cards, 7 is the face up card 

        while True:
            #Figure out action A_t from current state S_t on policy 
            action = policy(cur_state)
            #Find out next state S_t+1 and the reward reaped R_t+1 from current action A_t
            next_state, reward, end, useless = env.step(action)
            #Append and create S,R,A arrays for current episode
            states.append(cur_state)
            cur_state = next_state
            actions.append(action)
            rewards.append(reward)
            #If game has ended then exit from loop, just saving final state
            if end:
                states.append(cur_state)
                break
        
        # print(states,actions,rewards)
        G = 0
        T = len(states)
        for i in range(2,T+1):
            t = T-i
            G += rewards[t+1]
            St = states[t]

            if first_visit:
                #First Visit Implementation
                pass
            else:
                #Every Visit Implementation
                if St not in state_value:
                    state_value[St] = 0
                    state_count[St] = 0
                state_value[St] += G
                state_count[St] += 1

    for St in state_value:
        state_value[St] /= state_count[St]
        if St[2]:
            cur_state = (St[0],St[1])
            usable_ace_state_value[cur_state] = state_value[St]
        else:
            cur_state = (St[0],St[1])
            non_usable_ace_state_value[cur_state] = state_value[St]

    # print(usable_ace_state_value)
    # print()
    # print(non_usable_ace_state_value)

    usable_output = np.zeros((10,10)) 
    non_usable_output = np.zeros((10,10))

    for St in usable_ace_state_value:
        if St[0]>=12 and St[0]<=21 and St[1]>=1 and St[1]<=10:
            usable_output[St[0]-12][St[1]-1] = usable_ace_state_value[St]

    for St in non_usable_ace_state_value:
        if St[0]>=12 and St[0]<=21 and St[1]>=1 and St[1]<=10:
            non_usable_output[St[0]-12][St[1]-1] = non_usable_ace_state_value[St]
    
    for i in range(10):
        for j in range(10):
            print(round(usable_output[i][j],2), end = " ")
        print()
    print()
    for i in range(10):
        for j in range(10):
            print(round(non_usable_output[i][j],2), end = " ")
        print()
    
    x = [12,13,14,15,16,17,18,19,20,21]
    y = [1,2,3,4,5,6,7,8,9,10]
    X, Y = np.meshgrid(x,y)
    Z = np.array(usable_output)[X-12][Y-1]
    print()
    print(Z[0][0])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')   
    ax.plot_wireframe(X,Y,Z[0][0])
    plt.show()

    x = [12,13,14,15,16,17,18,19,20,21]
    y = [1,2,3,4,5,6,7,8,9,10]
    X, Y = np.meshgrid(x,y)
    Z = np.array(non_usable_output)[X-12][Y-1]
    print()
    print(Z[0][0])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')   
    ax.plot_wireframe(X,Y,Z[0][0])
    plt.show()

f1(False)    



        


