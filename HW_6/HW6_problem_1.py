'''
Created on Dec 8, 2016

@author: Uri Smashnov

Home work 6 - problem 1 - Markov decision processes
Calculate optimal policy using greedy strategy
Action - dictionary of actions. Key: current state, value: tuple - possible transition states from current states
Reward - rewards dictionary. Key is tuple with (from_state, to_state) and value is the reward
P_init : initial policy 
'''
import copy

Action={1:(1,2,3),2:(1,2,3),3:(1,2,3)}
Reward={(1,2):1,(2,3):1,(3,1):-1, (1,3):0.25, (3,2):0.25, (2,1):0.25, (1,1):0, (2,2):0, (3,3):0}
P_init={1:2,2:3,3:1}
gamma=0.9

def value_funct(policy,state,gamma):
    value=0
    state_p=state
    #print(Action[state_p])
    #print(state_p,policy[state_p], R[(state_p,policy[state_p])])
    for i in range(100):
        value=value + Reward[(state_p,policy[state_p])]*(gamma**i)
        state_n=policy[state_p]
        state_p=state_n
    return value

def get_best_action_reward(state_0):
    max_R=-100
    best_action=0
    #print("Actions", Action[state_0])
    for action in Action[state_0]:
        #print("Iter",(state_0,action),R[(state_0,action)])
        if max_R<Reward[(state_0,action)]:
            max_R=Reward[(state_0,action)]
            best_action=action
    return best_action,max_R            

policy_init={1:1,2:2,3:3}
policy_init={1:2,2:3,3:1}
state_init=1

policy_value=value_funct(policy_init, state_init, gamma)

print("Starting policy",policy_init)
print("Starting policy value", policy_value)
for i in range(100):
    policy_value=value_funct(policy_init, state_init, gamma)
    #print("State",state_init)
    #print("Policy value",policy_value)
    #print("Current policy",policy_init)
    for action in Action[state_init]:
        policy_next=copy.copy(policy_init)
        policy_next[state_init] = action
        value_next = value_funct(policy_next, state_init, gamma)
        if value_next>policy_value:
            #print("Policy",policy_next," Value",value_next)
            policy_value=value_next
            policy_init=policy_next
            last_action=action
    state_init=(state_init+1)
    if state_init==4:
        state_init=1        

print("Best policy", policy_init)
print("Best value", policy_value )