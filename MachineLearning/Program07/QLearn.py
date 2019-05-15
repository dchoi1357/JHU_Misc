import random, itertools
import numpy as np
from utilities import getVPolicyFromQ

def epsGreedy(Q, acts, temp=0.5):
    qs = np.array( [Q[k] for k in acts] ) # all Q(s,a) values
    P_a = np.exp(qs / temp) # numerator of softmax, exp[Q(s,a)]
    P_a = P_a / P_a.sum() # array of probabilites
    
    idx = np.argmax(np.random.random() < P_a.cumsum())
    return acts[idx]

def qEpisode(st, Q, TRs, gm, et, prF, trk, trace):
    for t in itertools.count(): # loop for exploration
        if trk[st[1],st[0]] == 'F': # if curr state is a goal
            break

        tr = TRs[st] # all possible transitions from curr state
        temp = max(0.05, 1/np.exp(t/150)**2) # temperature schedule

        attempt = epsGreedy(Q[st], list(tr.keys()), temp) # eps-greedy attemp
        if random.random() < prF: # failed to accelerate
            actual = (0,0) # fail to accelerate
        else: # successfully change accelerattion
            actual = attempt 
            
        newSt,reward = tr[actual] # next state and reward for action
        maxQ = max( Q[newSt][k] for k in TRs[newSt].keys() ) # Q of new state
        Q[st][actual] += et*(reward + gm*maxQ - Q[st][actual]) # update Q(s,a)
        st = newSt

    return Q, t # return Q values and number of steps

def qLearning(states, accels, track, TRs, nEpisodes=100000, gamma=0.9, eta=0.1,
              pr_fail=0.2, trace=None):
    states = list(states)
    #states = [x for x in states if track[x[1],x[0]]!='F'] # remove goal states
    
    Qs = dict()
    for st in states: # initialize Q table to all 0's
        Qs[st] = {a: 0 for a in accels}
        
    epsLen = np.zeros(nEpisodes, int) # length of each episode
    for ep in range(nEpisodes):
        start = random.choice(states) # choose random starting location
        # run one episode of q-learning, get new Qs, ep. len, ep. cum. reward
        Qs,epsLen[ep] = qEpisode(start, Qs, TRs, gamma, eta, pr_fail,
                                 track, trace)
        if trace and (ep%trace)==0:
            print('[%05d] Episode len = %d'%(ep,epsLen[ep]))
    if trace:
        print('[%05d] Last episode, len = %d'%(ep,epsLen[ep]))
    
    policy = dict() # pre-allocate policy
    for st in states: # loop over all states to get policy for each state
        tmp, policy[st] = getVPolicyFromQ(Qs[st]) # best policy accord. to Qs
    return policy, epsLen, Qs # return policy and stats