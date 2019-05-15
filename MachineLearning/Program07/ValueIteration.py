import itertools
from utilities import getVPolicyFromQ

def calcQfromV(state, tr, V_0, failPr, gamma):
    Q = {k:0 for k in tr} # initialize Q to all 0 for all actions
    failSt = tr[(0,0)][0] # resultant state of a failed acceleration
    
    for accel,(newState,reward) in tr.items(): # loop over all transitions
        tmp = (1-failPr)*V_0[newState] + failPr*V_0[failSt] # sum of pr*V
        Q[accel] = reward + gamma * tmp # E[r|s,a] + gamma * sum(V)
    return Q

def maxDiffTwoVs(v1, v2):
    pairedVals = zip(v1.values(), v2.values()) # get all values from both V vals
    return max([abs(x-y) for x,y in pairedVals]) # return max diff of all pairs

def valueIteration(states, TRs_all, gamma=0.9, eps=1e-9, pr_fail=0.2,
                   trace=False):
    Vs = {k:0 for k in states}
    Qs = {k:None for k in states}
    pols = dict()
    
    for t in itertools.count(): # loop until converged
        Vs_old = Vs.copy() # copy of Vs as old V values
        for st in states: # loop over all states
            Qs[st] = calcQfromV(st, TRs_all[st], Vs_old, pr_fail, gamma)
            Vs[st],pols[st] = getVPolicyFromQ(Qs[st])
        
        maxDiff = maxDiffTwoVs(Vs, Vs_old)
        if trace and (t % 10 == 0):
            print('[%05d]: diff=%f'%(t,maxDiff))
        
        if (maxDiff<eps) or (t >= 1e4): # max 1000 iters if not converged
            break

    if trace:
        print('Total iters: %d, max util diff = %f'%(t,maxDiff))
    return pols, t, maxDiff