import numpy as np
import random, itertools, time
from IPython.display import clear_output

def compPolicy(base, comp):
    same = np.zeros(len(base), bool) # if policy states the same
    for n,state in enumerate(base.keys()): # loop over all states
        same[n] = base[state] == comp[state]
        
    return same

def trackToStr(track, state):
    trk = track.copy()
    trk[state[1],state[0]] = 'X'
    s = '\n'.join([''.join(l) for l in trk]) + '\n'
    s += 'x-vel=%d, y-vel=%d\n'%state[2:]
    return s

def simulateRace(trk, TRs, startPt, pol, p_fail=0.2, trace=False):
    stepLimit = np.prod(trk.shape)*2/(1-p_fail) # step limit for non-viability
    
    st = startPt
    if type(startPt) is list: # if more than one start point provided
        st = random.choice(startPt) # choose starting point randomly        
    if len(st) != 4: # only location provided
        st = st + (0,0) # set starting state, 0 velocity

    viable = False
    for t in itertools.count(): # counter for total distance
        if trace:
            clear_output(wait=True)
            print(trackToStr(trk,st))
            time.sleep(0.2)
        
        if trk[st[1],st[0]] == 'F': # if curr state is a goal
            viable = True
            break
        if t > stepLimit: # too many steps, not viable policy
            break
        
        if random.random() < p_fail: # failed to accelerate
            accel = (0,0)
        else:
            accel = pol[st] # optimal accel according to policy
            
        st = TRs[st][accel][0] # set new state to result of transition
    return t, startPt, viable # return steps taken, start pt, and viability

def compNSteps(base, comp, trk, TRs):
    nStepsBase = np.zeros(len(base), int) # number of steps
    nStepsComp = np.zeros(len(base), int) # number of steps
    viable = np.zeros(len(base), bool) # whether viable policy or not

    for n,st in enumerate(base.keys()):
        nStepsBase[n],tmp,viable[n] = simulateRace(trk, TRs, st, base)
        nStepsComp[n],tmp,d = simulateRace(trk, TRs, st, comp)

    np.seterr(all='ignore')
    return nStepsBase/nStepsComp, viable