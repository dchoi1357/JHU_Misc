from Bresenham import getPath
import itertools
import numpy as np

def readTrackFile(file, velocs):
    ''' Read a specific track file, return the entire state space of locations
    and velocities, as well as the list of goals, the track space as a numpy
    2D array, and the set of starting points.
    '''

    with open(file, 'r') as f:
        raw = [[s for s in l.strip()] for l in f.readlines()]
        track = np.array(raw[1:]) # strip out lines with world size

    locs,goals,starts = set(),set(),list()
    for x,y in itertools.product(range(track.shape[1]), range(track.shape[0])):
        space = track[y,x] # current track space
        if space == '#': # wall space, not valid state
            continue
        else:
            locs.add( (x,y) ) # valid state of car
            if space == 'S': # starting points
                starts.append( (x,y) ) # add to list of starting locations
            if space == 'F': # is finish line space
                goals.add( (x,y) ) # add to list of goal locations

    # enumerate possible states as cartesian prod of locations and velocities
    states = set((x,y,a,b) for (x,y),(a,b) in itertools.product(locs,velocs))
    return states,goals,track,starts

def transFromState(state,actions,track,startPts,crashReset):
    gen = np.random.RandomState(42) # use independent stream for reproducibility

    x0, y0, vx0, vy0 = state
    trans = dict()
    for ax,ay in actions: # loop through all possible accelerations
        vx1 = min(max(vx0 + ax,-5),5) # updated velocity
        vy1 = min(max(vy0 + ay,-5),5)
        paths = getPath(x0, y0, vx1, vy1) # path car takes with velocity vec

        r = -1 # default reward
        x1, y1 = (x0,y0) # set to starting location
        for xC,yC in paths: # x,y coordinate of every step in path
            if track[yC,xC] == '#': # run into wall
                if crashReset: # reset to a random starting point
                    n = gen.choice(len(startPts)) # randomly choose start points
                    x1,y1 = startPts[n]
                vx1, vy1 = (0,0)
                break
            if track[yC,xC] == 'F': # goes over finish line
                x1, y1 = (xC,yC)
                vx1, vy1 = (0,0)
                r = 0
                break
            x1, y1 = (xC,yC) # update location to new place
        trans[(ax,ay)] = ((x1,y1,vx1,vy1),r)  # new state after transition
    
    return trans

def getTransitions(states,accels,world,starts,hardCrash=False):
    '''
    Produce all valid transitions from all input states for a given world.

    The input to the functions are:
        states - set of tuples in the form of (x, y, x_veloc, y_veloc)
        accels - set of tuples in the form of (x_accel, y_accel)
        world - 2D array of track spaces

    The function loops through all states, and for every valid states (i.e. 
    passible and non-goal states), attemps all actions in the accelerations 
    input from the state. Only valid transitions are added to the output, 
    meaning that a transition onto impassible walls or out of bound are not 
    returned, nor are any transition from finish line. For all transition, 
    a reward is attached to all transitions either -1 or 0 for goal.

    The output is a dict of dict, where the keys of the outer dict are tuples of
    states, and for every state, there is a dict with the key of different 
    accelerations. The value of the dict are the resulting state and reward.

    For example, Rs[(2,1,0,0)][(1,0)] = ((3,1,1,0), -1) means applying 
    acceleraiton of (1,0) for x=2, y=1, x_veloc=0, y_veloc=0 results in the
    agent being at the state of (3,1,1,0), with a reward of -1.
    '''
    Rs = dict()
    for st in states:
        if world[st[1],st[0]] == 'F': # is finish line
            immob = (st[0], st[1], 0, 0) # immobilize at curr goal position
            Rs[st] = {(0,0): (immob, 0)} # only valid trans is staying put
        else: # get all transitions from current state if not finish line
            Rs[st] = transFromState(st,accels,world,starts,hardCrash)
    return Rs

def getVPolicyFromQ(Q):
    '''T he function takes a set Q-values for a given state, and returns both
    the maximum Q-value and the corresponding optimal policy. 
    '''
    acts = Q.keys()
    pol = max(acts, key=(lambda k: Q[k])) # get the key with highest reward
    return Q[pol], pol