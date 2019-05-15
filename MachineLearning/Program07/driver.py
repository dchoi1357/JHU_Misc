import sys, os, itertools
from utilities import readTrackFile, getTransitions
from ValueIteration import valueIteration
from QLearn import qLearning
from evaluate import compPolicy, simulateRace, compNSteps
import numpy as np

valids = {'L-track.txt', 'R-track.txt', 'O-track.txt', 'test-track.txt'}
if sys.argv[1] not in valids: # check if input file exist
	sys.exit("No such track data as %s"%(sys.argv[1]))

if len(sys.argv) > 2:
	nEps = int(sys.argv[2])
else:
	nEps = 100000

fpath = os.path.join('data', sys.argv[1])

accelList = [1,0,-1] # valid acceleraion values
velocList = range(-5,6) # valid velocities
accels = list(itertools.product(accelList,accelList)) # cartesion products
velocs = list(itertools.product(velocList,velocList)) # cartesian products

trStates, trGoals, trTrack, trStarts = readTrackFile(fpath, velocs)

# pre-generate all valid transitions from all states, soft and hard crashes
TRs_soft = getTransitions(trStates, accels, trTrack, trStarts) 
TRs_hard = getTransitions(trStates, accels, trTrack, trStarts, hardCrash=True)

vi_soft = valueIteration(trStates, TRs_soft, trace=True)
vi_hard = valueIteration(trStates, TRs_hard, trace=True)

qlearn_soft = qLearning(trStates, accels, trTrack, TRs_soft, nEps, trace=nEps/10)
qlearn_hard = qLearning(trStates, accels, trTrack, TRs_hard, nEps, trace=nEps/10)

# comparing policy from Q-Learning to Value Iteration
softPolComp = compPolicy(qlearn_soft[0], vi_soft[0])
hardPolComp = compPolicy(qlearn_hard[0], vi_hard[0])

softStepRatio,softViable = compNSteps(qlearn_soft[0], vi_soft[0], trTrack, TRs_soft)
hardStepRatio,hardViable = compNSteps(qlearn_soft[0], vi_soft[0], trTrack, TRs_hard)

print("== Soft Crash Comparison ==")
print( "Policy Concordance Rate: %f"%softPolComp.mean() )
print( "Mean Ratio of N Steps: %f"%np.nanmean(softStepRatio) )
print( "St Dev Ratio of N Steps: %f"%np.nanstd(softStepRatio) )
print( "%% of Policies from Q-Learning Viable: %f"%softViable.mean())
print()

print("== Hard Crash Comparison ==")
print( "Policy Concordance Rate: %f"%hardPolComp.mean() )
print( "Mean Ratio of N Steps: %f"%np.nanmean(hardStepRatio) )
print( "St Dev Ratio of N Steps: %f"%np.nanstd(hardStepRatio) )
print( "%% of Policies from Q-Learning Viable: %f"%hardViable.mean())
print()

## Multiple simulations
nSims = 100
vi_steps = np.zeros([nSims,2], int)
ql_steps = np.zeros([nSims,2], int)

for n in range(nSims):
	for st in trStarts:
		vi_steps[n,0] += simulateRace(trTrack, TRs_soft, st, vi_soft[0])[0]
		vi_steps[n,1] += simulateRace(trTrack, TRs_hard, st, vi_hard[0])[0]

		ql_steps[n,0] += simulateRace(trTrack, TRs_soft, st, qlearn_soft[0])[0]
		ql_steps[n,1] += simulateRace(trTrack, TRs_hard, st, qlearn_hard[0])[0]

vi_steps = vi_steps/len(trStarts)
ql_steps = ql_steps/len(trStarts)

print("== Simulations ==")
print("VI Mean Steps in Sim: Soft=%f, Hard=%f"%tuple(vi_steps.mean(axis=0)))
print("VI Steps Std Dev in Sim: Soft=%f, Hard=%f"%tuple(vi_steps.std(axis=0)))
print("QL Mean Steps in Sim: Soft=%f, Hard=%f"%tuple(ql_steps.mean(axis=0)))
print("QL Steps Std Dev in Sim: Soft=%f, Hard=%f"%tuple(ql_steps.std(axis=0)))