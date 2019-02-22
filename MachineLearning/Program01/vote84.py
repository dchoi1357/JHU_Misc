import numpy as np
import os
import pandas as pd
import functions.modeling as mdl # import modeling functions

vote84Data = os.path.join('./data/', 'house-votes-84.data')
vote84Names = ['class', 'infant', 'water', 'budget', 'doctorfee','salvador',
              'religion', 'satellite', 'contras', 'missile', 'immigration',
              'synfuels', 'education', 'superfund', 'crime', 'exports',
              'ZAF']

raw = pd.read_csv(vote84Data , names=vote84Names ) # read in vote file

# generate one-hot coding for issues with lots of missing votes
def oneHot(data, colName):
    ''' Generate one-hot coding for selected columns
    data is a pandas dataframe, and colName are a list of variables for which 
    one-hot coding is to be generated. The possible codes are 'y', 'n', '?'. 
    For every column will result in three columns representing whether the 
    source column has the specified coding. 
    Returns a pandas dataframe.
    '''
    x = data.loc[:,colName]
    oneHotMat = pd.concat([(x=='y'),(x=='n'),(x=='?')], axis=1)
    oneHotMat.columns = [colName+'_'+suff for suff in ['y','n','q']]
    return oneHotMat

oneHotCols = pd.concat([oneHot(raw,'water'), oneHot(raw,'education'), 
                        oneHot(raw,'ZAF')], axis=1)

# remove variables with completed one-hot coding from list of variables
yesVars = np.setdiff1d(vote84Names[1:],['water','education','ZAF'])
yesVote = raw.loc[:,yesVars] == 'y' # boolean for vote='yes' for rest of vars
yesVote.columns = [s+'_y' for s in yesVote.columns]
repub = raw.loc[:,['class']] == 'republican' # boolean for republicans
voteData = pd.concat([yesVote,oneHotCols], axis=1) # concat two dataframes
voteMat = voteData.values * 1 # give matrixs of 0 & 1 for calculation
repubVec = repub.values.ravel() * 1 # vector of 0 & 1 for calculation

votePrm = {'theta': voteMat.shape[1]/4, 'alpha': 2} # hyper-parameters 
mdl.crossValidate(voteMat, repubVec, votePrm) # cross validation test
wts,nbProb = mdl.estModels(voteMat,repubVec,votePrm) # estimate with all data
print()
mdl.printWinnowModel(wts, voteData.columns)
mdl.printNBmodel(nbProb, voteData.columns)