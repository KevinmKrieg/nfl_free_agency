import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#%matplotlib inline

data = pd.read_csv('contracts_players_perf.csv', index_col=0)

data['pos_mean'] = data['APY'].groupby(data['Position']).transform('mean')
data['pos_std'] = data['APY'].groupby(data['Position']).transform('std')

fapl = pd.read_excel('Free Agent Preliminary List.xlsx')
fapl['Name'] = fapl['goes_by'].map(str) + ' ' + fapl['last_name']
FAS_2018 = data[data['Name'].isin(fapl['Name'])]

predictors = ['Position', 'Role', 'Accrued','pos_mean', 'pos_std', 'weight', 'DOB', 'college',
              'Years',  'original_team_id', 'draft_year', 'draft_round', 'draft_pick',
              'overall', 'overall_rk',
              'pass', 'run', 'receiving', 'passblock', 'runblock', 'passrush',
              'rundefense', 'coverage', 'discipline', 'pass_snaps', 'receiving_snaps',
              'passblock_snaps', 'run_snaps', 'runblock_snaps', 'rundefense_snaps', 
              'passrush_snaps', 'coverage_snaps', 'total_snaps', 'games'] 

target = 'APY'

X = data[predictors]
Y = data[target]
X.columns[X.dtypes=='object']

pd.options.mode.chained_assignment = None

X = X.replace('\\N', np.nan)
X['Years']      = X['Years'].replace('s', np.nan).astype('float')
X['pass']       = X['pass'].astype('float')
X['run']        = X['run'].astype('float')
X['receiving']  = X['receiving'].astype('float')
X['passblock']  = X['passblock'].astype('float')
X['runblock']   = X['runblock'].astype('float')
X['passrush']   = X['passrush'].astype('float')
X['rundefense'] = X['rundefense'].astype('float')
X['coverage']   = X['coverage'].astype('float')
X['overall_rk'] = X['overall_rk'].astype('float')

X['DOB'] = (pd.datetime(2017, 1, 1) - pd.to_datetime(X.DOB)).apply(lambda x: np.nan if pd.isnull(x) else x.days/365) 

def fixHeight(x):
    if pd.isnull(x) or len(x)<2: return np.nan
    x = x.split('-')
    return int(x[0])*12 + int(x[1])

#X['height'] = X['height'].replace(' ', np.nan).apply(fixHeight).astype('float')

for col in X.columns[X.dtypes=='object']:
    print(col, ':', len(X[col].value_counts()))
    
X.college[X.college.isin(X.college.value_counts()[25:].index)] = np.nan
X = pd.concat([X, pd.get_dummies(X[['Position', 'Role', 'college']])], axis=1)

X = X.drop(['Position', 'Role', 'college'], axis=1)
for col in X.columns:
    if X[col].isnull().any():
        X[col+'_nan']= X[col].isnull()
        X[col] = X[col].fillna(X[col].mean())
        
Xtrain, Ytrain = X[data.year_signed < 2016],  Y[data.year_signed < 2016]/1000000
Xval,   Yval   = X[data.year_signed == 2016], Y[data.year_signed == 2016]/1000000
Xtest,  _      = X[data.year_signed == 2017], Y[data.year_signed == 2017]/1000000

train = Xtrain.copy(); train['APY_millions'] = Ytrain
val   = Xval.copy();   val['APY_millions']   = Yval
test  = Xtest.copy();

train.to_csv('train.csv')
val.to_csv('val.csv')
test.to_csv('test.csv')
rf = RandomForestRegressor(n_estimators=5000, n_jobs=-1)
rf.fit(Xtrain, Ytrain)

Ytrain_hat = rf.predict(Xtrain)
Yval_hat = rf.predict(Xval)
Ytest_hat = rf.predict(Xtest)

mse_train = mean_squared_error(Ytrain_hat, Ytrain)
mse_val = mean_squared_error(Yval_hat, Yval)
                             
print('------ Train ------')
print('RF:    MSE = %06.3f'%mse_train)
print('---- Validation ---')
print('Zeros: MSE = %06.3f'%mean_squared_error([0]*len(Ytrain), Ytrain))
print('Mean:  MSE = %06.3f'%mean_squared_error([Ytrain.mean()]*len(Ytrain), Ytrain))
print('RF:    MSE = %06.3f'%mse_val)