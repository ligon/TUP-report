
import numpy as np
import pandas as pd
import cfe.estimation as nd
import statsmodels.api as sm
import sys
DATADIR = "../../data/"
sys.path.append("../../data")
from TUP import full_data, consumption_data, regressions, reg_table, df_to_orgtbl, mobile_data
food =  ['cereals', 'maize', 'sorghum', 'millet', 'potato', 'sweetpotato', 'rice', 'bread', 'beans', 'oil', 'salt', 'sugar', 'meat', 'livestock', 'poultry', 'fish', 'egg', 'nuts', 'milk', 'vegetables', 'fruit', 'tea', 'spices', 'alcohol', 'otherfood']
month = ['fuel', 'medicine', 'airtime', 'cosmetics', 'soap', 'transport', 'entertainment', 'childcare', 'tobacco', 'batteries', 'church', 'othermonth']    
ConsumptionItems = food+['airtime','fuel']
mobile=True

D = full_data(DIR=DATADIR)
C, HH, T = consumption_data(D,WRITE=False,include2016=False)
HH['log HHSIZE'] = HH["hh_size"].apply(np.log)
HH = HH.drop("hh_size",1)
y,z = C.replace(0,np.nan).apply(np.log).sort_index(level=[0,1,2])[ConsumptionItems].copy(),HH.sort_index(level=[0,1,2]).copy()
y.index.names, z.index.names = ['j','t','mkt'], ['j','t','mkt']
keep = pd.notnull(y.index.get_level_values("mkt"))
y,z = y.loc[keep,:].align(z,join="left",axis=0)
b,ce,d,sed= nd.estimate_reduced_form(y,z,return_se=True,VERBOSE=True)
ce = ce.dropna(how='all')
print("Getting Loglambdas")
bphi,logL=nd.get_loglambdas(ce,TEST="warn")
try:
   xrange
   logL.to_pickle(DATADIR + "ss-lambdas.df")
except NameError: logL.to_pickle(DATADIR + "ss-lambdas3.df")

if mobile:
    M,Mc,Mhh = mobile_data(use_dates=True,DIR = DATADIR+"Mobile/")
    y = Mc.replace(0,np.nan).apply(np.log).sort_index(level=[0,1,2]).filter(items=ConsumptionItems).copy()
    z = Mhh.sort_index(level=[0,1,2]).copy()
    y.index.names, z.index.names = ['j','t','mkt'], ['j','t','mkt']
    keep = pd.notnull(y.index.get_level_values("mkt"))
    y,z = y.loc[keep,:].align(z,join="left",axis=0)
    b,ce,d,sed= nd.estimate_reduced_form(y,z,return_se=True,VERBOSE=True)
    ce = ce.dropna(how='all')
    print("Getting Loglambdas")
    Mbphi,MlogL=nd.get_loglambdas(ce,TEST="warn")
    MlogL -= MlogL.mean()
    MlogL /= MlogL.std()
    MlogL = MlogL.unstack('t').drop('4February',1).stack()
    try:
      xrange
      MlogL.to_pickle(DATADIR + "ss-lambdas_mobile.df")
    except NameError: MlogL.to_pickle(DATADIR + "ss-lambdas_mobile3.df")
