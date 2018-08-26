
import sys
DATADIR = "../../data/"
sys.path.append("../../data")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from TUP import full_data, consumption_data, regressions, reg_table, df_to_orgtbl, mobile_data
ITEMS = ["beans", "sugar", "fish", "nuts", "vegetables", "airtime", "fuel"]

D = full_data(DIR=DATADIR)
HH, T = consumption_data(D,WRITE=True)[1:] #"csv")
M, C,mHH= mobile_data(DIR = DATADIR+"Mobile/")
try: logL = pd.read_pickle(DATADIR+"ss-lambdas_mobile.df")
except EnvironmentError: raise IOError("Need to run SS-lambdas.py")
logL.index.names=["HH","Year","Location"]
logL.name       =["loglambda"]
C    = C.join(logL,how="left").rename(columns={"loglambda":"$\log\lambda_{it}$"})
C    = C.reorder_levels([1,0,2]).sortlevel()
keep = pd.notnull(C.index.get_level_values("Location"))
C    = C.loc[keep,:]
# Make aggregate variables
C["Tot"]    = C.filter(ITEMS).sum(axis=1).replace(0,np.nan)
C["logTot"] = C["Tot"].apply(np.log)
C           = C.join(T, how="left",lsuffix="_")
C['const']  = 1.

Outcomes =["Tot",  "logTot", "$\log\lambda_{it}$"]
Controls= ['const', 'TUP', 'CSH']

regs = regressions(C,outcomes=Outcomes, controls=Controls, Baseline=2013)
results, SE  = reg_table(regs,  resultdf=True,table_info=["N","F-stat"])
CTL = C["TUP"]+C["CSH"] ==0
CTLmean = {var: C.loc[CTL,var].mean() for var in Outcomes}
CTLsd = {var: C.loc[CTL,var].std() for var in Outcomes}
diff, diff_se = pd.DataFrame(CTLmean,index=["CTL mean"]), pd.DataFrame(CTLsd,index=["CTL mean"])

for var in Outcomes:
    ttest= regs[var].t_test("TUP - CSH = 0").summary_frame()
    diff.loc[   r"$\beta^{TUP}-\beta^{CSH}$", var] = ttest["coef"][0]
    diff_se.loc[r"$\beta^{TUP}-\beta^{CSH}$", var] = ttest["std err"][0]

results = results.append(diff)
SE = SE.append(diff_se)
mtab = df_to_orgtbl(results, sedf=SE)
