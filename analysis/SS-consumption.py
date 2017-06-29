
import sys
DATADIR = "../../data/"
sys.path.append("../../data")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from TUP import full_data, consumption_data, regressions, reg_table, df_to_orgtbl, mobile_data
food = ['cereals', 'maize', 'sorghum', 'millet', 'potato', 'sweetpotato', 'rice', 'bread', 'beans', 'oil', 'salt', 'sugar', 'meat', 'livestock', 'poultry', 'fish', 'egg', 'nuts', 'milk', 'vegetables', 'fruit', 'tea', 'spices', 'alcohol', 'otherfood']
month = ['fuel', 'medicine', 'airtime', 'cosmetics', 'soap', 'transport', 'entertainment', 'childcare', 'tobacco', 'batteries', 'church', 'othermonth']    
year = ['clothesfootwear', 'womensclothes', 'childrensclothes', 'shoes', 'homeimprovement', 'utensils', 'furniture', 'textiles', 'ceremonies', 'funerals', 'charities', 'dowry', 'other']    

D = full_data(DIR=DATADIR)
C, HH, T = consumption_data(D,WRITE=True) #"csv")
logL = pd.read_pickle(DATADIR + "ss-lambdas.df")
logL.index.names=["HH","Year","Location"]
C = C.join(logL,how="left").rename(columns={"loglambda":"$\log\lambda_{it}$"})
C = C.reorder_levels([1,2,0]).sortlevel()
keep = pd.notnull(C.index.get_level_values("Location"))
C = C.loc[keep,:]

# Make aggregate variables
C["Food"]   = C.filter(items=food).sum(axis=1).replace(0,np.nan)
C["Month"]   = C.filter(items=food).sum(axis=1)
C["Year"]   = C.filter(items=food).sum(axis=1)
C["Tot"]    = C[["Food","Month","Year"]].sum(axis=1).replace(0,np.nan)
C["z-score"]  = (C["Tot"]-C["Tot"].mean())/C["Tot"].std()
C["FoodShr"]= C["Food"].div(C["Tot"]) #$\approx$ FoodShare variable
C["logTot"] = C["Tot"].apply(np.log)
C = C.join(T, how="left",lsuffix="_")

Outcomes = ["Tot","FoodShr", "Food", "z-score", "$\log\lambda_{it}$"]

#$\approx$ Make Baseline variable
for var in Outcomes: 
    Bl = C.loc[2013,var].reset_index("Location",drop=True)
    #if var in mC: mC = mC.join(Bl,rsuffix="2013", how="left")
    C = C.join(Bl,rsuffix="2013", how="left")


C["Y"]=np.nan
for yr in (2013, 2014, 2015): C.loc[yr,"Y"]=str(int(yr))

C = C.join(pd.get_dummies(C["Y"]), how="left",lsuffix="_")
for group in ("TUP", "CSH"):
    for year in ("2013", "2014", "2015"):
        interaction = C[group]*C[year]
        if interaction.sum()>0: C["{}*{}".format(group,year)] = interaction
Controls = ["2014","2015", 'TUP*2014', 'CSH*2014', 'TUP*2015', 'CSH*2015']
#~ This is the main specification. Given the mismatch in timing, we compare CSH*2015 to both TUP*2014 and TUP*2015
C = C.loc[2014:2015]
regs  = regressions(C, outcomes=Outcomes,  controls=Controls,  Baseline=2013, baseline_na=True)

results, SE  = reg_table(regs,  resultdf=True,table_info=["N","F-stat"])

CTL = C["TUP"]+C["CSH"] ==0
CTLmean = {var: C[CTL].loc[2015,var].mean() for var in Outcomes}
CTLsd = {var: C[CTL].loc[2015,var].std() for var in Outcomes}
diff, diff_se = pd.DataFrame(CTLmean,index=["CTL mean"]), pd.DataFrame(CTLsd,index=["CTL mean"])

for var in Outcomes:
    ttest1= regs[var].t_test("TUP*2014 - CSH*2015 = 0").summary_frame()
    ttest2= regs[var].t_test("TUP*2015 - CSH*2015 = 0").summary_frame()

    diff.loc[   r"$\beta^{TUP}_{2014}-\beta^{CSH}$", var] = ttest1["coef"][0]
    diff_se.loc[r"$\beta^{TUP}_{2014}-\beta^{CSH}$", var] = ttest1["std err"][0]

    diff.loc[   r"$\beta^{TUP}_{2015}-\beta^{CSH}$", var] = ttest2["coef"][0]
    diff_se.loc[r"$\beta^{TUP}_{2015}-\beta^{CSH}$", var] = ttest2["std err"][0]

results = results.append(diff)
SE = SE.append(diff_se)

tab = df_to_orgtbl(results, sedf=SE)
