
import numpy as np
import pandas as pd
from pandas.io import stata
import statsmodels.api as sm
from matplotlib import pyplot as plt
import sys
sys.path.append("../../data")
from TUP import full_data, regressions, reg_table, df_to_orgtbl
"""
Note that topcoding has a large effect on the distribution here, and we see only a small (presumably non-random) portion of actual income for each household.
"""

# Top-Code or censor outliers?
def topcode(var, Nstd=3, drop=False):
    if drop: var[var>var.mean()+Nstd*var.std()] = np.nan
    else: var[var>var.mean()+Nstd*var.std()] = var.mean()+Nstd*var.std() 
    return var

D = full_data(balance=[])
keep = D.index

I_file = '../../data/Endline/sections_8_17.csv'
I = pd.read_csv(I_file).rename(columns={"id":"HH"}).set_index("HH", drop=True).ix[keep]

#~Getting non-agriculture income data is easy
I = I.filter(regex="^s16")
Imonths    = I.filter(regex="s16_\dc").rename(columns=lambda x: x[:-1])
Ipermonth  = I.filter(regex="s16_\dd").rename(columns=lambda x: x[:-1])
Income_12m = Imonths.mul(Ipermonth).sum(axis=1)
Iyear      = I.filter(regex="s16_\de").rename(columns=lambda x: x[:-1]).sum(axis=1)

A_file = "../../data/Endline/Agriculture_cleaned.csv"
A = pd.read_csv(A_file).rename(columns={"id":"HH"}).set_index("HH",drop=False).ix[keep]
unit_prices = A.groupby(["harvest_type", "harvest_price_unit"])["harvest_price"].median()
prices = unit_prices.loc[zip(A["harvest_type"],A["harvest_price_unit"])]
A["price"]=list(prices)

A["harvest_unit_match"] = A["harvest_price_unit"] == A["harvest_unit"]
A["price"] = A["harvest_unit_match"]*A["harvest_price"] + (1-A["harvest_unit_match"])*A["price"]

A["income_farm_year"] = A["harvest_size"]*A["price"]
Ayear = A.groupby("HH")["income_farm_year"].sum()

unit_prices = A.groupby(["livestock_type", "livestock_price_unit"])["livestock_price"].median()
prices = unit_prices.loc[zip(A["livestock_type"],A["livestock_price_unit"])]
A["price"]=list(prices)
A["livestock_unit_match"] = A["livestock_price_unit"] == A["livestock_unit"]
A["price"] = A["livestock_unit_match"]*A["livestock_price"] + (1-A["livestock_unit_match"])*A["price"]

A["income_livestock_year"] = A["livestock_size"]*A["price"]
Lyear = A.groupby("HH")["income_livestock_year"].sum()

Outcomes = ["Total", "Non-Farm", "Farm",  "Livestock"]
Controls = ["cons", "TUP","CSH"]
Vals = pd.DataFrame({"Non-Farm": Income_12m, "Farm":Ayear, "Livestock":Lyear})
Vals = Vals.apply(topcode)

Vals["Total"] = Vals.sum(axis=1)
Vals["cons"] = 1.

Vals = Vals.join(D[["TUP","CSH"]],how="left")
Vals["CTL"] = (Vals["TUP"]+Vals["CSH"] ==0).apply(int)

#~ Make graph of distribution
stringify = lambda var: Vals[var].apply(lambda x: var if x else "")
Vals["Group"] = stringify("TUP")+stringify("CSH")+stringify("CTL")
Vals.dropna(subset=[["Total","TUP","CSH","CTL"]]).groupby("Group")["Total"].plot(kind="kde")
plt.title("Total Income Distribution by Group")
plt.savefig("../figures/IncomeDistribution.png")
plt.clf()
#~ Make bar graphs
Imean  = Vals.groupby("Group").mean()[Outcomes]
Icount = Vals.groupby("Group").count()[Outcomes]
Istd = Vals.groupby("Group").std()[Outcomes]
Ise=Istd/np.sqrt(Icount)
Imean.T.plot(kind="bar",yerr=Ise.T)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('../figures/Income_group.png')
plt.clf()

regs = {var: sm.OLS(Vals[var], Vals[Controls], missing="drop").fit() for var in Outcomes}
results, SE  = reg_table(regs,  resultdf=True,table_info=["N","F-stat"])

CTL = Vals["CTL"] 
CTLmean = {var: Vals.query("CTL==1")[var].mean() for var in Outcomes}
CTLsd = {var: Vals.query("CTL==1").std() for var in Outcomes}
diff, diff_se = pd.DataFrame(CTLmean,index=["CTL mean"]), pd.DataFrame(CTLsd,index=["CTL mean"])

for var in Outcomes:
    ttest1= regs[var].t_test("TUP - CSH = 0").summary_frame()

    diff.loc[   r"$\beta^{TUP}-\beta^{CSH}$", var] = ttest1["coef"][0]
    diff_se.loc[r"$\beta^{TUP}-\beta^{CSH}$", var] = ttest1["std err"][0]

results = results.append(diff)
SE = SE.append(diff_se)

tab = df_to_orgtbl(results, sedf=SE)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
sys.path.append("../../data")
from TUP import full_data, regressions, asset_vars, reg_table , df_to_orgtbl

# Top-Code or censor outliers?
def topcode(var, Nstd=3, drop=False):
    if drop: var[var>var.mean()+Nstd*var.std()] = np.nan
    else: var[var>var.mean()+Nstd*var.std()] = var.mean()+Nstd*var.std() 
    return var

#~ Read in data
D = full_data("../../data/TUP_full.dta", balance=[])
D = D[D.merge_midline != 1]
C = D.filter(like="conflict").rename(columns = lambda x: x[:-2]) #~ Set up empty DataFrame to fill

#~ Make Outcome variables
#~ NOTE: Not looking at whether they protected assets as only 50 #~ said they did, and all said "Migrated to stay with other family"
C["Bsln_NaN"] = D["merge_midline"] == 2
C["Worried"]  = C["conflict_worried"]
C["Affected"] = C["conflict_affected"]
protect_lives_codes = lambda x: {"Nothing": 0, "Migrate to stay with friend/family": 1, "Migrated and found new accommodation": 1,
                                 "Looked for protection with Govt. Military": 2, "Looked for protection with NGO": 2}.get(x)
C["ProtectLives"] = C["conflict_protectlives"].apply(protect_lives_codes)
C["Migrated"] = (C["ProtectLives"]==1) + \
                (C["conflict_affected1"]=="Needed to elocate or migrate")
C["NoMeans"] = C["conflict_whynotprotect"]=="Didn't have the means"
C["NoInvest"]= C.filter(like="affected").applymap(lambda x: x=="Could not plant crop or invest in business").sum(axis=1)
C = C.drop([var for var in C if var.startswith("conflict_")], 1)
Outcomes = ["ProtectLives", "Worried", "Affected", "Migrated", "NoMeans", "NoInvest"]

#~ Bring in Treatment variables
C["TUP"] = D["TUP"]
C["cons"] = 1.
C = C.applymap(lambda x: int(x) if not np.isnan(x) else x)
Controls = ["cons", "TUP"] #~, "Bsln_NaN"]

#~ Plot
byT=C.groupby("TUP")
Cbar=byT.mean().drop(["Bsln_NaN",'cons'],1)
Cstd=byT.std().drop(["Bsln_NaN",'cons'],1)
Cn=byT.count().drop(["Bsln_NaN",'cons'],1)
for df in (Cbar,Cstd,Cn): 
        df.index=["CTL","TUP"]
        df.index.name="Group"
Cse=Cstd/np.sqrt(Cn)
Cbar.T.plot(kind="bar",yerr=Cse.T)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig("../figures/conflict_exposure.png")
plt.clf()


C_regs = regressions(C, outcomes=Outcomes, controls=Controls, Baseline=False, baseline_na=False)
C_results, C_SE  = reg_table(C_regs,  resultdf=True,table_info=["N","F-stat"])

#~ Get control group means and standard deviations and add to regression table
CTLmean = {var: C.query("TUP==0")[var].mean() for var in Outcomes}
CTLsdv  = {var: C.query("TUP==0")[var].std()  for var in Outcomes}
CTLmean, CTLsdv = pd.DataFrame(CTLmean,index=["CTL mean"]), pd.DataFrame(CTLsdv,index=["CTL mean"])
C_results = C_results.append(CTLmean)
C_SE      = C_SE.append(CTLsdv)
C_results.drop('cons',inplace=True)
C_SE.drop('cons',inplace=True)

Table = df_to_orgtbl(C_results, sedf=C_SE)
return Table

import numpy as np
 import pandas as pd
 import statsmodels.api as sm
 import sys
 DATADIR = "../../data/"
 sys.path.append(DATADIR)
 from TUP import full_data, regressions, asset_vars, reg_table , df_to_orgtbl
 D = full_data(DIR = DATADIR,balance=[])

 Decide14 = D.filter(regex="^decide_.*_m").rename(columns=lambda x: x[:-2].split("_")[-1])#~.applymap(recode)
 Decide15 = D.filter(regex="^decide_.*_e").rename(columns=lambda x: x[:-2].split("_")[-1])#~.applymap(recode)

{'decide_arguments',
 'decide_fearful',

 'decide_pressuretospend'}

 who_decides = ['dailypurchase', 'familyvisits', 'healthchild', 'healthown', 'majorpurchase', 'moneyown']
 who_recode  = lambda x: float("Herself" in x) if pd.notnull(x) else x

 if WEEKLY:
     weekly = lambda i: float(i<3) if pd.notnull(i) else np.nan
     Aval2013 = Aval2013.applymap(weekly)
     Aval2014 = Aval2014.applymap(weekly)      
     Aval2015 = Aval2015.applymap(weekly)
 
 index_vars = "worried,portions,fewmeals,nofood,hungry,wholeday".split(",")
 Outcomes = index_vars+["z-score"]
 #~ Creates Year dummies, z-scores and baseline values as `var'2013
 for Year, Aval in zip((2013, 2014, 2015), (Aval2013, Aval2014, Aval2015)):
     Aval["Year"]=Year
     if not weekly:
        for var in index_vars:
            Aval[index_vars] = (Aval[index_vars]-Aval[index_vars].mean())/Aval[index_vars].std()
     FS_sum = Aval[index_vars].sum(axis=1)
     Aval["z-score"] = (FS_sum-FS_sum.mean())/FS_sum.std()
     for var in Outcomes: Aval[var+"2013"] = Aval2013[var]
    
 Vals = pd.concat((Aval2013, Aval2014, Aval2015)).reset_index().set_index(["Year", "HH"], drop=False)
 Vals = Vals.join(pd.get_dummies(Vals["Year"]).rename(columns=lambda col: str(int(col))), how="left")
 Vals = Vals.join(D[["TUP","CSH"]])

 for group in ("TUP", "CSH"):
     for year in ("2013", "2014", "2015"):
         Vals["{}*{}".format(group,year)] = Vals[group]*Vals[year]

 Controls = ['2014', '2015', 'TUP*2014', 'TUP*2015', 'CSH*2014', 'CSH*2015']

 #~ This is the main specification. Given the mismatch in timing, we compare CSH*2015 to both TUP*2014 and TUP*2015
 Vals=Vals.loc[2014:2015]
 regs = regressions(Vals, outcomes=Outcomes, controls=Controls, Baseline=2013, baseline_na=True)

 results, SE  = reg_table(regs,  resultdf=True,table_info=["N","F-stat"])

 CTL = Vals["TUP"]+Vals["CSH"] ==0
 CTLmean = {var: Vals[CTL].loc[2015,var].mean() for var in Outcomes}
 CTLsd = {var: Vals[CTL].loc[2015,var].std() for var in Outcomes}
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

import numpy as np
 import pandas as pd
 import statsmodels.api as sm
 import sys
 DATADIR = "../../data/"
 sys.path.append(DATADIR)
 from TUP import full_data, regressions, asset_vars, reg_table , df_to_orgtbl
 D = full_data(DIR = DATADIR,balance=[])

 autonomy_codes = {'go_NGO',
                   'go_church',
                   'go_healthcenter',
                   'go_housenonrelative',
                   'go_houserelative',
                   'go_market',
                   'go_school',
                   'go_water'}

 codes = {"1-2 times a week": 3,
         "3-6 times a week": 2,
         "Everyday": 1,
         "everyday": 1,
         "Less than once a week": 4,
         "less than once a week": 4,
         "Never": 5,
         "never": 5}

 recode = lambda x: codes.setdefault(x,x)

 Autnmy14 = D.filter(regex="^cango_.*_m").rename(columns=lambda x: x[3:-2]).applymap(recode)
 Autnmy15 = D.filter(regex="^cango_.*_e").rename(columns=lambda x: x[3:-2]).applymap(recode)

{'decide_arguments',
 'decide_fearful',

 'decide_dailypurchase',
 'decide_familyvisits',
 'decide_healthchild',
 'decide_healthown',
 'decide_majorpurchase',
 'decide_moneyown',
 'decide_pressuretospend'}

 Decide14 = D.filter(regex="^decide_.*_m").rename(columns=lambda x: x[:-2])#~.applymap(recode)
 Decide15 = D.filter(regex="^decide_.*_e").rename(columns=lambda x: x[:-2])#~.applymap(recode)

 Confid14 = D.filter(regex="^conf_.*_m").rename(columns=lambda x: x[3:-2]).applymap(recode)
 Confid15 = D.filter(regex="^conf_.*_e").rename(columns=lambda x: x[3:-2]).applymap(recode)

 if WEEKLY:
     weekly = lambda i: float(i<3) if pd.notnull(i) else np.nan
     Aval2013 = Aval2013.applymap(weekly)
     Aval2014 = Aval2014.applymap(weekly)      
     Aval2015 = Aval2015.applymap(weekly)
 
 index_vars = "worried,portions,fewmeals,nofood,hungry,wholeday".split(",")
 Outcomes = index_vars+["z-score"]
 #~ Creates Year dummies, z-scores and baseline values as `var'2013
 for Year, Aval in zip((2013, 2014, 2015), (Aval2013, Aval2014, Aval2015)):
     Aval["Year"]=Year
     if not weekly:
        for var in index_vars:
            Aval[index_vars] = (Aval[index_vars]-Aval[index_vars].mean())/Aval[index_vars].std()
     FS_sum = Aval[index_vars].sum(axis=1)
     Aval["z-score"] = (FS_sum-FS_sum.mean())/FS_sum.std()
     for var in Outcomes: Aval[var+"2013"] = Aval2013[var]
    
 Vals = pd.concat((Aval2013, Aval2014, Aval2015)).reset_index().set_index(["Year", "HH"], drop=False)
 Vals = Vals.join(pd.get_dummies(Vals["Year"]).rename(columns=lambda col: str(int(col))), how="left")
 Vals = Vals.join(D[["TUP","CSH"]])

 for group in ("TUP", "CSH"):
     for year in ("2013", "2014", "2015"):
         Vals["{}*{}".format(group,year)] = Vals[group]*Vals[year]

 Controls = ['2014', '2015', 'TUP*2014', 'TUP*2015', 'CSH*2014', 'CSH*2015']

 #~ This is the main specification. Given the mismatch in timing, we compare CSH*2015 to both TUP*2014 and TUP*2015
 Vals=Vals.loc[2014:2015]
 regs = regressions(Vals, outcomes=Outcomes, controls=Controls, Baseline=2013, baseline_na=True)

 results, SE  = reg_table(regs,  resultdf=True,table_info=["N","F-stat"])

 CTL = Vals["TUP"]+Vals["CSH"] ==0
 CTLmean = {var: Vals[CTL].loc[2015,var].mean() for var in Outcomes}
 CTLsd = {var: Vals[CTL].loc[2015,var].std() for var in Outcomes}
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
