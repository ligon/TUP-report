
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

def align_indices(df1,df2):
   """
   Reorder levels of df2 to match that of df1
   Must have same index.names
   """
   I1, I2 = df1.index, df2.index
   try: assert(not set(I1.names).difference(I2.names))
   except AssertionError: raise ValueError("Index names must be the same")
   new_order = []
   for lvl in I1.names: new_order.append(I2.names.index(lvl))
   df2 = df2.reorder_levels(new_order)
   return df1, df2
def winsorize(Series, **kwargs):
   """
   Need to implement two-sided censoring as well.
   WARNING: if Top<0, all zeros will be changed to Top
   """

   percent    = kwargs.setdefault("percent",99)
   stdev      = kwargs.setdefault("stdev",False)
   drop       = kwargs.setdefault("drop",False)
   drop_zeros = kwargs.setdefault("drop_zeros",True)
   twoway     = kwargs.setdefault("twoway",False)

   if drop_zeros: S = Series.replace(0,np.nan).dropna()
   else: S = Series.dropna()
   N_OBS = S.notnull().sum()
   if N_OBS<10: return S

   if percent: Top = np.percentile(S, percent)
   if stdev:   
       Top =  S.dropna().mean()
       Top += stdev*S.dropna().std()
   try: assert((not drop_zeros) or Top>0)
   except AssertionError: raise ValueError("Top < 0 but zeros excluded")
   if drop: replace_with = np.nan
   else:    replace_with = Top

   _Series = Series.copy()
   _Series[_Series>Top]=replace_with

   if not twoway: return _Series
   else:
       kwargs['twoway'] = False
       return -1*winsorize(-1*_Series, **kwargs)
def USD_conversion(Exp,exchange_rate=1.,PPP=1.,inflation=1.,time='Year'):
   """
   Convert nominal local currency into price- and inflation-adjusted USD

   Exp - A numeric or pd.Series object 
   exchange_rate - Taken as LCU/USD. 
   PPP - Taken as $Real/$nominal
   inflation - Taken as % inflation compared to some baseline.
   time - If a list is passed, `time' indicates the name or position of the time level in Exp.index
       NOTE: This has to be a cumulative number, so if inflation is 20% for two straight years, that year should be divided by (1+.2)**2
   Final calculation will basically be Exp_usdppp = Exp*(exchange_rate*PPP)/inflation

   if pd.Series are passed for any kwarg, index name needs to be in the multi-index of Exp.
   """
   if type(inflation)==list: inflation=[1./i for i in inflation]
   else: inflation = 1/inflation
   if type(exchange_rate)==list: exchange_rate=[1./i for i in exchange_rate]
   else: exchange_rate = 1/exchange_rate
   
   _Exp = Exp.copy()
   VARS = (exchange_rate, PPP,inflation)
   if list in map(type,VARS):
       if time in _Exp.index.names: time=_Exp.index.names.index(time)
       time = _Exp.index.levels[time]
   for var in VARS:
       if type(var)==list: var=pd.Series(var,index=time)
       try: _Exp = _Exp.mul(var)
       except ValueError: #~ If Series index doesn't have a name, try this...
           var.index.name = var.name
           _Exp = _Exp.mul(var)
   return _Exp
def percapita_conversion(Exp,HH,children=["boys","girls"],adult_equivalent=1.,minus_children='hh_size'):
   """
   Returns household per-capita expenditures given:
       `Exp'- Total household expenditures
       `HH' - Total number of individuals in the household
           If HH is a pd.DataFrame, Exp is divided by HH.sum(1)
           if `children' is the name of a column or a list of column names, 
           those first get divided by the factor adult_equivalent
   """
   try: HH.columns #~ If HH is a series, just divide though
   except AttributeError: return Exp.div(HH)
   _HH = HH.copy()
   if type(children)==str: children=[children]
   children = _HH.columns.intersection(children).tolist()
   if minus_children: _HH[minus_children] -= _HH[children].sum(1)
   if children: _HH[children] *= adult_equivalent
   Exp,_HH = align_indices(Exp,_HH)
   return Exp.div(_HH.sum(1).replace(0,1))

#~ Source: http://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=SS&name_desc=false
xrate = [ 2.161, 2.196, 3.293] #~ To avoid confusion, using PPP adjusted xrate and just setting PPP=1.
PPP = 1.
inflation= 1. #~ Bank data uses international $, which is inflation adjusted.
C["Exp_usd"] = winsorize(USD_conversion(C["Tot"],exchange_rate=xrate,PPP=PPP,inflation=inflation))
C["Tot_pc"] = percapita_conversion(C["Exp_usd"],HH,adult_equivalent=.5)
#C["Exp_usdpc_tc"] = winsorize(C["Exp_usdpc"])


C["z-score"]  = (C["Tot"]-C["Tot"].mean())/C["Tot"].std()
C["FoodShr"]= C["Food"].div(C["Tot"]) #$\approx$ FoodShare variable
C["logTot"] = C["Tot"].apply(np.log)
C = C.join(T, how="left",lsuffix="_")

Outcomes = ["Tot","FoodShr", "Food", "$\log\lambda_{it}$","z-score"]

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
