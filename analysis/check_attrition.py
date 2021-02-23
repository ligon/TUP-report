#!/usr/bin/env python

import sys
DATADIR = "../../TUP-data/"
sys.path.append(DATADIR)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind
from TUP import full_data, consumption_data, asset_vars

def df_to_orgtbl(df,tdf=None,sedf=None,float_fmt='%5.3f'):
    """
    Print pd.DataFrame in format which forms an org-table.
    Note that headers for code block should include ':results table raw'.
    """
    if len(df.shape)==1: # We have a series?
       df=pd.DataFrame(df)

    if (tdf is None) and (sedf is None):
        return '|'+df.to_csv(sep='|',float_format=float_fmt,line_terminator='|\n|')
    elif not (tdf is None) and (sedf is None):
        s = '| |'+'|  '.join(df.columns)+' |\n|-\n'
        for i in df.index:
            s+='| %s ' % i
            for j in df.columns:
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                if np.isnan(df.loc[i,j]): entry='| $ $ '
                else: entry='| $'+float_fmt+stars+'$ '
                s+=entry % df.loc[i,j]
            s+='|\n'
        return s

    elif not sedf is None: # Print standard errors on alternate rows
        s = '| |'+'|  '.join(df.columns)+' |\n|-\n'
        for i in df.index:
            s+='| %s ' % i
            for j in df.columns: # Point estimates
                try:
                    stars = (np.abs(df.loc[i,j]/sedf.loc[i,j])>1.65) + 0.
                    stars+= (np.abs(df.loc[i,j]/sedf.loc[i,j])>1.96) + 0.
                    stars+= (np.abs(df.loc[i,j]/sedf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                if np.isnan(df.loc[i,j]): entry='| $ $ '
                else: entry='| $'+float_fmt+stars+'$ '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now standard errors
                s+=' '
                try:
                    if not np.isnan(sedf.loc[i,j]):
                        se='$(' + float_fmt % sedf.loc[i,j] + ')$' 
                        entry='| '+se+' '
                    else: entry='| '
                except KeyError: entry='| '
                s+=entry 
            s+='|\n'
        return s
def add_interactions(DF,varnames, overwrite=False):
    """
    Returns DF with specified interaction terms.
    varnames is a list of variable names connected by * or :
    The product of the two variables is added to DF
    if * -- adds the un-interacted variables to varnames
    if : -- changes it to * but does not add un-interacted variables
    returns DF *and* varnames
    """
    if type(varnames) not in (list, tuple): varnames=[varnames]
    Vars = varnames[:]
    #~ Crazy inefficient to do this?
    if not overwrite: DF = DF.copy()
    for v in Vars:
        #if (v in DF) and (not overwrite): continue
        if "*" in v:
            i,j = v.split("*")
            for k in (i,j):
                if k not in Vars: Vars.append(k)
            DF[v] = DF[i]*DF[j]
        elif ":" in v:
            i,j = v.split(":")
            #~ varnames.remove(v)
            #~ v = i+"*"+j
            #~ varnames.append(v)
            DF[v] = DF[i]*DF[j]
        else: pass
    return DF, Vars

def ttest_table(D,vardict,groups=["TUP","CSH"], p_stars=(.1, .05, .01)):
    for group in groups: vardict.setdefault(group,group)
    T = D[vardict.keys()]
    T.rename(columns=vardict,inplace=True)
    #~ There's got to be a better way to invert the get_dummies function
    T["group"] = ""
    for group in groups: T["group"]+= [group if i else "" for i in T[group]]
    T["group"]=[i if i else "CTL" for i in T["group"]]
    treat = {group: df.drop(groups+["group"],1) for group,df in T.groupby("group")}
    Table = T.groupby('group').mean().T
    
    for var in Table: Table[var] = Table[var].apply(lambda x: round(x,3))

    for group in groups:
        Table['$\Delta${}'.format(group)] = map(str,Table[group]-Table[1.0])
        Table['$p$_{{{}}}'.format(group)] = 0
    Table['$N$']=(T>0).sum()

    for group in groups:
        for var in T:
            if var in groups+["group"]: continue
            pval = ttest_ind(treat[group][var].dropna(), treat["CTL"][var].dropna())[1]
            pval = round(pval,3)
            Table.ix[var,'$p$_{{{}}}'.format(group)]+= pval
            nstar=sum(pval<threshold for threshold in p_stars)
            if nstar: Table.ix[var,'$\Delta${}'.format(group)]+="^{{{}}}".format("*"*nstar)
    DROP=groups+["$p$_{{{}}}".format(group) for group in groups]
    few=T.shape[0]/15.
    Table = Table[Table['$N$']>few]
    return Table.drop(DROP,1)

def attrition_table(D,vardict,rounds=["In14","In15"], p_stars=(.1, .05, .01)):
    """
    Take a DataFrame of baseline variables and a set of indicator variables for whether they DID end up in each subsequent round.

    For each round, t, Estimate the linear regression Outcome = a + B*(Was in Round t) + e

    """
    vardict.setdefault(group,group)
    T = D[vardict.keys()]
    T.rename(columns=vardict,inplace=True)
    #~ There's got to be a better way to invert the get_dummies function
    Table = T.groupby(group).mean().T
    Table["BL mean"] = T.mean().T
    
    for var in Table: Table[var] = Table[var].apply(lambda x: round(x,3))

    Table['$\Delta$ {}'.format(group)] = Table[3.0] - Table["BL mean"]# map(str,Table-Table[1.0])
    Table['$N$']=(T>0).sum()

    for var in T:
        if var==group: continue
        pval = ttest_ind(T.loc[T.In15==3,var].dropna(), T.loc[:,var].dropna())[1]
        pval = round(pval,3)
        Pvals['var'] = str(pval)
        nstar=sum(pval<threshold for threshold in p_stars)
        if nstar: Pvals[var]+="^{{{}}}".format("*"*nstar)
    Table.ix[var,'$p$_{{{}}}'.format(group)]+= pval
    DROP=groups+["$p$_{{{}}}".format(group) for group in groups]
    few=T.shape[0]/15.
    Table = Table[Table['$N$']>few]
    return Table.drop(DROP,1)

def regressions(DF,outcomes=[],controls=[], **kwargs):
    """ Runs a set of regressions and return a dict of {Outcome: sm.OLS (or RLM) model} for each model
     DF:
         The full dataset with outcomes and control variables.
     Year:
         A suffix on each outcome variable, specifying which round of data is being used. (Default to "")
     Baseline:
         A suffix on each variable to be used as a baseline covariate, specifying which round of data is being used.
         If the outcome variable doesn't have a corresponding column with that suffix, no baseline control is included and the function passes without error.
         (Default to "_bsln")
     Controls:
         A list or tuple of variables to be used as covariates in each regression.
     Outcomes:
         The list of outcomes (also the names of the models)
     rhs_extra: (Optional)
         A dictionary of covariates to be added to the regression for specific outcomes.
         So of the form {outcome: [list of controls]} for each outcome.
     Baseline_na:
         If True, code missing values of baseline variable as zero and include a "Bsln_NAN" indicator in Controls.
     Robust:
         If True, use statsmodel's RLM class instead of OLS (defaults to Huber-T se's)
     Cluster:
         If a variable name (a string) is given, clusters standard errors at that level
         If a dictionary of {outcome:level} is given, clusters standard errors at that level
            (applies statsmodels .get_robustcov_results())
     fe:
         If fe is a (non-empty) list
     drop_var:
         Statsmodels assumes a constant is present when prompted for a robust covariance matrix
         If a full set of indicators (say, year dummies) is supplied instead, doesn't know how to drop one like Stat does
         If supplied, drop_var manually specifies which variable gets dropped. (e.g. baseline year dummy, control group dummy, etc.)
     names:
         Alternative names for the models other than the name of the outcome variable
         Takes a dict with names as keys and outcomes as values.
         If not a dict, gets converted into one.
     Return:
         dict {outcome var:model} for each outcome in outcomes.
    """
    #~ Kwargs
    Year        = kwargs.setdefault("Year",  "")
    Baseline    = kwargs.setdefault("Baseline",  "_bsln")
    rhs_extra   = kwargs.setdefault("rhs_extra", {})
    baseline_na = kwargs.setdefault("baseline_na", True)
    robust      = kwargs.setdefault("robust",    False)
    interactions= kwargs.setdefault("interactions", True)
    cluster     = kwargs.setdefault("cluster", None)
    drop_var    = kwargs.setdefault("drop_var", None)
    names       = kwargs.setdefault("names", [])
    fe          = kwargs.setdefault("fe", [])
    
    if fe: regress = lambda y,X: fe_reg(y,X,group=fe)
    elif robust: regress=sm.RLM
    else: regress=sm.OLS
    if not type(Year)==str: Year=str(Year)
    if not type(Baseline)==str: Baseline=str(Baseline)

    models = {}
    if interactions: DF,ctls = add_interactions(DF,controls)
    if cluster: #~ Can optionally provide a dict relating each outcome to a different clustering scheme.
        if type(cluster)!=dict: cluster={outcome:cluster for outcome in outcomes}

    for outcome in outcomes: #~ Run regressions and store models in a dictionary
        Xvars = ctls[:]

        #~ Add baseline value as control if present
        if outcome+Baseline in DF: #~ Present in DataFrame
            if DF[outcome+Baseline].count(): #~ Contains non-null values
                Xvars.append(outcome+Baseline)

        if outcome in rhs_extra: #~ If additional controls specified, make sure they're in there and add them
            if not type(rhs_extra[outcome]) in (list,tuple): rhs_extra[outcome] = [rhs_extra[outcome]] #~ Make into a list if not already.
            for x in rhs_extra[outcome]: #~ Check that it's present. (Warn user if not)
                try: assert(x in DF)
                except AssertionError: raise KeyError("Extra Covariate --{}-- for outcome --{}-- not found in data".format(x,outcome))
            if interactions: DF, rhs_extra[outcome] = add_interactions(DF, rhs_extra[outcome])
            Xvars += list(rhs_extra[outcome])
        othervars = [outcome+Year]
        if cluster and (cluster[outcome] not in othervars): othervars.append(cluster[outcome])
        
        df = DF[othervars +Xvars].rename(columns={outcome+Baseline:Baseline}) #~ if cluster: include cluster var in df

        #~ Include baseline==missing indicator as a control
        if Baseline in df and baseline_na:
            df["Bsln_NAN"] = df["Bsln"+Baseline].isnull().apply(int)
            df[Baseline].fillna(0,inplace=True)
            Xvars.append("Bsln_NAN")

        df = df.dropna()

        #~ Full-sample OLS
        models[outcome] = regress(df[outcome+Year], df[Xvars]).fit()
        if cluster:      #~ If specified, cluster standard errors
            if drop_var: #~ If specified, assign variable to be dropped
                drop_idx = df[Xvars].columns.tolist().index(drop_var)
                models[outcome].model.data.const_idx=drop_idx
            models[outcome] = models[outcome].get_robustcov_results(cov_type="cluster", groups=df[cluster[outcome]])
        del df, Xvars #~ remove from memory; just housekeeping, shouldn't be strictly necessary
    if names:
        if type(names)==str: names = {outcome+names:outcome for outcome in outcomes}
        if type(names) in (list,tuple): names=dict(zip(names,outcomes))
        for name in names.keys(): models[name]=models.pop(names[name])
        

    return models
def reg_table(models,**kwargs):
    """ Take a list or dict of sm.RegressionResults objects and create a nice table.
     Summary: (Default)
       If True, return a summary_col object (from sm.iolib.summary2), which allows for as_text and as_latex
     Orgtbl:
       If True, return an orgtable (uses df_to_orgtbl) for the OLS model params.
     Resultdf:
       Returns the coefficient and SE df's for modification and subsequent entry into df_to_orgtbl.
       Useful for adding other columns/rows, like control-group means
     table_info:
       A list of model statistics that can be included at the bottom (like with stata's esttab)
       Allows for "N", "R2", "R2-adj", "F-stat"
       Defaults to just "N"
     Transpose:
       Places outcomes on left with regressors on top.
    """

    summary    = kwargs.setdefault("summary",   True)
    orgtbl     = kwargs.setdefault("orgtbl",    False)
    resultdf   = kwargs.setdefault("resultdf",  False)
    table_info = kwargs.setdefault("table_info", "N")
    Transpose  = kwargs.setdefault("Transpose", False)
    summary    = not any((orgtbl, resultdf)) #~ Summary by default
 
    #~ Construct the Summary table, using either table or df_to_orgtbl
    if table_info:
        if type(table_info) not in (list,tuple): table_info=[table_info]
        info_dict = {"N": lambda model: model.nobs,
                     "R2": lambda model: model.rsquared,
                     "R2-adj": lambda model: model.rsquared_adj,
                     "F-stat": lambda model: model.fvalue}
        info_dict = dict([(x,info_dict[x]) for x in table_info])

    if summary:
        from statsmodels.iolib import summary2
        Summary = summary2.summary_col(list(models.values()), stars=True, float_format='%.3f',info_dict=info_dict)
        #~ This mangles much of the pretty left to the Summary2 object and returns a pd.DF w/o se's
        if Transpose: Summary = Summary.tables[0].T.drop("",1)

    else:
        # Extras = lambda model: pd.Series({"N":model.nobs})
        # results = pd.DataFrame({Var:model.params.append(Extras(model)) for Var,model in models.iteritems()})
        try:
            xrange
            Ms = lambda: models.iteritems()
        except NameError: Ms = lambda: models.items()
        results = pd.DataFrame({Var:model.params for Var,model in Ms()})
        SEs     = pd.DataFrame({Var:model.bse    for Var,model in Ms()})
        if table_info:
            try:
                info_dict.iteritems()
                info_items = lambda: info_dict.iteritems()
            except AttributeError: info_items = lambda: info_dict.items()
            extras = pd.DataFrame({Var: pd.Series({name:stat(model) for name,stat in info_items()}) for Var,model in Ms()})
            results = results.append(extras)
        if Transpose: results,SEs = results.T, SEs.T

        if orgtbl: Summary = df_to_orgtbl(results,sedf=SEs)
        else:
            assert(resultdf)
            Summary = results, SEs

    return Summary
def winsorize(Series, **kwargs):
    """
    Need to implement two-sided censoring as well.
    WARNING: if Top<0, all zeros will be changed to Top
    """

    percent    = kwargs.setdefault("percent",98)
    stdev      = kwargs.setdefault("stdev",False)
    drop       = kwargs.setdefault("drop",False)
    drop_zeros = kwargs.setdefault("drop_zeros",True)
    twoway     = kwargs.setdefault("twoway",False)
    
    Ser = Series.copy()

    if drop_zeros: S = Ser.replace(0,np.nan).dropna()
    else: S = Ser.dropna()
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
    Ser[Ser>Top]=replace_with

    if not twoway: return Series
    else:
        kwargs['twoway'] = False
        return -1*winsorize(-1*Ser, **kwargs)

InSample = {1: 0, 2: np.nan, 3: 1}

if True: #~ Make DataFrame
    D = full_data(DIR=DATADIR)
    D["In14"] = D["merge_midline"].apply(lambda x: InSample.get(x))
    D["In15"] = D["merge_endline"].apply(lambda x: InSample.get(x))

    A = asset_vars(D,year=2013)[0].apply(winsorize)
    D['Asset Tot'] = A['Total']
    D['Asset Prod'] = A['Productive']
    D["Cash Savings"] = D.filter(regex="^savings_.*_b$").sum(axis=1)
    D["Land Access (fedan)"] = D.filter(regex="^land_.*_b$").sum(axis=1)
    #~C = consumption_data(D)[0].ix[2014]
    C = consumption_data(D)[0].reorder_levels([1,2,0]).sort_index()
    food = ['cereals', 'maize', 'sorghum', 'millet', 'potato', 'sweetpotato', 'rice', 'bread', 'beans', 'oil', 'salt', 'sugar', 'meat', 'livestock', 'poultry', 'fish', 'egg', 'nuts', 'milk', 'vegetables', 'fruit', 'tea', 'spices', 'alcohol', 'otherfood']
    month = ['fuel', 'medicine', 'airtime', 'cosmetics', 'soap', 'transport', 'entertainment', 'childcare', 'tobacco', 'batteries', 'church', 'othermonth']    
    year = ['clothesfootwear', 'womensclothes', 'childrensclothes', 'shoes', 'homeimprovement', 'utensils', 'furniture', 'textiles', 'ceremonies', 'funerals', 'charities', 'dowry', 'other']    
    C["Food"]  = C[[item for item in food  if item in C]].sum(axis=1).replace(0,np.nan)
    C["Month"] = C[[item for item in month if item in C]].sum(axis=1).replace(0,np.nan)
    C["Year"]  = C[[item for item in year  if item in C]].sum(axis=1).replace(0,np.nan)
    C["Tot"]   = C[["Food","Month","Year"]].sum(axis=1)
    D["Daily Exp"] = C["Tot"].loc[2013].groupby(level="HH").first()
    D["Daily Food"] = C["Food"].loc[2013].groupby(level="HH").first()

    drop_vars = ['c_milk', 'c_alcohol', 'c_spices', 'c_entertainment', 'c_otherfood', 'asset_val_house', 'asset_val_plough']
    D.drop([item for item in D if any(var in item for var in drop_vars)], 1, inplace=True)

consumption = dict([(c,c[2:-2].capitalize()) for c in D.filter(regex='^c_.*_b$').columns])
assets = dict([(a,a[10:-2].capitalize()) for a in D.filter(regex='^asset_val.*_b$').columns])
other_outcomes = {"hh_size_b":"HH size",
        "child_total_b": "# Child",
        'asset_n_house_b':'No. Houses',
        'in_business_b':'In Business',
        'c_cereals_b':'Cereals',
        'c_cosmetics_b':'Cosmetics',
        'Land Access (fedan)':'Land Access (fedan)',
        'Cash Savings': 'Cash Savings',
        'Asset Tot': 'Asset Tot.',
        'Asset Prod': 'Asset Prod.',
        'Daily Exp':'Daily Exp',
        'Daily Food':'Daily Food'}

D["Tgroup"] = D['group'].apply(lambda x: "TUP" if "sset" in x else x)
Table = D.groupby("Tgroup").sum()[['Base','Mid','End']].applymap(int)    
Table.loc["All"] = Table.sum().apply(lambda x: round(x,3))


df = D[D["Base"]]
TabBal = df.groupby("Tgroup").sum()[['Base','Mid','End']].applymap(int)    
TabBal.loc["All"] = TabBal.sum().apply(lambda x: round(x,3))
Table.append(TabBal)

Table.loc[". "] = pd.Series(dict(zip(['Base','Mid','End'], ["Mean_{Bsln}","\\beta_{Mid}","\\beta_{End}"])))

regs14 = {}
regs15 = {}
BASE   = pd.Series()
for vardict in (consumption, assets, other_outcomes):
    #~for group in ("In14","In15"): vardict.setdefault(group,group)
    df = D[list(vardict.keys())+["In14","In15"]].rename(columns=vardict)
    BASE = BASE.append(df.mean())
    df["const"] = 1.
    regs14.update(regressions(df,outcomes=vardict.values(),controls=["const","In14"]))
    regs15.update(regressions(df,outcomes=vardict.values(),controls=["const","In15"]))

TAB = reg_table(regs14,Transpose=True,Restultdf=True)
BASE.name="Base"
TAB = TAB.join(BASE)
TAB15 = reg_table(regs15,Transpose=True,Restultdf=True)
TAB["In15"] = TAB15["In15"]
TAB = TAB.rename(columns={"In14":"Mid","In15":"End"})[["Base","Mid","End"]]
Table = Table.append(TAB).rename(columns=dict(zip(['Base','Mid','End'], ["2013","2014","2015"])))
TAB = df_to_orgtbl(Table)
#print(TAB) 
        

#~c_table     = ttest_table(D,consumption,groups = ["In14", "In15"])
#~a_table     = ttest_table(D,assets,groups = ["In14", "In15"])
#~other_table = ttest_table(D,other_outcomes,groups = ["In14", "In15"])
#~
#~c_orgtable     = df_to_orgtbl(c_table,float_fmt='%5.2f')
#~a_orgtable     = df_to_orgtbl(a_table,float_fmt='%5.2f')
#~other_orgtable = df_to_orgtbl(other_table,float_fmt='%5.2f')
#~    
#~tables="\n".join((c_orgtable,a_orgtable,other_orgtable))
