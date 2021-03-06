import numpy as np
import pandas as pd
from pandas.io import stata
import statsmodels.api as sm
from matplotlib import pyplot as plt

def full_data(File="../data/TUP_full.dta", balance = [],normalize=True):
    """
    Reads in TUP_full.dta, the full dataset after the cleaning in stata (which is where most of the variable selection happen
    If you need a variable not in TUP_full, include it in the keep command in `year'_cleanup.do and re-run TUP_merge.do)
    NOTE: This function is taking the merged data in wide format
        with base/mid/endline data having suffixes _b, _m, _e.
    normalize:
        Normalizes consumption to SSP/day, given recall window in each.
        Takes the form {days in window: list of goods with that recall window}
    balance: 
        Enforces balance of households across the panel consisting of the years speficied in `balance'
        (any of ['Base','Mid','End'])

    Returns D
    """
    Df = stata.read_stata(File)
    Df.rename(columns={'idno':'HH', "Control":"CTL", "Cash":"CSH"},inplace=True)
    Df.set_index("HH",inplace=True,drop=False)
    for t in ['CTL','CSH','TUP']: Df[t].fillna(0,inplace=True)
    #~ Organize merge and attrition variables
    mergedict = {'master only (1)':  1, 'using only (2)':  2, 'matched (3)':  3}
    for col in Df.filter(like='merge_').columns:
        Df[col]=Df[col].apply(lambda i: mergedict.get(i))
    
    Df['Base'] =  Df['merge_census_b']>1
    Df['Mid']  =  Df['merge_midline']>1
    Df['End']  =  Df['merge_endline']>1

    if normalize:
        try: len(normalize)
        except TypeError:
            food = ['c_cereals', 'c_maize', 'c_sorghum', 'c_millet', 'c_potato', 'c_sweetpotato', 'c_rice', 'c_bread', 'c_beans', 'c_oil', 'c_salt', 'c_sugar', 'c_meat', 'c_livestock', 'c_poultry', 'c_fish', 'c_egg', 'c_nuts', 'c_milk', 'c_vegetables', 'c_fruit', 'c_tea', 'c_spices', 'c_alcohol', 'c_otherfood']
            month = ['c_fuel', 'c_medicine', 'c_airtime', 'c_cosmetics', 'c_soap', 'c_transport', 'c_entertainment', 'c_childcare', 'c_tobacco', 'c_batteries', 'c_church', 'c_othermonth']    
            year = ['c_clothesfootwear', 'c_womensclothes', 'c_childrensclothes', 'c_shoes', 'c_homeimprovement', 'c_utensils', 'c_furniture', 'c_textiles', 'c_ceremonies', 'c_funerals', 'c_charities', 'c_dowry', 'c_other']    
            normalize = {3:food, 30:month, 360:year}
    for col in Df.columns:
        for window, category in normalize.iteritems():
            try:
                if col[:-2] in category:   Df[col] /= window
            except KeyError: print "{} not in Df".format(col)    
    
    #~ Remove these for Endline!!! You have disaggregate versions of these for the mid-to-end comparison
    Df.drop(["c_cereals_e","c_meat_e"],axis=1, inplace=True) #~ , "c_cereals_m","c_meat_m"
    D  = Df[Df[balance].all(axis=1)] 
    del Df
    return D

def consumption_data(D, how="long", hh_vars=["hh_size","child_total"], goods_from_years=[]):
    """
        Takes the DataFrame D from full_data()

        Reshapes HH & C into long format if how=="long". Else, leaves as wide with _b,_m,_e suffixes

        Returns:

        C- Consumption df using a set of goods specified

        HH- HH df containing a set of characteristics specified

        T- Treatment variables

        hh_vars: control variables to be pulled from full dataset and included in HH
        normalize: Divide variables by number of days in their recall windows (3, 30, or 360)
        balance: Base, Mid, and End-- Drops to balance on all years in list.
            If estimation is restricted to 1 or 2 years, don't drop those just missing in unused years.
        goods_from_years: Any year in ["Base", "Mid", "End"]; returns C with the intersection of consumption categories from all years in list.
    """
    #~ Read in and clean up full data

    C  = D.filter(regex='^c_')
    HH = D.filter([i for i in D.columns if any(j in i for j in hh_vars)]) #~ Convoluted, but includes all specified hh_vars w/ any suffix

    #~ Balance expenditure categories across years in "goods_from_years" (Options)
    suffix = {'Base':'_b','Mid':'_m','End':'_e'}
    
    if goods_from_years: #~ Chosen to balance included expenditure categories across years
        #~ If specified "Base" or "End" switch to suffixes
        if goods_from_years[0] in suffix: goods_from_years=[suffix[year] for year in goods_from_years] 
        keep_goods = [good[:-2] for good in C if good.endswith(goods_from_years[0])]
        for survey in goods_from_years[1:]:
            list2 = [good[:-2] for good in C if good.endswith(survey)]
            keep_goods = [item for item in keep_goods if item in list2]
            
        #~ This is how one gets all columns matching any string in a list
        #~ Dealing with this hideous subscript notation that I'll try to phase out at some point.
        C = C.filter(regex="|".join(keep_goods))

    C.to_pickle('/tmp/ss-consumption.df')

    if how=="long":
    ####~ Reshape Consumption Data ~####
        #~ Cs breaks C down by year (by checking suffixes via regex), removes the suffix
        Cs = [C.filter(regex='_{}$'.format(year)).rename(columns=lambda i: i[:-2]) for year in list('bme')]
        for i in xrange(len(Cs)):
            #~ Then specify year
            Cs[i]['Year']=2013+i
            #~ Re-insert HH id
            Cs[i]['HH']=D['HH']
        #~ And concat into long form
        C = pd.concat(Cs)

        #~ Reshape Household Data (Same dance as above)
        HHs = [HH.filter(regex='_{}$'.format(year)).rename(columns=lambda i: i[:-2]) for year in list('bme')]
        for year in xrange(len(HHs)):
            HHs[i]['Year']=2013+i
            HHs[i]['HH']=D['HH']
        HH = pd.concat(HHs)
        del Cs
        del HHs

        C.set_index(["Year","HH"],  inplace=True, drop=True)
        HH.set_index(["Year","HH"], inplace=True, drop=True)

    T = D[['HH','CTL','CSH','TUP']].set_index("HH", drop=True)
    
    return C, HH, T

def regressions(DF,Year="", **kwargs):
    """ Run a set of regressions and return a dict of {Outcome: sm.OLS (or RLM) model} for each model
     DF:
         The full dataset with outcomes and control variables.
     Year:
         A suffix on each outcome variable, specifying which round of data is being used. (Default to "")
     Baseline:
         A suffix on each variable to be used as a baseline covariate, specifying which round of data is being used.
         If the outcome variable doesn't have a corresponding column with that suffix, passes without error.
         (Default to 2013)
     Controls:
         A list or tuple of variables to be used as covariates in each regression.
     Outcomes:
         The list of outcomes (also the names of the models)
     rhs_extra:
         A dictionary of covariates to be added to the regression for specific outcomes.
     Baseline_na:
         If True, code missing values of baseline variable as zero and include a "Bsln_NAN" indicator in outcomes.
     Robust:
         If True, use statsmodel's RLM class instead of OLS (defaults to Huber-T se's)
     Return:
         dict {outcome var:model} for each outcome in outcomes.
    """
    #~ Kwargs
    Baseline    = kwargs.setdefault("Baseline",  2013)
    controls    = kwargs.setdefault("controls",  ["cons",'Cash','TUP'])
    rhs_extra   = kwargs.setdefault("rhs_extra", {})
    outcomes    = kwargs.setdefault("outcomes",  [])
    baseline_na = kwargs.setdefault("baseline_na", True)
    robust      = kwargs.setdefault("robust",    False)
    

    if robust: regress=sm.RLM
    else: regress=sm.OLS
    if not type(Year)==str: Year=str(Year)
    if not type(Baseline)==str: Baseline=str(Baseline)
    models_ols = {}

    for outcome in outcomes: #~ Run regressions and store models in a dictionary
        Yt = [outcome+Year]
        if outcome+Baseline in DF: #~ Present in DataFrame
            if DF[outcome+Baseline].isnull().sum(): Yt.append(outcome+Baseline)
        if outcome in rhs_extra:
            if not type(rhs_extra[outcome]) in (list,tuple): rhs_extra[outcome] = [rhs_extra[outcome]]
            for x in rhs_extra[outcome]:
                try: assert(x in DF)
                except AssertionError: raise KeyError("Extra Covariate for outcome {} not found in data".format(x,outcome))
            Yt += list(rhs_extra[outcome])
        df = DF[Yt+controls].rename(columns={outcome+Baseline:"Bsln"+Baseline})
        if "Bsln"+Baseline in df and baseline_na:
            df["Bsln_NAN"] = df["Bsln"+Baseline].isnull().apply(int)
            df["Bsln"+Baseline].fillna(0,inplace=True)
        df = df.dropna()
        #~ Full-sample OLS
        models_ols[outcome] = regress(df[outcome+Year], df.drop(outcome+Year,1)).fit()
        del df
    return models_ols
    #~ TODO: SPLIT models and results into two functions.

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
        Summary = summary2.summary_col(models.values(), stars=True, float_format='%.3f',info_dict=info_dict)
        #~ This mangles much of the pretty left to the Summary2 object and returns a pd.DF w/o se's
        if Transpose: Summary = Summary.tables[0].T.drop("",1)

    else:
        # Extras = lambda model: pd.Series({"N":model.nobs})
        # results = pd.DataFrame({Var:model.params.append(Extras(model)) for Var,model in models.iteritems()})
        results = pd.DataFrame({Var:model.params for Var,model in models.iteritems()})
        SEs     = pd.DataFrame({Var:model.bse    for Var,model in models.iteritems()})
        if table_info:
            extras = pd.DataFrame({Var: pd.Series({name:stat(model) for name,stat in info_dict.iteritems()}) for Var,model in models.iteritems()})
            results = results.append(extras)
        if Transpose: results, SEs = results.T, SEs.T

        if orgtbl: Summary = df_to_orgtbl(results,sedf=SEs)
        else:
            assert(resultdf)
            Summary = results, SEs

    return Summary

def df_to_orgtbl(df,tdf=None,sedf=None,float_fmt='%5.2f'):
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



#~ Separate consumption categories by recall window and normalize each to SSP/day measures
food = ['c_cereals', 'c_maize', 'c_sorghum', 'c_millet', 'c_potato', 'c_sweetpotato', 'c_rice', 'c_bread', 'c_beans', 'c_oil',
        'c_salt', 'c_sugar', 'c_meat', 'c_livestock', 'c_poultry', 'c_fish', 'c_egg', 'c_nuts', 'c_milk', 'c_vegetables',
        'c_fruit', 'c_tea', 'c_spices', 'c_alcohol', 'c_otherfood']
month = ['c_fuel', 'c_medicine', 'c_airtime', 'c_cosmetics', 'c_soap', 'c_transport', 'c_entertainment', 'c_childcare', 'c_tobacco', 'c_batteries',
         'c_church', 'c_othermonth']    
year = ['c_clothesfootwear', 'c_womensclothes', 'c_childrensclothes', 'c_shoes', 'c_homeimprovement', 'c_utensils', 'c_furniture', 'c_textiles', 'c_ceremonies', 'c_funerals',
        'c_charities', 'c_dowry', 'c_other']    

normalize = {3:food, 30:month, 360:year}

D = full_data(normalize=normalize)

C, HH, T = consumption_data(D, how="long")
C = C.join(T, how="left")
Outcomes = ["Tot", "FoodShr", "Food",  "logTot"]

#~ Make aggregate variables
for Year,suffix in ( ("2013","_b"), ("2014","_m"), ("2015","_e") ):
    C["Food"]   = C[[item for item in food  if item in C]].sum(axis=1).replace(0,np.nan)
    C["Month"]  = C[[item for item in month if item in C]].sum(axis=1).replace(0,np.nan)
    C["Year"]   = C[[item for item in year  if item in C]].sum(axis=1).replace(0,np.nan)
    C["Tot"]    = C[["Food","Month","Year"]].sum(axis=1).replace(0,np.nan)
    C["FoodShr"]= C["Food"]/C["Tot"] #~ FoodShare variable
    C["logTot"] = C["Tot"].apply(np.log)

#~ Make Baseline variable
for var in Outcomes: 
    Bl = C.loc[2013,var]
    C = C.join(Bl,rsuffix="2013", how="left")


C["Y"]=np.nan
for yr in (2013, 2014, 2015): C.loc[yr,"Y"]=str(int(yr))

C = C.join(pd.get_dummies(C["Y"]), how="left")
for group in ("TUP", "CSH"):
    for year in ("2013", "2014", "2015"):
        interaction = C[group]*C[year]
        if interaction.sum()>0: C["{}*{}".format(group,year)] = interaction

Controls = ['2014', '2015', 'TUP*2014', 'TUP*2015', 'CSH*2014', 'CSH*2015']
C = C.loc[2014:2015]
#~ This is the main specification. Given the mismatch in timing, we compare CSH*2015 to both TUP*2014 and TUP*2015
regs = regressions(C, outcomes=Outcomes, controls=Controls, Baseline=2013, baseline_na=True)
#~ regs = {var: sm.OLS(C[var], C[Controls], missing='drop').fit() for var in Outcomes}

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
