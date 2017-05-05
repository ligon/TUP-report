#!/usr/bin/env python
'''
File: TUP.py
Author: Elliott Collins
Description: Necessary functions for reading in, cleaning, and analyzing TUP data
Updated: 2015-12-02
'''

if True: #~ Imports
    import numpy as np
    import pandas as pd
    from pandas.io import stata
    import statsmodels.api as sm
    from matplotlib import pyplot as plt

def Range(*args):
    """ Silly Python3 compatability stuff"""
    try: return(xrange(*args))
    except NameError: return(range(*args))    

def Items(dct):
    """ Silly Python3 compatability stuff"""
    try: return(dct.iteritems())
    except AttributeError: return(dct.items())    
   
def full_data(File="TUP_full.dta",DIR="../../data/", balance = [],normalize=True, locations=True):
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
    NEW: Pulls in and includes location variables (but does not set them as an index)

    Returns D
    """
    File = DIR+File
    if "csv" in File:   Df = pd.read_csv(File)
    elif "dta" in File: Df = stata.read_stata(File)
    else:               Df = pd.read_pickle(File)
    Df.rename(columns={'idno':'HH', "Control":"CTL", "Cash":"CSH"},inplace=True)
    Df.set_index("HH",inplace=True,drop=False)
    for t in ['CTL','CSH','TUP']: Df[t].fillna(0,inplace=True)
    #~ 114 households are not listed as being in any group...
    #~ Several of these are also in the baseline survey, so would have been in the randomization file...
    #~ Those *not* in the baseline are potentially a non-random sample, what with differential attrition.
    #~ CTL2 includes all non-treatment households as controls. Importantly, so does any regression including a constant instead of a saturated model approach
    Df["CTL2"] = 1-Df[["TUP","CSH"]].sum(1)

    #~ Organize merge and attrition variables
    mergedict = {'master only (1)':  1, 'using only (2)':  2, 'matched (3)':  3}
    for col in Df.filter(like='merge_').columns:
        Df[col]=Df[col].apply(lambda i: mergedict.get(i)).astype(float)
    
    Df['Base'] =  Df['merge_census_b']>1
    Df['Mid']  =  Df['merge_midline']>1
    Df['End']  =  Df['merge_endline']>1

    if normalize:           #~ Set consumption variables to their daily values
        try: len(normalize) #~ Check if a dictionary was passed
        except TypeError:   #~ Otherwise make the obvious one
            food = ['c_cereals', 'c_maize', 'c_sorghum', 'c_millet', 'c_potato', 'c_sweetpotato', 'c_rice', 'c_bread', 'c_beans', 'c_oil', 'c_salt', 'c_sugar', 'c_meat', 'c_livestock', 'c_poultry', 'c_fish', 'c_egg', 'c_nuts', 'c_milk', 'c_vegetables', 'c_fruit', 'c_tea', 'c_spices', 'c_alcohol', 'c_otherfood']
            month = ['c_fuel', 'c_medicine', 'c_airtime', 'c_cosmetics', 'c_soap', 'c_transport', 'c_entertainment', 'c_childcare', 'c_tobacco', 'c_batteries', 'c_church', 'c_othermonth']    
            year = ['c_clothesfootwear', 'c_womensclothes', 'c_childrensclothes', 'c_shoes', 'c_homeimprovement', 'c_utensils', 'c_furniture', 'c_textiles', 'c_ceremonies', 'c_funerals', 'c_charities', 'c_dowry', 'c_other']    
            normalize = {3:food, 30:month, 360:year}
        for col in Df.columns: #~ And actually normalize consumption variables to daily values
            for window, category in Items(normalize):
                try:
                    if col[:-2] in category:   Df[col] /= window
                except KeyError: print("{} not in Df".format(col))
    
    #~ Df.drop(["c_cereals_e","c_meat_e"],axis=1, inplace=True) #~ , "c_cereals_m","c_meat_m"
    D  = Df[Df[balance].all(axis=1)] 
    if locations:
        L = pd.read_csv(DIR+"Locations.csv").rename(columns={"RespID":"HH"}).set_index("HH")["Location"]
        D = D.drop("location_b",1).join(L,how="left")
    else: D["Location"]=1.
    del Df
    return D

def consumption_data(D, include2016=False, how="long", hh_vars=["hh_size","child_total"], goods_from_years=[],WRITE=False,use_dates=False):
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
        include2016: calls mobile_data() and concatenates that information into C and HH
            IF include2016, then use_dates decides if each wave is indexed separately or if they're duplicately indexed as "2016"

   """
    #~ Read in and clean up full data

    def newdf(vars,D,index=["HH","Location"]):
        X=D.drop(D.index.names,1).reset_index().copy()
        if type(vars)==str: df=X.filter(regex=vars).join(X[index]).set_index(index)
        else:               df=X[vars+index].set_index(index)
        return df
    
    C  = newdf("^c_",D)
    HH = newdf([i for i in D.columns if any(i.startswith(j) for j in hh_vars)],D) #~ Includes all specified hh_vars w/ any suffix

    #~ Balance expenditure categories across years in "goods_from_years" (Options)
    suffix = {"Base":'_b',"Mid":'_m',"End":'_e'}
    
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

    if how=="long":
    ####~ Reshape Consumption Data ~####
        #~ Cs breaks C down by year (by checking suffixes via regex), removes the suffix *AND* prefix
        Cs = [C.filter(regex='_{}$'.format(year)).rename(columns=lambda i: i[2:-2]) for year in list('bme')]
        HHs = [HH.filter(regex='_{}$'.format(year)).rename(columns=lambda i: i[:-2]) for year in list('bme')]
        for i in Range(len(Cs)):
            Cs[i]['Year']=2013+i
            HHs[i]['Year']=2013+i
        C = pd.concat(Cs).set_index("Year", append=True).reorder_levels([0,2,1])
        HH = pd.concat(HHs).set_index("Year", append=True).reorder_levels([0,2,1])
        del Cs
        del HHs
        if include2016:
            M,Mc,Mhh = mobile_data(use_dates=use_dates)
            Mc = Mc.groupby(level=['HH'])
            HH=pd.concat([HH, Mhh])
            C=pd.concat([C, Mc])

    T = D[['HH','CTL','CSH','TUP']].set_index("HH", drop=True)
    
    if WRITE=="csv":
        C.to_csv('/tmp/ss-consumption{}.csv'.format(str("_wide"*bool(how!="long"))))
        HH.to_csv('/tmp/ss-hh{}.csv'.format(str("_wide"*bool(how!="long"))))
        T.to_csv('/tmp/ss-treatment{}.csv'.format(str("_wide"*bool(how!="long"))))
   elif WRITE:
        C.to_pickle('/tmp/ss-consumption{}.df'.format(str("_wide"*bool(how!="long"))))
        HH.to_pickle('/tmp/ss-hh{}.df'.format(str("_wide"*bool(how!="long"))))
        T.to_pickle('/tmp/ss-treatment{}.df'.format(str("_wide"*bool(how!="long"))))

    return C, HH, T

def mobile_data(File="remote_survey_Nov2015_April2016.dta", use_dates=True, DIR="../../data/Mobile/", set_thresholds = False):

    File = DIR+File
    LocationsFile = DIR+"Locations.csv"
    use_bsln =  "b" #~ "m" "e"
    M = stata.read_stata(File)
    M["Sell"]  = M.filter(regex="^S6_[abcd]_2").sum(1)
    M["Buy"]   = M.filter(regex="^S7S7_[abcd]_2").sum(1)
    M["iSell"] = (M["Sell"]>0).apply(int)
    M["iBuy"]  = (M["Buy"]>0).apply(int)
    VARNAMES = {"Sell":"Sell","Buy":"Buy","iSell":"iSell","iBuy":"iBuy","introDate_Int":"date", "introId_Number":"HH", "introRes_name":"Name", "introEnu_name":"Enumerator",
            "S3S3_a":"hh_size", "S3S3_b":"child_total", "S3S3_c":"num_meals", "S4S4_a":"vegetables",
            "S4S4_b":"sugar", "S4S4_c":"fish", "S4S4_d":"nuts", "S4S4_e":"beans",
            "S5S5_a":"fuel", "S5S5_b":"medicine", "S5S5_c":"airtime", "S5S5_d":"cosmetics",
            "S5S5_e":"soap", "SubmissionDate_year":"year", "SubmissionDate_month":"month", "SubmissionDate_day":"day",
            "SubmissionDate_hour":"hour", "SubmissionDate_minute":"minute", "SubmissionDate_second":"second", "month":"monthname"}

    Datename = {"November 2015":"1November", "December 2015":"2December",
    "January 2016":"3January", "February 2016":"4February", "March 2016":"5March", "April 2016":"6April"}
    HH_vars = ["hh_size", "child_total"]
    ITEMS = ["vegetables", "sugar", "fish", "nuts", "beans", "fuel", "medicine", "airtime", "cosmetics", "soap"]
    Food, Durables = ITEMS[:5], ITEMS[5:]

    M = M.rename(columns=VARNAMES)[list(VARNAMES.values())]
    M['t'] = M['monthname'].replace(Datename)
    M = M.set_index(["t","HH"],drop=False)
    #~ Eliminate duplicates
    M = M.groupby(level=["t","HH"]).last()

    if not set_thresholds: #~ thresholds set manually by looking for outliers graphically.
        thresholds={'soap': 750, 'airtime': 600,
                    'fuel': 510, 'fish': 150,
                    'nuts': 150, 'medicine': 5000,
                    'sugar': 500, 'cosmetics': 1100,
                    'beans': 200, 'vegetables': 400,
                    'hh_size': 25, 'child_total': 17}
    else: #~ Make graphs and set thresholds manually
        M["Total"] = M[ITEMS].sum(1)
        thresholds={}
        for item in ITEMS:
            find_outliers(item,M,"Total")
            plt.show()
            thresholds[item]=int(Input(item+": "))
        for item in HH_vars:
            find_outliers(item,M)
            plt.show()
            thresholds[item]=int(Input(item+": "))
    for item,thresh in Items(thresholds): #~ Topcode
        M.loc[M[item]>thresh,item]=thresh
    
    L = pd.read_csv("../../data/Locations.csv").rename(columns={"RespID":"HH"}).set_index("HH")["Location"]
    M = M.join(L,how="left")
    if use_dates: M = M.rename(columns={"t":"Year"}) #~ ["Year"]=2016
    else:         M["Year"]=2016
    M = M.set_index(['HH', 'Year', 'Location'])
    M[Food] /= 3.
    M[Durables] /= 30.
    Mc, Mhh = M[ITEMS], M[HH_vars]
    return M, Mc, Mhh 


def read_data(File="../../data/csv/TUP_full.csv",hh_vars=["hh_size","child_total"], normalize=True, balance = [], goods_from_years=[]):
    """
        Defunct- Split into full_data (returns a clean version of the full dataset) and consumption_data (returns C, HH, and T)

        Reads in raw data, returning:
        C- Consumption df using a set of goods specified
        HH- HH df containing a set of characteristics specified
        D- Full dataset; T- Treatment variables (useful for groupby)
        NOTE: This function is taking the merged data in wide format
            with base/mid/endline data having suffixes _b, _m, _e.
            Not the most elegant, but it fits with Banerjee et al. (2015)
        hh_vars: control variables to be pulled from full dataset and included in HH
        normalize: Divide variables by number of days in their recall windows (3, 30, or 360)
        balance: Base, Mid, and End-- Drops to balance on all years in list.
            If estimation is restricted to 1 or 2 years, don't drop those just missing in unused years.
        goods_from: same values as balance; returns C with the intersection of consumption categories from all years in list.
    """
    #~ Read in and clean up full data
    Df = pd.read_csv(File)
    Df.rename(columns={'idno':'HH','Control':'CTL','Cash':'CSH'},inplace=True)
    Df.set_index("HH",inplace=True,drop=False)
    for t in ['CTL','CSH','TUP']: Df[t].fillna(0,inplace=True)
    #~ Organize merge and attrition variables...
    mergedict = {'master only (1)':  1,
        'using only (2)':  2,
        'matched (3)':  3}
    suffix = {'Base':'_b','Mid':'_m','End':'_e'}
    for col in Df.filter(like='merge_').columns: Df[col]=Df[col].apply(lambda i: mergedict.get(i))
    Df['Base'] = (Df['merge_census_b']>1) 
    Df['Mid']  =  (Df['merge_midline']>1) 
    Df['End']  =  (Df['merge_endline']>1) 

    if normalize:
        food = ['c_cereals', 'c_maize', 'c_sorghum', 'c_millet', 'c_potato', 'c_sweetpotato', 'c_rice', 'c_bread', 'c_beans', 'c_oil', 'c_salt', 'c_sugar', 'c_meat', 'c_livestock', 'c_poultry', 'c_fish', 'c_egg', 'c_nuts', 'c_milk', 'c_vegetables', 'c_fruit', 'c_tea', 'c_spices', 'c_alcohol', 'c_otherfood']
        month = ['c_fuel', 'c_medicine', 'c_airtime', 'c_cosmetics', 'c_soap', 'c_transport', 'c_entertainment', 'c_childcare', 'c_tobacco', 'c_batteries', 'c_church', 'c_othermonth']    
        year = ['c_clothesfootwear', 'c_womensclothes', 'c_childrensclothes', 'c_shoes', 'c_homeimprovement', 'c_utensils', 'c_furniture', 'c_textiles', 'c_ceremonies', 'c_funerals', 'c_charities', 'c_dowry', 'c_other']    
        normalize = {3:food, 30:month, 360:year}
        for col in Df.columns:
            try: #~ Normalizing to daily average consumption across monitored window.
                if col[:-2] in food:  Df[col]/=3.
                if col[:-2] in month: Df[col]/=30.
                if col[:-2] in year:  Df[col]/=360.
            except KeyError: print("{} not in Df".format(col))

    #~ Balance panel across years in "balance"
    D  = Df[Df[balance].all(axis=1)]

    C  = D.filter(regex='^c_')
    HH = D.filter([i for i in D.columns if any(j in i for j in hh_vars)])

    #~ Balance expenditure categories across years in "goods_from"
    if not goods_from_years: goods_from_years = ["Base", "Mid"]
    if goods_from_years[0] in suffix: goods_from_years=[suffix[year] for year in goods_from_years] #~ If specified "Base" switch to suffixes
    keep_goods = [good for good in C if good.endswith(goods_from_years[0])]
    for i in range(1,len(goods_from_years)):
        keep_goods += [good for good in C if good.endswith(goods_from_years[i]) and good[:-2]+goods_from_years[i-1] in keep_goods]
    C = C[keep_goods]
    C.to_pickle('/tmp/ss-consumption.df')

    #~ Reshape Consumption Data
    Cs = [C.filter(regex='_{}$'.format(year)).rename(columns=lambda i: i[:-2]) for year in list('bme')]
    for year in Range(len(Cs)):
        Cs[year]['Year']=2013+year
        Cs[year]['HH']=D['HH']
    C = pd.concat(Cs)

    #~ Reshape Household Data
    HHs = [HH.filter(regex='_{}$'.format(year)).rename(columns=lambda i: i[:-2]) for year in list('bme')]
    for year in Range(len(HHs)):
        HHs[year]['Year']=2013+year
        HHs[year]['HH']=D['HH']
    HH = pd.concat(HHs)
    del Cs
    del HHs

    C.set_index(["Year","HH"],  inplace=True, drop=True)
    HH.set_index(["Year","HH"], inplace=True, drop=True)

    T = D[['HH','CTL','CSH','TUP']]
    T.set_index("HH", inplace=True,drop=True)
    return D, C, HH, T

def process_data(*args, **kwargs):
    """
    If C, HH, and T are passed as arguments (in that order), uses those
        Otherwise, calls consumption_data(how="wide")
    year: pass a pair of suffixes to be used. Defaults to baseline _b and endline _e.
        This is really written for two-year estimation, so "balance" and "goods_from" in read_data gets set to "Year"
    max_zeros: Need a full-rank covariance matrix for standard errors, 
        and shouldn't be using goods with very few non-zero responses in general.
        Eliminates any column with fewer than ... non-zero responses. Defaults to 30,
        which is enough to avoid singular matricies in this data.
    kwargs: all other kwargs get passed to read_data()
    save: Saves pickled version of ss-goods to /data/modified for more permanent storage. Defaults to False
    """
    #~ Manage arguments
    Y = {'_b':2013,'_m':2014,'_e':2015}
    Survey = {'_b':'Base','_m':'Mid','_e':'End'}
    year = kwargs.setdefault('year', ('_b', '_m'))
    kwargs['balance']    = kwargs.setdefault('balance'   ,[Survey[yr] for yr in year])
    kwargs['goods_from'] = kwargs.setdefault('goods_from',[Survey[yr] for yr in year])
    max_zeros = kwargs.setdefault('max_zeros', 30)
    save = kwargs.setdefault('save', False)
    try: C,HH,T = args
    except ValueError:
        D = full_data()
        C, HH, T = consumption_data(D, how="wide")

    #~ Get location variables for each (Just neighborhood dummies)
    L = pd.read_csv('../../data/csv/Locations.csv')
    L = L.rename(columns={'RespID':'HH'}).set_index('HH')['Location'].apply(lambda x: x.lower())

    #~ Delete items with too many zeros
    many_zeros = lambda Suffix: [col for col in C if col.endswith(Suffix) and sum(C[col]>0)<max_zeros]
    toDrop = list(set(many_zeros(year[0])+many_zeros(year[1]) + (year[0]=='_m')*['c_cereals_m','c_meat_m', 'c_cereals_e','c_meat_e']))
    print("Too Many Zeros: {}".format(repr(toDrop)))
    for item in toDrop:
        try: del C[item]
        except KeyError: pass

    #~ Set 0 to missing & use log consumption
    goods = np.log(C.replace(to_replace=0.,value=np.NaN))
    #~ Join on baseline HH characteristics (baseline being first year specified in "year" tuple
    goods = goods.join(HH, how='left')
    #~ Demean by good?
    goods -= goods.mean(axis=0)

    #~ Make Treatment categorical
    grps = ['TUP','CSH','CTL']
    T["Group"] = ""
    for g in grps: T["Group"] += map(lambda i: g*i,T[g])

    #~ Join on Treatment dummy
    goods= goods.merge(T, how='left',right_index=True, left_index=True)

    #~ Demean HH characteristics by treatment group
    hh_vars = ['hh_size_b', 'hh_size_m', 'hh_size_e', 'child_total_b', 'child_total_m', 'child_total_e']
    goods[hh_vars] = goods.groupby('Group')[hh_vars].apply(lambda group: group-group.mean())

    #~ Import area Dummies
    goods['Loc']=L
    goods = goods.join(pd.get_dummies(L,prefix='Loc',dummy_na=True), how='left')
    goods['Constant'] = 1.
    locations = [loc for loc in goods if loc.startswith('Loc')]
    
    if save: goods.to_pickle('../data/modified/ss-goods{}{}.df'.format(*year))
    return goods

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
        # results = pd.DataFrame({Var:model.params.append(Extras(model)) for Var,model in items(models)})
        results = pd.DataFrame({Var:model.params for Var,model in Items(models)})
        SEs     = pd.DataFrame({Var:model.bse    for Var,model in Items(models)})
        if table_info:
            extras = pd.DataFrame({Var: pd.Series({name:stat(model) for name,stat in Items(info_dict)}) for Var,model in Items(models)})
            results = results.append(extras)
        if Transpose: results, SEs = results.T, SEs.T

        if orgtbl: Summary = df_to_orgtbl(results,sedf=SEs)
        else:
            assert(resultdf)
            Summary = results, SEs

    return Summary

def asset_vars(D, year=2014, append=False,logs = False, topcode_prices=3, output=False):
    """
    Construct asset variables for year:
        Total asset value
        Total productive asset value
    Note: value colums have format asset_val_{good}, quantity colums have format asset_n_{good}
    topcode_prices --> If inferred price (val/n) is >mean+3sigma, set to mean+3sigma
    TODO: Rename all columns to be the same in An, Aval, price
    """
    A = D.filter(regex="^asset_")
    #~ Some assets to ignore, either because numbers turned out to be more or less meaningless, or because they overlap (e.g. nets & ITN nets)
    A.drop([col for col in A.columns if any([good in col for good in ('house','homeste','ITN')])], axis=1,inplace=True)
    if year==2014:   A=A.filter(regex="_m$").rename(columns=lambda col: col[:-2])
    elif year==2013: A=A.filter(regex="_b$").rename(columns=lambda col: col[:-2])
    elif year==2015: A=A.filter(regex="_e$").rename(columns=lambda col: col[:-2])
    An = A.filter(like='_n_').rename(columns=lambda col: col[8:])
    Aval = A.filter(like='_val_').rename(columns=lambda col: col[10:]) 
    price = Aval.divide(An)
    if topcode_prices: #~ Made necessary by a very long right tail.
        for good in price.columns:
            x = price[good]
            top = x.mean()+topcode_prices*x.std()
            x[x>top]=top
            price[good]=x
            Aval[good]=(x*An[good])
    #~ Make aggregate Assets & Productive Assets

    if year>2013:
        Aval['poultry']=Aval[['chickens','ducks']].sum(axis=1)
        An['poultry']=An[['chickens','ducks']].sum(axis=1)
                
    if output:
        Aval.to_pickle('/tmp/asset_values_%s.df' % year)    
        An.to_pickle('/tmp/asset_count_%s.df' % year)    
    Aval['Total'.format(year)] = Aval.sum(axis=1)
    if year>2013:
        productive=['cows', 'smallanimals', 'chickens', 'ducks', 'plough', 'shed', 'shop', 'pangas', 'axes', 'mobile', 'carts', 'sewing']
        livestock=['cows', 'smallanimals', 'chickens', 'ducks']
        Aval['Productive'.format(year)] = Aval[productive].sum(axis=1)
        Aval['Livestock'.format(year)] = Aval[livestock].sum(axis=1)
    elif year==2013:
        productive=['cows', 'smallanimals', 'poultry', 'plough', 'shed', 'shop', 'mobile', 'carts', 'sewing']
        livestock=['cows', 'smallanimals', 'poultry']
        Aval['Productive'.format(year)] = Aval[productive].sum(axis=1)
        Aval['Livestock'.format(year)] = Aval[livestock].sum(axis=1)


    if logs: Aval,An,price = map(lambda x: np.log(x.replace(0,np.e)), (Aval,An,price) )

    if append: D = D.merge(price, right_index=True, left_index=True)

    return Aval,An,price

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

if __name__ == '__main__':
    D = full_data()
    df = process_data(save=True)

