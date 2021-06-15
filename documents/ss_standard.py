import statsmodels.api as sm
from metrics_miscellany.estimators import ols
from metrics_miscellany import utils
from cfe.df_utils import df_to_orgtbl, use_indices, drop_missing
import pandas as pd
from scipy.linalg import block_diag
from scipy import stats
import numpy as np
import re

def basic_data(outcomes=None, filter=None):

    data_assignment  = "../../Report/documents/master_assignment.csv"
    data_baseline = "../../TUP-data/data/Baseline/TUP_baseline.dta"
    data_midline  = "../../TUP-data/Midline/TUP_midline.dta"
    data_endline = "../../TUP-data/Endline/TUP_endline.dta"

    DFs = []
    
    # Baseline
    df = pd.read_stata(data_baseline)
    df['Year'] = '2013'
    df.rename(columns=dict(zip([s for s in df.columns.to_list() if s[-2:]=='_b'],
                               [s[:-2] for s in df.columns.to_list() if s[-2:]=='_b'])),
              inplace=True)

    DFs.append(df)        
    
    df = pd.read_stata(data_midline)
    df['Year'] = '2014'
    df.rename(columns=dict(zip([s for s in df.columns.to_list() if  s[-2:]=='_m'],
                               [s[:-2] for s in df.columns.to_list() if s[-2:]=='_m'])),
              inplace=True)

    DFs.append(df)        

    df = pd.read_stata(data_endline)
    df['Year'] = '2015'
    
    df.rename(columns=dict(zip([s for s in df.columns.to_list() if s[-2:]=='_e'],
                               [s[:-2] for s in df.columns.to_list() if s[-2:]=='_e'])),
              inplace=True)

    DFs.append(df)

    df = pd.concat(DFs,axis=0)
    df['idno'] = df.idno.astype(int)
    df.set_index(['idno','Year'],inplace=True)

    DFs = []
    if outcomes is not None:
        DFs.append(df[df.columns.intersection(outcomes)])

    if filter is not None:
        use = df.filter(regex=filter).columns
        df1 = df.filter(regex=filter)
        c = re.compile(r'('+filter+')')
        DFs.append(df1.rename(columns=dict(zip(use,[c.sub('',s) for s in df1.columns.tolist()]))))

    df = pd.concat(DFs,axis=1)

    # Find original assignment
    assignment = pd.read_csv(data_assignment).rename(columns={'RespID':'idno'})[['idno','Group']].set_index('idno')
    df = df.join(assignment,on='idno')

    return df

def ancova_form(df,outcomes):
    """
    Return dataframe in "ANCOVA" form, with dummies for years, treatment groups, and baseline values.
    """

    years = pd.get_dummies(use_indices(df,['Year']).squeeze())
    groups = pd.get_dummies(df['Group'])

    baseline = df.xs('2013',level='Year')[outcomes]

    df = pd.concat([df[outcomes],years,groups],axis=1)

    df = df.join(baseline,on='idno',rsuffix='2013')

    return df.query("Year in ['2014','2015']")
    

def results(df,outcomes,controls=None,baseline_na=True,logs=False,nonzero=False,elide=False,return_stats=False):

    df = df.copy()
    # make interaction terms

    try: # Not all outcomes observed in multiple years
        df.insert(len(df.columns), 'UPG*2013', df['2013']*df['TUP'])
        df.insert(len(df.columns), 'UPG*2014', df['2014']*df['TUP'])
        df.insert(len(df.columns), 'UPG*2015', df['2015']*df['TUP'])
        df.insert(len(df.columns), 'UCT*2013', df['2013']*df['UCT'])
        df.insert(len(df.columns), 'UCT*2014', df['2014']*df['UCT'])
        df.insert(len(df.columns), 'UCT*2015', df['2015']*df['UCT'])

        if controls is None:
            controls = ['UPG*2014', 'UPG*2015', 'UCT*2014', 'UCT*2015', '2014', '2015']

        # remove observations from 2013
        df = df[df['Year'] != '2013']
        df.index.name = 'idno'
        df = df.reset_index().set_index(['idno','Year'])
    except KeyError:
        df['Constant'] = 1
        df.rename(columns={'TUP':'UPG'},inplace=True)
        if controls is None:
            controls = ['UPG', 'UCT', 'Constant']
        df.index.name = 'idno'

    myX = {}
    myY = {}
    for outcome in outcomes:
        temp_controls = controls
        try:
            temp_df = df[ [outcome, outcome + "2013", 'Control'] + controls]
            temp_df.rename(columns={outcome+"2013":'Baseline value'},inplace=True)
            temp_controls = temp_controls + ["Baseline value"]
            if baseline_na:

                # indicator for whether outcome in 2013 is na, and cast it to be an integer
                missings = temp_df["Baseline value"].isnull().apply(int)
                if missings.sum()>0:
                    temp_df["Baseline missing"] = missings
                    temp_controls = temp_controls + ['Baseline missing']
                temp_df['Baseline value'] = temp_df['Baseline value'].fillna(0)
        except KeyError: # No baseline?
            temp_df = df[ [outcome, 'Control'] + controls]
    
        temp_df = temp_df.dropna()

        myX[outcome] = temp_df[temp_controls]
        myY[outcome] = temp_df[outcome]

    myY = pd.concat(myY)
    myX = pd.DataFrame(block_diag(*myX.values()),
                       columns=pd.concat(myX,axis=1).columns,
                       index=myY.index)
    myX.columns.names = ['Outcome','Variable']
    try:
        myY.index.set_names(['Outcome','idno','Year'],inplace=True)
    except ValueError:
        myY.index.set_names(['Outcome','idno'],inplace=True)

    if nonzero:
        myY = (myY != 0) + 0
        for v in myX.columns.levels[0]:
            try:
                myX[(v,'Baseline value')] = (myX[(v,'Baseline value')]!=0) + 0
            except KeyError:
                pass

    if logs:
        myY = np.log(myY.replace(0,np.nan))
        for v in myX.columns.levels[0]:
            try:
                myX[(v,'Baseline value')] = np.log(myX[(v,'Baseline value')].replace(0,np.nan))
            except KeyError:
                pass
        keep = ~np.isnan(myY)
        myY = myY[keep]
        myX = myX[keep]
        myX = myX.replace(np.nan,0)

    B = {}
    SE = {}
    for outcome in outcomes:
        try:
            b,V = ols(myX.xs(outcome,level='Outcome').xs(outcome,level='Outcome',axis=1),myY.xs(outcome,level='Outcome'))
            B[outcome] = b.squeeze()
            SE[outcome] = pd.Series(np.sqrt(np.diag(V)),index=B[outcome].index)
        except KeyError: pass

    #b.index = pd.MultiIndex.from_tuples([tuple(i.split('_')) for i in est.params.index])
    #se.index = pd.MultiIndex.from_tuples([tuple(i.split('_')) for i in est.params.index])    

    b = pd.DataFrame(B).T 
    se = pd.DataFrame(SE).T 
   
    try:
        b['UPG*2014 - UCT*2015'] = b['UPG*2014'] - b['UCT*2015']
        b['UPG*2015 - UCT*2015'] = b['UPG*2015'] - b['UCT*2015']

        se['UPG*2014 - UCT*2015'] = np.sqrt(se['UPG*2014']**2 + se['UCT*2015']**2)
        se['UPG*2015 - UCT*2015'] = np.sqrt(se['UPG*2015']**2 + se['UCT*2015']**2)
    except KeyError: # No treatment-year interactions?
        try:
            b['UPG - UCT'] = b['UPG'] - b['UCT']
            se['UPG - UCT'] = np.sqrt(se['UPG']**2 + se['UCT']**2)
        except KeyError: # No pair of treatments at all
            pass


    b = b.T
    se = se.T

    latex_labels = {'UPG*2014 - UCT*2015':r'$\beta_{UPG}^{2014} - \beta_{UCT}^{2015}$',
                    'UPG*2015 - UCT*2015':r'$\beta_{UPG}^{2015} - \beta_{UCT}^{2015}$',
                    'UPG*2014':r'$\beta_{UPG}^{2014}$',
                    'UPG*2015':r'$\beta_{UPG}^{2015}$',
                    'UCT*2014':r'$\beta_{UCT}^{2014}$',
                    'UCT*2015':r'$\beta_{UCT}^{2015}$',
                    'UPG - UCT':r'$\beta_{UPG}-\beta_{UCT}$',
                    'UPG':r'$\beta_{UPG}$',
                    'UCT':r'$\beta_{UCT}$'}

    b = b.rename(index=latex_labels)
    b.index.name = 'Variable'
    se = se.rename(index=latex_labels)

    try:
        controls = 1-myX.loc[:,myX.columns.isin(['UPG*2014','UPG*2015','UCT*2014','UCT*2015'],level='Variable')].sum(axis=1)
    except KeyError:
        controls = 1-myX.loc[:,myX.columns.isin(['UPG','UCT'],level='Variable')].sum(axis=1)

    ybar = myY.loc[controls==1.]

    try:
        otherstats = pd.DataFrame({'2014':ybar.unstack(level=0).xs('2014',level='Year').mean(),
                                   '2015':ybar.unstack(level=0).xs('2015',level='Year').mean(),
                                   'N':myY.unstack('Outcome').count()}).T
    except TypeError:
        otherstats = pd.DataFrame({'N':myY.unstack('Outcome').count()}).T

    if return_stats:
        return b,se,otherstats
    
    if elide:
        b = b.filter(regex='^\$',axis=0)
        se = se.filter(regex='^\$',axis=0)
        
    Table = df_to_orgtbl(b,sedf=se,float_fmt='\(%6.2f\)')[:-1].split('\n') + df_to_orgtbl(otherstats,float_fmt='\(%6.2f\)').split('\n')[1:]

    return '\n'.join(Table)

def system_data(df,outcomes,controls=None,baseline_na=False,logs=False,nonzero=False,elide=False,return_stats=False):

    df = df.copy()
    # make interaction terms

    try: # Not all outcomes observed in multiple years
        df.insert(len(df.columns), 'UPG*2013', df['2013']*df['TUP'])
        df.insert(len(df.columns), 'UPG*2014', df['2014']*df['TUP'])
        df.insert(len(df.columns), 'UPG*2015', df['2015']*df['TUP'])
        df.insert(len(df.columns), 'UCT*2013', df['2013']*df['UCT'])
        df.insert(len(df.columns), 'UCT*2014', df['2014']*df['UCT'])
        df.insert(len(df.columns), 'UCT*2015', df['2015']*df['UCT'])

        if controls is None:
            controls = ['UPG*2014', 'UPG*2015', 'UCT*2014', 'UCT*2015', '2014', '2015']

        # remove observations from 2013
        df = df[df['Year'] != '2013']
        df.index.name = 'idno'
        df = df.reset_index().set_index(['idno','Year'])
    except KeyError:
        df['Constant'] = 1
        df.rename(columns={'TUP':'UPG'},inplace=True)
        if controls is None:
            controls = ['UPG', 'UCT', 'Constant']
        df.index.name = 'idno'

    myX = {}
    myY = {}
    for outcome in outcomes:
        temp_controls = controls
        try:
            temp_df = df[ [outcome, outcome + "2013", 'Control'] + controls]
            temp_df.rename(columns={outcome+"2013":'Baseline value'},inplace=True)
            temp_controls = temp_controls + ["Baseline value"]
            if baseline_na:

                # indicator for whether outcome in 2013 is na, and cast it to be an integer
                missings = temp_df["Baseline value"].isnull().apply(int)
                if missings.sum()>0:
                    temp_df.loc[:,"Baseline missing"] = missings
                    temp_controls = temp_controls + ['Baseline missing']
                temp_df.loc[:,'Baseline value'] = temp_df['Baseline value'].fillna(0)
        except KeyError: # No baseline?
            temp_df = df[ [outcome, 'Control'] + controls]
    
        temp_df = temp_df.dropna()
        if temp_df[outcome].std()>0:
            myX[outcome] = temp_df[temp_controls]
            myY[outcome] = temp_df[outcome]
        else:
            outcomes.remove(outcome)

    Ybar = pd.concat(myY)
    Xbar = pd.DataFrame(block_diag(*myX.values()),
                       columns=pd.concat(myX,axis=1).columns,
                       index=Ybar.index)
    
    Xbar.columns.names = ['Outcome','Variable']
    try:
        Ybar.index.set_names(['Outcome','idno','Year'],inplace=True)
    except ValueError:
        Ybar.index.set_names(['Outcome','idno'],inplace=True)

    if nonzero:
        Ybar = (Ybar != 0) + 0
        for v in Xbar.columns.levels[0]:
            try:
                Xbar[(v,'Baseline value')] = (Xbar[(v,'Baseline value')]!=0) + 0
            except KeyError:
                pass

    if logs:
        Ybar = np.log(Ybar.replace(0,np.nan))
        for v in Xbar.columns.levels[0]:
            try:
                Xbar[(v,'Baseline value')] = np.log(Xbar[(v,'Baseline value')].replace(0,np.nan))
            except KeyError:
                pass
        keep = ~np.isnan(Ybar)
        Ybar = Ybar[keep]
        Xbar = Xbar[keep]
        Xbar = Xbar.fillna(0)

    return Xbar,Ybar

def system_estimation(Xbar,Ybar):
    
    Ybar,Xbar = drop_missing([Ybar,Xbar])

    Xbar = Xbar.loc[:,Xbar.std()>0]
    
    b = np.linalg.lstsq(Xbar.T@Xbar,Xbar.T@Ybar,rcond=None)[0]

    e = (Ybar - Xbar@b).squeeze().unstack('Outcome')

    b = pd.DataFrame({'Coefficients':b.squeeze()},index=Xbar.columns)

    # "Collapsed" version of Xbar
    X = Xbar.groupby(['idno','Year']).sum()
    
    Xe = X.multiply(e.reindex(X.columns,axis=1,level=0))
    XeeX = Xe.cov(min_periods=2) + Xe.mean().T@Xe.mean()

    XeeX = (XeeX.T + XeeX)/2 # Make symmetric!
    XeeX = utils.cov_nearest(XeeX,threshold=1e-5)
    XeeX = pd.DataFrame(XeeX,index=Xe.columns,columns=Xe.columns)

    XXinv = {}
    working_outcomes = list(set(Xbar.index.get_level_values('Outcome')))
    working_columns = []
    for k in working_outcomes:
        try:
            v = X.xs(k,level='Outcome',axis=1)
            XXinv[k] = np.linalg.inv((v.dropna().T@v.dropna())) # + np.eye(v.shape[1])*1)
            assert len(Xbar.columns[Xbar.columns.isin([k],level=0)])==XXinv[k].shape[0]
            working_columns += [(k,j) for j in v.columns]
        except np.linalg.LinAlgError:
            working_outcomes.remove(k)

    usecols = pd.MultiIndex.from_tuples(working_columns)

    XXinv = pd.DataFrame(block_diag(*XXinv.values()),
                         columns=usecols,
                         index=usecols)

    XeeX = XeeX.reindex_like(XXinv)
    Vb = XXinv@XeeX@XXinv

    b = b.reindex(Vb.index) # Drop estimated coefficients without covariances
    b.index.names = ['Outcome','Variable']

    return b,Vb


def residuals(df,outcomes,controls=None,baseline_na=True,elide=False):

    df = df.copy()
    # make interaction terms

    if controls is None:
        controls = ['2014', '2015']

    # remove observations from 2013
    df = df.query("Year in ['2014','2015']")

    myX = {}
    myY = {}
    for outcome in outcomes:
        temp_controls = controls
        try:
            temp_df = df[ [outcome, outcome + "2013"] + controls]
            temp_df.rename(columns={outcome+"2013":'Baseline value'},inplace=True)
            temp_controls = temp_controls + ["Baseline value"]
            if baseline_na:

                # indicator for whether outcome in 2013 is na, and cast it to be an integer
                missings = temp_df["Baseline value"].isnull().apply(int)
                if missings.sum()>0:
                    temp_df["Baseline missing"] = missings
                    temp_controls = temp_controls + ['Baseline missing']
                temp_df['Baseline value'] = temp_df['Baseline value'].fillna(0)
        except KeyError: # No baseline?
            temp_df = df[ [outcome] + controls]
    
        temp_df = temp_df.dropna()

        myX[outcome] = temp_df[temp_controls]
        myY[outcome] = temp_df[outcome]

    myY = pd.concat(myY)
    myX = pd.DataFrame(block_diag(*myX.values()),
                       columns=pd.concat(myX,axis=1).columns,
                       index=myY.index)
    myX.columns.names = ['Outcome','Variable']
    try:
        myY.index.set_names(['Outcome','idno','Year'],inplace=True)
    except ValueError:
        myY.index.set_names(['Outcome','idno'],inplace=True)


    E = {}
    for outcome in outcomes:
        try:
            est = sm.OLS(myY.xs(outcome,level='Outcome'),myX.xs(outcome,level='Outcome').xs(outcome,level='Outcome',axis=1)).fit()
            E[outcome] = est.resid 
        except KeyError: pass

    return pd.DataFrame(E)

def table_stacked_by_class(Results, elide=True, transpose=False):
    b = pd.DataFrame({k:r[0].stack() for k,r in Results.items()})
    b.index.names = ['Variable','Outcome']
    b.columns.name = 'Class'
    b = b.stack().unstack('Variable').reorder_levels(['Class','Outcome']).sort_index(level='Class')
    b = b.reindex(Results.keys(),level='Class')

    se = pd.DataFrame({k:r[1].stack() for k,r in Results.items()})
    se.index.names = ['Variable','Outcome']
    se.columns.name = 'Class'
    se = se.stack().unstack('Variable').reorder_levels(['Class','Outcome']).sort_index(level='Class')
    se = se.reindex(Results.keys(),level='Class')

    bonus = pd.DataFrame({k:r[2].stack() for k,r in Results.items()})
    bonus.index.names = ['Variable','Outcome']
    bonus.columns.name = 'Class'
    bonus = bonus.stack().unstack('Variable').reorder_levels(['Class','Outcome']).sort_index(level='Class')
    N = pd.DataFrame({'N':bonus.reindex(Results.keys(),level='Class')['N'].apply(lambda x: '$N=%d$' % x)})
    
    if elide:
        b = b.filter(regex='^\$')
        se = se.filter(regex='^\$')
        
    return df_to_orgtbl(b,sedf=se,float_fmt='\(%6.2f\)',bonus_stats=N)
