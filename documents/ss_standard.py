import statsmodels.api as sm
from cfe.df_utils import df_to_orgtbl, ols
import pandas as pd
from scipy.linalg import block_diag
import numpy as np

def results(df, outcomes, baseline_na=True,logs=False,positive=False,elide=False,return_stats=False):

    df = df.copy()
    # make interaction terms

    try: # Not all outcomes observed in multiple years
        df.insert(len(df.columns), 'UPG*2013', df['2013']*df['TUP'])
        df.insert(len(df.columns), 'UPG*2014', df['2014']*df['TUP'])
        df.insert(len(df.columns), 'UPG*2015', df['2015']*df['TUP'])
        df.insert(len(df.columns), 'UCT*2013', df['2013']*df['UCT'])
        df.insert(len(df.columns), 'UCT*2014', df['2014']*df['UCT'])
        df.insert(len(df.columns), 'UCT*2015', df['2015']*df['UCT'])

        controls = ['UPG*2014', 'UPG*2015', 'UCT*2014', 'UCT*2015', '2014', '2015']

        # remove observations from 2013
        df = df[df['Year'] != '2013']
        df.index.name = 'idno'
        df = df.reset_index().set_index(['idno','Year'])
    except KeyError:
        df['Constant'] = 1
        df.rename(columns={'TUP':'UPG'},inplace=True)
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
                    # code missing values of the baseline variable as 0
                    temp_df["Baseline missing"].fillna(0,inplace=True)
                    temp_controls = temp_controls + ['Baseline missing']
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

    if positive:
        myY = (myY>0) + 0
        for v in myX.columns.levels[0]:
            try:
                myX[(v,'Baseline value')] = (myX[(v,'Baseline value')]>0) + 0
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
        #b,se = ols(myX.xs(outcome,level='Outcome').xs(outcome,level='Outcome',axis=1),myY.xs(outcome,level='Outcome'))
        est = sm.OLS(myY.xs(outcome,level='Outcome'),myX.xs(outcome,level='Outcome').xs(outcome,level='Outcome',axis=1)).fit()
        B[outcome] = est.params
        SE[outcome] = pd.Series(est.get_robustcov_results(cov_type='HC3').bse,index=B[outcome].index)

    #b.index = pd.MultiIndex.from_tuples([tuple(i.split('_')) for i in est.params.index])
    #se.index = pd.MultiIndex.from_tuples([tuple(i.split('_')) for i in est.params.index])    

    b = pd.DataFrame(B).T #b.unstack().T[outcomes].T
    se = pd.DataFrame(SE).T #se.unstack().T[outcomes].T
   
    try:
        b['UPG*2014 - UCT*2015'] = b['UPG*2014'] - b['UCT*2015']
        b['UPG*2015 - UCT*2015'] = b['UPG*2015'] - b['UCT*2015']

        se['UPG*2014 - UCT*2015'] = np.sqrt(se['UPG*2014']**2 + se['UCT*2015']**2)
        se['UPG*2015 - UCT*2015'] = np.sqrt(se['UPG*2015']**2 + se['UCT*2015']**2)
    except KeyError: # No treatment-year interactions?
        b['UPG - UCT'] = b['UPG'] - b['UCT']
        se['UPG - UCT'] = np.sqrt(se['UPG']**2 + se['UCT']**2)

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

def table_stacked_by_class(Results, elide=True, transpose=False):
    b = pd.DataFrame({k:r[0].stack() for k,r in Results.items()})
    b.index.names = ['Variable','Outcome']
    b.columns.name = 'Class'
    b = b.stack().unstack('Variable').reorder_levels(['Class','Outcome']).sort_index()
    b = b.reindex(Results.keys(),level='Class')

    se = pd.DataFrame({k:r[1].stack() for k,r in Results.items()})
    se.index.names = ['Variable','Outcome']
    se.columns.name = 'Class'
    se = se.stack().unstack('Variable').reorder_levels(['Class','Outcome']).sort_index()
    se = se.reindex(Results.keys(),level='Class')

    bonus = pd.DataFrame({k:r[2].stack() for k,r in Results.items()})
    bonus.index.names = ['Variable','Outcome']
    bonus.columns.name = 'Class'
    bonus = bonus.stack().unstack('Variable').reorder_levels(['Class','Outcome']).sort_index()
    N = pd.DataFrame({'N':bonus.reindex(Results.keys(),level='Class')['N'].apply(lambda x: '$N=%d$' % x)})
    
    if elide:
        b = b.filter(regex='^\$')
        se = se.filter(regex='^\$')
        
    return df_to_orgtbl(b,sedf=se,float_fmt='\(%6.2f\)',bonus_stats=N)
