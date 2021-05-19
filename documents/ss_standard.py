import statsmodels.api as sm
from cfe.df_utils import df_to_orgtbl

def results(df,outcomes):
    # make interaction terms
    df.insert(len(df.columns), 'UPG*2013', df['2013']*df['TUP'])
    df.insert(len(df.columns), 'UPG*2014', df['2014']*df['TUP'])
    df.insert(len(df.columns), 'UPG*2015', df['2015']*df['TUP'])
    df.insert(len(df.columns), 'UCT*2013', df['2013']*df['UCT'])
    df.insert(len(df.columns), 'UCT*2014', df['2014']*df['UCT'])
    df.insert(len(df.columns), 'UCT*2015', df['2015']*df['UCT'])

    models_ols_FE = {}
    models_ols_no_FE = {}
    d_FE = {}
    d_no_FE = {}
    baseline_na = True

    #controls = ['UPG*2014', 'UPG*2015', 'UCT*2014', 'UCT*2015', '2014', '2015', 'UPG', 'UCT']
    controls = ['UPG*2014', 'UPG*2015', 'UCT*2014', 'UCT*2015', '2014', '2015']

    # remove observations from 2013
    df = df[df['Year'] != '2013']
    df.index.name = 'idno'
    df = df.reset_index().set_index(['idno','Year'])

    myX = {}
    myY = {}
    for outcome in outcomes: 
        temp_df = df[ [outcome, outcome + "2013", 'Control'] + controls]
        temp_df.rename(columns={outcome+"2013":'Baseline value'},inplace=True)
        temp_controls = controls
    
        if baseline_na==True:

            # indicator for whether outcome in 2013 is na, and cast it to be an integer
            temp_df["Baseline missing"] = temp_df["Baseline value"].isnull().apply(int)

            # code missing values of the baseline variable as 0
            temp_df["Baseline missing"].fillna(0,inplace=True)

            temp_controls = temp_controls + ['Baseline missing']

        temp_controls = temp_controls + ["Baseline value"]
        temp_df = temp_df.dropna()

        myX[outcome] = temp_df[temp_controls]
        myY[outcome] = temp_df[outcome]

    myY = pd.concat(myY)
    myX = pd.DataFrame(block_diag(*myX.values()),
                       columns=pd.concat(myX,axis=1).columns,
                       index=myY.index)

    est = sm.OLS(myY,myX).fit()
    b = est.params
    se = est.bse

    b.index = pd.MultiIndex.from_tuples([tuple(i.split('_')) for i in est.params.index])
    se.index = pd.MultiIndex.from_tuples([tuple(i.split('_')) for i in est.params.index])    


    b = b.unstack()
    b['UPG*2014 - UCT*2015'] = b['UPG*2014'] - b['UCT*2015']
    b['UPG*2015 - UCT*2015'] = b['UPG*2015'] - b['UCT*2015']
    b = b.T

    se = se.unstack()
    se[r'$\beta_{UPG}^{2014} - \beta_{UCT}^2015'] = np.sqrt(se['UPG*2014']**2 + se['UCT*2015']**2)
    se['UPG*2015 - UCT*2015'] = np.sqrt(se['UPG*2015']**2 + se['UCT*2015']**2)

    se = se.T

    latex_labels = {'UPG*2014 - UCT*2015':r'$\beta_{UPG}^{2014} - \beta_{UCT}^2015',
                    'UPG*2015 - UCT*2015':r'$\beta_{UPG}^{2015} - \beta_{UCT}^2015',
                    'UPG*2014':r'$UPG\times 2014$'
                    'UPG*2015':r'$UPG\times 2015$'}

    b = b.rename(index=latex_labels)
    se = se.rename(index=latex_labels)

    return df_to_orgtbl(b,sedf=se,float_fmt='%6.2f')

