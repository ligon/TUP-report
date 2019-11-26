DATA="../../TUP-data/TUP_full.dta"
"""
Create table comparing baseline means across treatments; also build analysis dataframes:
   - C : Consumption
   - Z : HH characteristics
   - Avalue : Values of assets
"""

import pandas as pd
import numpy as np
from cfe.df_utils import df_to_orgtbl

full = pd.read_stata(DATA)

full.rename(columns={'idno':'HH', "Control":"CTL", "Cash":"CSH",'location_b':"Location"},inplace=True)
full.set_index(["HH","Location"],inplace=True,drop=True)

full.loc[full.query("TUP>0").index,'Treatment'] = 'TUP'
full.loc[full.query("CTL>0").index,'Treatment'] = 'CTL'
full.loc[full.query("CSH>0").index,'Treatment'] = 'CSH'

# Build dataframe indicating treatment status
Tmt = full[['CTL','CSH','TUP']]
Tmt["CTL2"] = 1 - Tmt[["TUP","CSH"]].sum(1)

mergedict = {'master only (1)':  1, 'using only (2)':  2, 'matched (3)':  3}
for col in full.filter(like='merge_').columns:
    full[col]=full[col].apply(lambda i: mergedict.get(i)).astype(float)

Tmt['Base'] =  full['merge_census_b']>1
Tmt['Mid']  =  full['merge_midline']>1
Tmt['End']  =  full['merge_endline']>1

# Reorganize full so that period indicates baseline, midline, endline.

baseline = full.filter(axis='columns',regex="_b$").rename(columns=lambda s: s[:-2])
baseline['Period'] = 'Baseline'
baseline = baseline.reset_index().set_index(['HH','Period','Location'])

midline = full.filter(axis='columns',regex="_m$").rename(columns=lambda s: s[:-2])
midline['Period'] = 'Midline'
midline = midline.reset_index().set_index(['HH','Period','Location'])

endline = full.filter(axis='columns',regex="_e$").rename(columns=lambda s: s[:-2])
endline['Period'] = 'Endline'
endline = endline.reset_index().set_index(['HH','Period','Location'])

ofull = pd.concat([baseline,midline,endline],axis=0)
ofull = ofull.reset_index().set_index(['HH','Period','Location'])


# Build dataframe of consumption

## Different recall frequencies
recall = {'daily':['c_cereals', 'c_maize', 'c_sorghum', 'c_millet',
                   'c_potato', 'c_sweetpotato', 'c_rice', 'c_bread',
                   'c_beans', 'c_oil', 'c_salt', 'c_sugar', 'c_meat',
                   'c_livestock', 'c_poultry', 'c_fish', 'c_egg',
                   'c_nuts', 'c_milk', 'c_vegetables', 'c_fruit',
                   'c_tea', 'c_spices', 'c_alcohol', 'c_otherfood']}

recall['monthly'] = ['c_fuel', 'c_medicine', 'c_airtime',
                     'c_cosmetics', 'c_soap', 'c_transport',
                     'c_entertainment', 'c_childcare', 'c_tobacco',
                     'c_batteries', 'c_church', 'c_othermonth']    

recall['annual'] = ['c_clothesfootwear', 'c_womensclothes',
                    'c_childrensclothes', 'c_shoes', 'c_homeimprovement',
                    'c_utensils', 'c_furniture', 'c_textiles',
                    'c_ceremonies', 'c_funerals', 'c_charities',
                    'c_dowry', 'c_other']

# Consumption dataframe
C = pd.concat([ofull[recall['daily']]/3,
               ofull[recall['monthly']]/30,
               ofull[recall['annual']]/365],axis=1).rename(columns = lambda s: s[2:].capitalize())

# Rename some columns
C = C.rename(columns={'Poultry':'Chicken, etc.',
                      'Sweetpotato':'Sweet potato',
                      'Otherfood':'Other foods',
                      'Othermonth':'Other monthly',
                      'Clothesfootwear':'Clothes & footwear',
                      'Womensclothes':"Women's clothing",
                      'Childrensclothes':"Children's clothing",
                      'Homeimprovement':'Home improvements'})

C.index.name = 'Good'

C.to_pickle('../var/C.df')

# Build dataframes for assets

Avalue = ofull.filter(regex="^asset_val_").rename(columns = lambda s: s[10:].capitalize())
Avalue.index.name = 'Asset'
Avalue.rename(columns={'Chairtables':'Chairs & Tables',
                       'Smallanimals':'Small animals'},inplace=True)

Anumber = ofull.filter(regex="^asset_n_").rename(columns = lambda s: s[8:].capitalize())
Anumber.rename(columns={'Chairtables':'Chairs & Tables',
                        'Smallanimals':'Small animals'},inplace=True)

# Drop some highly bogus variables
for v in ['House','Homestead','Netitn']:
    del Avalue[v]
    del Anumber[v]

Avalue.to_pickle('../var/Avalue.df')
Anumber.to_pickle('../var/Anumber.df')

# Build dataframe for some hh characteristics
hh_chars = {"hh_size":"HH size",
            "child_total": "# Children",
            'asset_n_house':'# Houses',
            'in_business':'In Business'}

Z = ofull[list(hh_chars.keys())].rename(columns=hh_chars)

Z.to_pickle('../var/Z.df')

def balanced_means(df,treatment):
    
    dft = df.reset_index('Location',drop=True).join(treatment.reset_index('Location')).reset_index().set_index(['HH','Period','Location'])

    dfmeans = dft.query("Period=='Baseline'").groupby('Treatment').mean().T.dropna()
    dfcount = dft.query("Period=='Baseline'").groupby('Treatment').count().T.dropna()
    dfse = dft.query("Period=='Baseline'").groupby('Treatment').std().T.dropna()/np.sqrt(dfcount)
    dfvar = dft.query("Period=='Baseline'").groupby('Treatment').var().T.dropna()/dfcount
    dft = dfmeans.copy()
    dft['CTL'] = 0
    dft['TUP'] = np.abs(dfmeans['TUP'] - dfmeans['CTL'])/np.sqrt(dfvar['TUP'] + dfvar['CTL'])
    dft['CSH'] = np.abs(dfmeans['CSH'] - dfmeans['CTL'])/np.sqrt(dfvar['CSH'] + dfvar['CTL'])

    return dfmeans[['CTL','TUP','CSH']],dfse[['CTL','TUP','CSH']],dft[['CTL','TUP','CSH']]


full[['Treatment']].to_pickle('../var/T.df')

Means, SEs, Ts = balanced_means(C,full['Treatment'])
Means.index.name = 'Consumption'
print(df_to_orgtbl(Means,sedf=None,tdf=Ts,float_fmt="%3.1f"),end='')

print('|-')

Means, SEs, Ts = balanced_means(Avalue,full['Treatment'])
Means.index.name = 'Asset'
print(df_to_orgtbl(Means,sedf=None,tdf=Ts,float_fmt="%3.1f"),end='')

print('|-')

Means, SEs, Ts = balanced_means(Z,full['Treatment'])
Means.index.name = 'Household characteristics'
print(df_to_orgtbl(Means,sedf=None,tdf=Ts,float_fmt="%3.1f"),end='')

print('|-')
Nstr = '|'.join(['%d ' % x for x in full.groupby('Treatment').count().T.max().tolist()])
print('| $N$ |' + Nstr + '|')
