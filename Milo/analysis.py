import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import stats
from statsmodels.formula.api import ols
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv('clean_data_filtered.csv')
df['click_rate'] = df['clicks'] / df['impressions']
df['first_place'] = df['first_place'].astype(int)
df['winner'] = df['winner'].astype(int)
df['date'] = pd.to_datetime(df['created_at'])
delta = df['date'] - min(df['date'])
df['days'] = delta.apply(lambda x: x.days)

#winners = df[df['winner']==1]

#formula = 'pred ~ click_rate + winner + ifk '


def anova(formula,typ=2,df=df):
    lm = ols(formula, data=df).fit()
    return stats.anova_lm(lm, typ=typ)


no_overrides = df[df['first_place'] == df['winner']]
overrides = df[df['first_place'] != df['winner']]


win_match = (df['winner'] == df['rescaled_pred']).sum()
win_not_match = len(df) - win_match

first_match = (df['first_place'] == df['rescaled_pred']).sum()
first_not_match = len(df) - first_match

n_cutoff = 750

pre_overrides = overrides[overrides['days']<n_cutoff]
pre_no_overrides = no_overrides[no_overrides['days']<n_cutoff]

post_overrides = overrides[overrides['days']>n_cutoff]
post_no_overrides = no_overrides[no_overrides['days']>n_cutoff]


def get_day_pred(df):
    gb= df.groupby('days')
    day_n = gb.apply(lambda x: len(x))
    day = gb['days'].apply(lambda x: x.name)
    date = gb['date']
    day_overrides = gb.apply(lambda x: (x['first_place'] != x['winner']).sum()) / day_n
    day_pred= gb.apply(lambda x: x['pred'].sum()) / day_n
    day_clicks = gb.apply(lambda x: x['click_rate'].sum())/ day_n
    return date, day_pred, day_clicks

day, day_pred, day_clicks = get_day_pred(df)

override_day, override_pred, override_clicks = get_day_pred(overrides)
no_override_day, no_override_pred, no_override_clicks = get_day_pred(no_overrides)

'''plt.plot(override_day, savgol_filter(override_pred, 41, 3), label='overrides')
plt.plot(no_override_day, savgol_filter(no_override_pred, 41, 3), label='no overrides')
'''

'''plt.plot(override_day, override_pred, label='overrides')
plt.plot(no_override_day, no_override_pred, label='no overrides')'''


'''
override_day, override_pred = get_day_pred(overrides)
no_override_day, no_override_pred = get_day_pred(no_overrides)

plt.plot(override_day, override_pred, label='overrides')
plt.plot(no_override_day, no_override_pred, label='no overrides')
plt.legend()
plt.show()'''


'''w=[(col, spearmanr(df[col], df['click_rate'])) for col in df.loc[:, 'quotes':'MoralityGeneral']]
s=sorted(w, key=lambda x: abs(x[1][0]))

for q in s:
	print(q)'''

views = pd.read_csv('analytics_daily_pageviews.csv', thousands=',')
views['date'] = pd.to_datetime(views['date'])
views = views[views['date'] >= min(df['date'])]
delta = views['date'] - min(views['date'])
views['days'] = delta.apply(lambda x: x.days)
#plt.plot(views['date'], savgol_filter(views['pageviews']/max(views['pageviews']), 41, 3), label='views')



w=[(col, spearmanr(df[col], df['click_rate'])) for col in df.loc[:, 'quotes':'MoralityGeneral']]
s=sorted(w, key=lambda x: abs(x[1][0]))

#for q in s:
#	print(q)





'''
fig 3.2.2.1: smoothed days vs date
'''
