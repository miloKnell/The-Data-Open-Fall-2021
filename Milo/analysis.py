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

no_overrides = df[df['first_place'] == df['winner']]
overrides = df[df['first_place'] != df['winner']]


win_match = (df['winner'] == df['rescaled_pred']).sum()
win_not_match = len(df) - win_match

first_match = (df['first_place'] == df['rescaled_pred']).sum()
first_not_match = len(df) - first_match

n_cutoff = 748

pre_overrides = overrides[overrides['days']<n_cutoff]
pre_no_overrides = no_overrides[no_overrides['days']<n_cutoff]

post_overrides = overrides[overrides['days']>n_cutoff]
post_no_overrides = no_overrides[no_overrides['days']>n_cutoff]


def get_day_pred(df):
    gb= df.groupby('days')
    day_n = gb.apply(lambda x: len(x))
    day = gb['days'].apply(lambda x: x.name)
    date = gb['date'].apply(lambda x: x.iloc[0].date())
    day_overrides = gb.apply(lambda x: (x['first_place'] != x['winner']).sum()) / day_n
    day_pred= gb.apply(lambda x: x['pred'].sum()) / day_n
    day_clicks = gb.apply(lambda x: x['click_rate'].sum())/ day_n
    return date, day_pred, day_clicks

day, day_pred, day_clicks = get_day_pred(df)

override_day, override_pred, override_clicks = get_day_pred(overrides)
no_override_day, no_override_pred, no_override_clicks = get_day_pred(no_overrides)

views = pd.read_csv('analytics_daily_pageviews.csv', thousands=',')
views['date'] = pd.to_datetime(views['date'])
views = views[views['date'] >= min(df['date'])]
delta = views['date'] - min(views['date'])
views['days'] = delta.apply(lambda x: x.days)


'''
#fig 3.2.2.1: date vs smoothed click_rate
color1 = 'tab:blue'
color2 = 'tab:red'

fig, ax1 = plt.subplots()
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily average click rate')
ax1.plot(day,savgol_filter(day_clicks, 41, 3), color=color1, label='average click rate')
ax1.tick_params(axis='y', labelcolor=color1)


ax2 = ax1.twinx()
ax2.set_ylabel('Daily percent headlines predicted to be fake news')
ax2.plot(day, savgol_filter(day_pred, 41, 3), color=color2, label='percent fake')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Comparison of Daily Click Rate and Daily Percent Fake News')

plt.show()

'''

'''
analysis 3.2.2
corr = spearmanr(df['click_rate'], df['pred'])
'''




'''
#fig 3.3.x.1

plt.plot(override_day, savgol_filter(override_pred, 41, 3), label='Overrides')
plt.plot(no_override_day, savgol_filter(no_override_pred, 41, 3), label='No overrides')

plt.xlabel('Date')
plt.ylabel('Daily Percent Clickbaits (predicted fake news)')
plt.legend()
plt.show()
'''

'''
#fig 3.3.x.2

plt.bar(day, override_pred-no_override_pred)
plt.show()
'''
