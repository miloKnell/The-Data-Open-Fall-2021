import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import stats
from statsmodels.formula.api import ols


df = pd.read_csv('package_rescale.csv')
df['click_rate'] = df['clicks'] / df['impressions']
df['first_place'] = df['first_place'].astype(int)
df['winner'] = df['winner'].astype(int)
dates = pd.to_datetime(df['created_at'])
delta = dates - min(dates)
df['days'] = delta.apply(lambda x: x.days)

winners = df[df['winner']==1]

#formula = 'pred ~ click_rate + winner + ifk '


def anova(formula,typ=2,df=df):
    lm = ols(formula, data=df).fit()
    return stats.anova_lm(lm, typ=typ)


no_overrides = df[df['first_place'] == df['winner']]
overrides = df[df['first_place'] != df['winner']]

#real = df[df['rescaled_pred']==0]
#fake = df[df['rescaled_pred']==1]

#plt.boxplot([real['clicks'], fake['clicks']], showfliers=False)
#plt.boxplot([real['impressions'], fake['impressions']], showfliers=False)
#plt.boxplot([real['click_rate'], fake['click_rate']], showfliers=False)

#plt.scatter(df.groupby(['days'])['days'], df.groupby(['days']).mean()['click_rate'])
