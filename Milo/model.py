import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV




df = pd.read_csv('data.csv')
df = df[df['y'] != 2]
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#pipe = Pipeline(steps=[('scalar', StandardScaler()), ('svc', LinearSVC(C=0.005994842503189409))])

#pipe.fit(x,y)

scalar = StandardScaler()
svc = LinearSVC(C=0.005994842503189409)

x_trans = scalar.fit_transform(x)
svc.fit(x_trans, y)


package = pd.read_csv('package_feats.csv')
x = package.loc[:, 'quotes':]


a_x_trans = scalar.transform(x)
a_pred = svc.predict(a_x_trans)

b_x_trans = scalar.fit_transform(x)
b_pred = svc.predict(b_x_trans)

package['pred'] = a_pred
package['rescaled_pred'] = b_pred
package.to_csv('package_rescale.csv')


'''coef = pipe['svc'].coef_[0]
normed = coef**2 / sum(coef**2)

z=zip(normed, list(df.columns)[:-1])
best = sorted(z,reverse=True)'''


def plot_coef(coef,names):
    plt.xticks(range(len(names)), names, rotation=90)
    plt.scatter(range(len(names)), coef)
    plt.show()

#plot_coef(coef, list(df.columns)[:-1])

def do_grid():
    grid = {'svc__C':np.logspace(-4, 4, 10)}

    search = GridSearchCV(pipe, grid, scoring='accuracy')
    search.fit(x,y)


    print(search.best_score_, search.best_params_)

    #best: C = 0.005994842503189409 with acc of 0.8010196078431372 on CV
