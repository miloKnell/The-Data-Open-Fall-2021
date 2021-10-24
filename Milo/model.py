import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



df = pd.read_csv('data.csv')
df = df[df['y'] != 2]
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#X_train, X_test, y_train, y_test = train_test_split( x_trans, y, test_size=0.2, random_state=42)

#pipe = Pipeline(steps=[('scalar', StandardScaler()), ('svc', LinearSVC(C=0.005994842503189409))])

#pipe.fit(x,y)

scalar = StandardScaler()
svc = LinearSVC(C=0.005995)

x_trans = scalar.fit_transform(x)
svc.fit(x_trans, y)


#package = pd.read_csv('package_feats.csv')
#x = package.loc[:, 'quotes':]
#package = pd.read_csv('softrank.csv')
#x = package.loc[:, 'quotes':'MoralityGeneral'].values


#x_trans = scalar.fit_transform(x)
#pred = svc.predict(x_trans)

#package['pred'] = pred
#package.to_csv('package_pred.csv')


'''coef = svc.coef_[0]
#normed = coef**2 / sum(coef**2)
normed  = coef/sum(coef)

z=zip(normed, list(df.columns)[:-1])
best = sorted(z,reverse=True)'''


def plot_coef(coef,names):
    plt.xticks(range(len(names)), names, rotation=90)
    plt.scatter(range(len(names)), coef)
    plt.show()

#plot_coef(coef, list(df.columns)[:-1])

def do_grid():
    grid = {'C':np.logspace(-4, 4, 10)}

    search = GridSearchCV(svc, grid, verbose=0)
    search.fit(x_trans,y)


    print(search.best_score_, search.best_params_)

    #best: C = 0.005994842503189409 with acc of 0.8010196078431372 on CV




#from sklearn.linear_model import LinearRegression
#reg = LinearRegression().fit(x_trans, y)
