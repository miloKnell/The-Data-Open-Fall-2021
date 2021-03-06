'''
0: real
1: fake
2: satire
'''


#write a json file with x and y for real/fake news articles
'''
import os
import random
import json
def read_horne(path):
    titles = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), encoding='utf-8', errors='ignore') as f:
            titles.append(f.read())
    return titles

buzfeed_real = read_horne('Public Data/Buzzfeed Political News Dataset/Real_titles')
buzzfeed_fake = read_horne('Public Data/Buzzfeed Political News Dataset/Fake_titles')

pol_real = read_horne('Public Data/Random Poltical News Dataset/Real_titles')
pol_fake = read_horne('Public Data/Random Poltical News Dataset/Fake_titles')
pol_satire = read_horne('Public Data/Random Poltical News Dataset/Satire_titles')



raw_x = buzfeed_real + buzzfeed_fake + pol_real + pol_fake + pol_satire
raw_y = len(buzfeed_real)*[0] + len(buzzfeed_fake)*[1] + len(pol_real)*[0] + len(pol_fake)*[1] + len(pol_satire)*[2]

c = list(zip(raw_x,raw_y))

random.shuffle(c)

x,y = zip(*c)

data = {'x':x, 'y':y}

#with open('data.json', 'w') as f:
#    f.write(json.dumps(data))
'''







#write a csv with NELA features for the real/fake news dataset
'''
import json
from my_nela_features import NELAFeatureExtractor
import numpy as np
import pandas as pd

#with open('data.json') as f:
#    data = json.loads(f.read())

raw_x = data['x']
raw_y = data['y']

nela = NELAFeatureExtractor()

features = []
for headline in raw_x:
    feature_vector, feature_names = nela.extract_all(headline)
    features.append(feature_vector)


x = np.array(features)
y = np.array(raw_y)


df = pd.DataFrame(np.hstack((x,y.reshape(-1, 1))), columns=feature_names+['y'])
'''




#extract features from packages
'''import pandas as pd
from my_nela_features.nela_features import NELAFeatureExtractor
import json

df = pd.read_csv('packages.csv')
df = df[df['headline'].notna()].reset_index()
nela = NELAFeatureExtractor()


features = []
for headline in df['headline']:
    feature_vector, feature_names = nela.extract_all(headline)
    features.append(feature_vector)

with open('new_package_feats.json', 'w') as f:
    f.write(json.dumps(features))

#feats = pd.concat([old_feature_df, feature_df], axis=0)
# full_df = pd.concat([df.reset_index(), feats.reset_index()], axis=1)

#feats = old_feature_df.append(feature_df, sort=False).reset_index().drop(columns=['index', 'level_0'])

feats = pd.DataFrame(features, columns=feature_names)
#feats = pd.concat([old_feature_df, feature_df], axis=0)
full_df = pd.concat([df, feats], axis=1)
full_df.to_csv('package_feats.csv')
'''

'''
import pandas as pd
df = pd.read_csv('package_rescale.csv')
sf = pd.read_csv('softrank.csv')


filtered = df[df.index.isin(sf.index)]
filtered['performance'] = sf['performance']


new_df = []
for name, group in df.groupby('test_id'):
    if name in sf['test_id']
'''