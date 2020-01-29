import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('data/real_estate.csv')

"""Save the 'Id' column"""
train_ID = train['No']

"""Now drop the  'Id' column since it's unnecessary for  the prediction process."""
train.drop("No", axis=1, inplace=True)
print(train)

train.rename(columns={'Y house price of unit area': 'house_price'}, inplace=True)
train.rename(columns={'X2 house age': 'house_age'}, inplace=True)
train.rename(columns={'X3 distance to the nearest MRT station': 'distance_from_MRT'}, inplace=True)
train.rename(columns={'X4 number of convenience stores': 'no_of_store'}, inplace=True)


def data_details(df, pred=None):
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis=1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis=1,
                        sort=False)
        corr_col = 'corr ' + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col]

    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n', dtypes)
    print('___________________________')
    return str


details = data_details(train, 'house_price')
null_miss = details[['nulls', 'missing_ration']]
print(details.sort_values(by='corr house_price', ascending=False))
# print(null_miss)
print(train.describe().transpose())


"""Show relation between different features visually"""
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')


fig = plt.figure(figsize=(20, 15))
sns.set(font_scale=1.5)

# (Corr= 0.817185) Box plot overallqual/salePrice
fig1 = fig.add_subplot(221)
sns.scatterplot(x=train.house_age, y=train.house_price)

# (Corr= 0.700927) GrLivArea vs SalePrice plot
fig2 = fig.add_subplot(222)
sns.scatterplot(x=train.distance_from_MRT, y=train.house_price, hue=train.distance_from_MRT)

# (Corr= 0.680625) GarageCars vs SalePrice plot
fig3 = fig.add_subplot(223)
sns.scatterplot(x=train.no_of_store, y=train.house_price, hue=train.distance_from_MRT, palette='Spectral')

# (Corr= 0.680625) GarageCars vs SalePrice plot
fig4 = fig.add_subplot(224)
sns.scatterplot(x=train.no_of_store, y=train.house_price, hue=train.house_age, palette='Spectral')

plt.tight_layout()
plt.show()
