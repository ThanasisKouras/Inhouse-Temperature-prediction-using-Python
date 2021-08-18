import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# data loading and train-test split

df = pd.read_csv('bath_EMY.csv')

# convert "Time" to datetime
df.Time = pd.to_datetime(df.Time)

# generate new features from data
df['year'] = df.Time.dt.year
df['month'] = df.Time.dt.month
df['day'] = df.Time.dt.day
df['hour'] = df.Time.dt.hour

X = df.drop(['Time', 'avg_value'], axis=1)

y = df.avg_value.values

# standardize all the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)



#create a list for the rest models to use KFOLD cv

models = []
models.append(('Linear_Reg', LinearRegression()))
models.append(('Ridge_Reg', Ridge()))
models.append(('Lasso_Reg', Lasso()))
models.append(('Random_Forest_Reg', RandomForestRegressor()))
models.append(('SVR_Reg', SVR()))
models.append(('Decision_Tree_Reg', DecisionTreeRegressor()))
models.append(('MLP_Reg', MLPRegressor()))

# evaluate each model in turn
R2_scores_list = []
results = []
names = []
for name, model in models:
    # create a KFold object with 10 splits
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=folds) #scoring = 'neg_root_mean_squared_error' gia to rmse
    results.append(scores)
    names.append(name)
    R2_scores_list.append(scores.mean())
    print('%s: %f' % (name, scores.mean()))

print("the final R2_score list of all the models is: ", R2_scores_list)

#convert list to series
R2_scores_list = pd.Series(R2_scores_list)

#plot the series to visually compare the R2_score of all models
ax = R2_scores_list.plot(kind='bar', figsize=(10, 5), width=0.50, fontsize=15, color=['blue', 'green', 'red', 'gray', 'black', 'orange', 'brown'])
ax.set_xlabel("Models", fontsize=10)
ax.set_ylabel("R2 Score", fontsize=10)
ax.set_title('comparison of R2 score of each model', fontsize=10)
ax.set_xticklabels([ 'Linear_Reg', 'Ridge', 'Lasso', 'Random_Forest', 'SVR', 'DecisionTree', 'MLP'], rotation=45)
plt.show()