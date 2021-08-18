import pandas as pd

import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import cross_val_predict


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# data loading and train-test split

df = pd.read_csv('final_avg_bath_temp.csv')

#convert "Time" to datetime
df.Time = pd.to_datetime(df.Time)

#generate new features from data
df['year'] = df.Time.dt.year
df['month'] = df.Time.dt.month
df['day'] = df.Time.dt.day
df['hour'] = df.Time.dt.hour

print(df.head())

X = df.drop(['Time', 'avg_value'], axis=1)

y = df.avg_value.values

#standardize all the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

#OBJECTIVE TO TUNE THE HYPERPARAMETERS FOR RIDGE REGRESSION USING OPTUNA

def objective_ridge(trial):


      # the parameter that needs to be optimized
      ridge_alpha = trial.suggest_uniform('ridge_alpha', 0.0, 2.0)
      Ridge_Reg = Ridge(alpha=ridge_alpha)

      # model training and evaluation of 10 folds
      folds = KFold(n_splits=10, shuffle=True, random_state=1)
      scores = cross_val_score(Ridge_Reg, X_train, y_train, scoring='r2', cv=folds)


      return scores.mean()

#create the study trial
study1 = optuna.create_study(direction='maximize')
study1.optimize(objective_ridge, n_trials=50)



# To get the dictionary of parameter name and parameter values:
print("Return a dictionary of parameter name and parameter values:", study1.best_params)
# To get the best observed value of the objective function:
print("Return the best observed value of the RIDGE objective function:", study1.best_value)

#create a list to store the R2_scores for the models that have been optimized with optuna
R2_scores_list = []
R2_scores_list.append(study1.best_value)


#OBJECTIVE TO TUNE THE HYPERPARAMETER FOR LASSO REGRESSION USING OPTUNA
def objective_lasso(trial):
    # the parameter that needs to be optimized
    lasso_alpha = trial.suggest_uniform('lasso_alpha', 0.0, 2.0)
    Lasso_Reg = Lasso(alpha=lasso_alpha)

    # model training and evaluation of 10 folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(Lasso_Reg, X_train, y_train, scoring='r2', cv=folds)

    return scores.mean()


# create the study trial
study2 = optuna.create_study(direction='maximize')
study2.optimize(objective_lasso, n_trials=50)

# To get the dictionary of parameter name and parameter values:
print("Return a dictionary of parameter name and parameter values:", study2.best_params)
# To get the best observed value of the objective function:
print("Return the best observed value of the LASSO objective function:", study2.best_value)

#store the R2_score for the 2nd study
R2_scores_list.append(study2.best_value)
print(R2_scores_list)


#OBJECTIVE TO TUNE THE HYPERPARAMETERS FOR RANDOMFOREST REGRESSION USING OPTUNA
def objective_Random_Forest(trial):
    # the parameters that needs to be optimized
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 50)
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 10, log=True)
    RF_Reg = RandomForestRegressor(max_depth=rf_max_depth, n_estimators=rf_n_estimators, random_state=0)

    # model training and evaluation of 10 folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(RF_Reg, X_train, y_train, scoring='r2', cv=folds)

    return scores.mean()


# create the study trial
study3 = optuna.create_study(direction='maximize')
study3.optimize(objective_Random_Forest, n_trials=50)

# To get the dictionary of parameter name and parameter values:
print("Return a dictionary of parameter name and parameter values:", study3.best_params)
# To get the best observed value of the objective function:
print("Return the best observed value of the RANDOM FOREST objective function:", study3.best_value)

#store the R2_score for the 3rd study
R2_scores_list.append(study3.best_value)

print(R2_scores_list)


#OBJECTIVE TO TUNE THE HYPERPARAMETERS FOR SVR REGRESSION USING OPTUNA
def objective_SVR(trial):
    # the parameters that needs to be optimized
    svr_c = trial.suggest_uniform('svr_c', 0.0, 10)
    svr_epsilon = trial.suggest_uniform("svr_epsilon", 0.0, 2.0)
    SVR_Reg = SVR(C=svr_c, epsilon=svr_epsilon)

    # model training and evaluation of 10 folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(SVR_Reg, X_train, y_train, scoring='r2', cv=folds)

    return scores.mean()


# create the study trial
study4 = optuna.create_study(direction='maximize')
study4.optimize(objective_SVR, n_trials=50)

# To get the dictionary of parameter name and parameter values:
print("Return a dictionary of parameter name and parameter values:", study4.best_params)
# To get the best observed value of the objective function:
print("Return the best observed value of the SVR objective function:", study4.best_value)

#store the R2_score for the 4th study
R2_scores_list.append(study4.best_value)

print(R2_scores_list)



#OBJECTIVE TO TUNE THE HYPERPARAMETERS FOR DECISION TREE REGRESSION USING OPTUNA
def objective_DECISION_TREE(trial):
    # the parameters that needs to be optimized
    dt_max_depth = trial.suggest_int("dt_max_depth", 2, 10)
    dt_min_samples_split = trial.suggest_int("dt_min_samples_split", 1, 40)
    DT_Reg = DecisionTreeRegressor(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split)

    # model training and evaluation of 10 folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(DT_Reg, X_train, y_train, scoring='r2', cv=folds)

    return scores.mean()


# create the study trial
study5 = optuna.create_study(direction='maximize')
study5.optimize(objective_DECISION_TREE, n_trials=50)

# To get the dictionary of parameter name and parameter values:
print("Return a dictionary of parameter name and parameter values:", study5.best_params)
# To get the best observed value of the objective function:
print("Return the best observed value of the DECISION TREE objective function:", study5.best_value)

#store the R2_score for the 5th study
R2_scores_list.append(study5.best_value)

print(R2_scores_list)



#OBJECTIVE TO TUNE THE HYPERPARAMETERS FOR DECISION TREE REGRESSION USING OPTUNA
def objective_MLP(trial):
    # the parameters that needs to be optimized
    mlp_max_iter = trial.suggest_int("mlp_max_iter", 10, 500)
    mlp_alpha = trial.suggest_uniform("mlp_alpha", 0.0, 2.0)
    MLP_Reg = MLPRegressor(alpha=mlp_alpha, max_iter=mlp_max_iter)

    # model training and evaluation of 10 folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(MLP_Reg, X_train, y_train, scoring='r2', cv=folds)

    return scores.mean()


# create the study trial
study6 = optuna.create_study(direction='maximize')
study6.optimize(objective_MLP, n_trials=50)

# To get the dictionary of parameter name and parameter values:
print("Return a dictionary of parameter name and parameter values:", study6.best_params)
# To get the best observed value of the objective function:
print("Return the best observed value of the MLP objective function:", study6.best_value)

#store the R2_score for the 6th study
R2_scores_list.append(study6.best_value)

print(R2_scores_list)


#create a list for the rest models to use KFOLD cv (can add more models that dont need hyperparameter tuning)

models = []
models.append(('LR', LinearRegression()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    # create a KFold object with 10 splits
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=folds)
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
ax.set_xticklabels(['Ridge', 'Lasso', 'Random_Forest', 'SVR', 'DecisionTree', 'MLP', 'Linear_Reg'], rotation=45)
plt.show()
