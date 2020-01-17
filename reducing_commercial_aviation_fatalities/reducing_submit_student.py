import numpy as np # numpy library used mainly for linear algebra
import pandas as pd # pandas library used to read and manipulate tabular data
import lightgbm as lgb
import gc

from reducing_utils import reduce_mem_usage

# define random seed for reproducibility we will use it in other instances in the code
seed = 17
np.random.seed(seed)

# load our data
print("loading our dataset please wait..")
root_dir = "../datasets/reducing-commercial-aviation-fatalities/" # the root directory of the dataset
df_train = pd.read_csv(root_dir + "train.csv") # load training data
df_test = pd.read_csv(root_dir + "test.csv") # load testing data
print("data loaded")

# categorize and map with intergers
dic_exp = {'CA': 2, 'DA': 3, 'SS': 1, 'LOFT': 4}
# A = baseline, B = SS, C = CA, D = DA
dic_event = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
labels_exp = {v: k for k, v in dic_exp.items()}
labels_event = {v: k for k, v in dic_event.items()}
df_train["event"] = df_train["event"].apply(lambda x: dic_event[x])
df_train["event"] = df_train["event"].astype('int8')
df_train['experiment'] = df_train['experiment'].apply(lambda x: dic_exp[x])
df_train['experiment'] = df_train['experiment'].astype('int8')
df_test['experiment'] = df_test['experiment'].apply(lambda x: dic_exp[x])
df_test['experiment'] = df_test['experiment'].astype('int8')
print("data processed")

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

print("data optimized")

print("preparing")
# remove the experiment column and get the numpys
df_train, y_df_train = df_train.drop(["experiment", "time", "event"], axis=1).values.astype("float32"), df_train["event"]
df_test = df_test.drop(["experiment", "time", "id"], axis=1)

from sklearn.model_selection import train_test_split

print("Splitting")
# split our dataset to train/dev split with 20% validaton splitting
X_train, X_val, Y_train, Y_val = train_test_split(df_train, y_df_train, test_size=0.1, random_state=999, shuffle=False)

del df_train
del y_df_train
gc.collect()

# Creating our datasets
train_lgb = lgb.Dataset(X_train, label=Y_train, categorical_feature=[1])
val_lgb = lgb.Dataset(X_val, label=Y_val, categorical_feature=[1])

# creating the hyperparameters
params = {
        "objective" : "multiclass", 
        "metric" : "multi_error", 
        'num_class':4,
        "num_leaves" : 255, 
        "learning_rate" : 0.1, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':100, 
        'min_split_gain':0.00019
}

model = lgb.train(params, train_set = train_lgb,
                    num_boost_round=200,
                    early_stopping_rounds=300,
                    verbose_eval=100, 
                    valid_sets=[train_lgb,val_lgb]
                  )

print("predicting")
y_pred = model.predict(df_test, num_iteration=model.best_iteration)
print("Concatenating")
df_sub = pd.DataFrame(np.concatenate((np.arange(len(df_test))[:, np.newaxis], y_pred), axis=1), 
                      columns=['id', 'A', 'B', 'C', 'D'])
df_sub['id'] = df_sub['id'].astype(int)

print(df_sub)
df_sub.to_csv(root_dir + "submission_lgb_final.csv", index=False)

# confusion matrix
from sklearn.metrics import log_loss, confusion_matrix

pred_val = model.predict(X_val, num_iteration=model.best_iteration)
print("Log loss on validation data :", round(log_loss(np.array(Y_val), pred_val), 3))
conf_mat_val = confusion_matrix(np.argmax(pred_val, axis=1), Y_val)
plot_confusion_matrix(conf_mat_val, ["A", "B", "C", "D"], title='Confusion matrix on Validation data', normalize=True)