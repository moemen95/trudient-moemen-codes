import numpy as np # numpy library used mainly for linear algebra
import pandas as pd # pandas library used to read and manipulate tabular data
import lightgbm as lgb
import gc
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.utils.extmath import density


# define random seed for reproducibility we will use it in other instances in the code
seed = 17
np.random.seed(seed)

# load our data
print("loading our dataset please wait..")
root_dir = "datasets/reducing-commercial-aviation-fatalities/" # the root directory of the dataset
df_train = pd.read_csv(root_dir + "train.csv") # load training data
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

print("data processed")

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

df_train = reduce_mem_usage(df_train)
print("data optimized")

print("preparing")
# remove the experiment column and get the numpys
x_train, y_train = df_train.drop(["time", 
                                  "experiment",
                                  "event"], axis=1).values.astype("float32"), df_train["event"]
print(" Normalizing & Splitting")
# split our dataset to train/dev split with 20% validaton splitting
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=999, shuffle=True)
print("x_train shape: {}, dtype: {}".format(x_train.shape, x_train.dtype))
print("y_train shape: {}, dtype: {}".format(y_train.shape, y_train.dtype))
print("x_val shape: {}, dtype: {}".format(x_val.shape, x_val.dtype))
print("y_val shape: {}, dtype: {}".format(y_val.shape, y_val.dtype))

names = ["Logistic Regression",
         #"Neural Net",
         "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest",  "AdaBoost",
         "Naive Bayes", "LDA", "QDA",  "Nearest Neighbors"]

classifiers = [
    LogisticRegression(multi_class = 'multinomial', solver='saga', penalty='l2', max_iter=500, n_jobs=4),
    #MLPClassifier(max_iter=100),
    SVC(kernel="linear"),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=500),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(100000)]

df_metrics = pd.DataFrame(columns=['model', 'log_loss', 'accuracy'])

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print('_' * 80)
    print("Training: {}".format(name))
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(x_val)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    
    prob = clf.predict_proba(x_val)
    loss = log_loss(y_val, prob, labels=clf.classes_)
    print("Log loss: {}".format(loss))

    score = metrics.accuracy_score(y_val, pred)
    print("accuracy:   %0.3f" % score)
    
    df_metrics.loc[name] = [clf, loss, score]

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
    
    print("classification report:")
    print(metrics.classification_report(y_val, pred,
                                        target_names=["A","B","C","D"]))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_val, pred))

    print()
    clf_descr = str(clf).split('(')[0]

print(df_metrics)
df_metrics.to_csv("df_metrics.csv")

# loading the test data
#df_test = pd.read_csv(root_dir + "test.csv") # load testing data
#df_test['experiment'] = df_test['experiment'].apply(lambda x: dic_exp[x])
#df_test['experiment'] = df_test['experiment'].astype('int8')
#df_test = reduce_mem_usage(df_test)
#x_test = df_test.drop(["experiment", "id"], axis=1).values.astype("float32")
