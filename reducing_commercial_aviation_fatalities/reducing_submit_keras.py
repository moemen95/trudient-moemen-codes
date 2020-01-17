import numpy as np # numpy library used mainly for linear algebra
import pandas as pd # pandas library used to read and manipulate tabular data
import lightgbm as lgb
import gc
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn import metrics

from keras.models import Model
from keras.layers import Input, Dense, Dropout, ReLU
from keras.optimizers import Adam
from keras.regularizers import l2

# define random seed for reproducibility we will use it in other instances in the code
seed = 17
np.random.seed(seed)

# load our data
print("loading our dataset please wait..")
root_dir = "../datasets/reducing-commercial-aviation-fatalities/" # the root directory of the dataset
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

del df_train
gc.collect()

print(" Normalizing & Splitting")
# split our dataset to train/dev split with 20% validaton splitting
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=999, shuffle=False)
print("x_train shape: {}, dtype: {}".format(x_train.shape, x_train.dtype))
print("y_train shape: {}, dtype: {}".format(y_train.shape, y_train.dtype))
print("x_val shape: {}, dtype: {}".format(x_val.shape, x_val.dtype))
print("y_val shape: {}, dtype: {}".format(y_val.shape, y_val.dtype))

input_features = Input(shape=(x_train.shape[1],)) 
dense_1_out = Dense(256, activation="relu",
                    use_bias=True, 
                    kernel_initializer='glorot_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=l2(1e-5))(input_features)
dense_1_drop = Dropout(0.1)(dense_1_out)
dense_2_out = Dense(64, activation="relu",
                    use_bias=True, 
                    kernel_initializer='glorot_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=l2(1e-5))(dense_1_drop)
dense_2_drop = Dropout(0.05)(dense_2_out)
logits = Dense(4, activation='softmax')(dense_2_drop) # 4 classes

# Finalizing the model by specifying the inputs and the outputs
print("building model")
model = Model(inputs=input_features, outputs=logits)

# Let's define our hyperparameters
learning_rate = 0.01
batch_size = 64
num_epochs = 20

optimizer = Adam(lr=learning_rate)

print("Compiling")
model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                    validation_data=(x_val,y_val), verbose=2, shuffle=True)

del x_train
del y_train
gc.collect()

# loading the test data
print("loading test data")
df_test = pd.read_csv(root_dir + "test.csv") # load testing data
df_test = reduce_mem_usage(df_test)
x_test = df_test.drop(["time", "experiment", "id"], axis=1).values.astype("float32")
x_test = scaler.transform(x_test)

del df_test
gc.collect()

print("Predicting")
# get the prediction of the model on the test data
prediction = model.predict(x_test)
print("prediction shape: {},  dtype: {}".format(prediction.shape, prediction.dtype))

print("Concatenating")
df_sub = pd.DataFrame({"A" : prediction[:,0], 
                       "B" : prediction[:,1], 
                       "C" : prediction[:,2], 
                       "D" : prediction[:,3]})
print(df_sub)
df_sub.to_csv(root_dir + "submission_MLP_final.csv", index_label='id')

prob = model.predict(x_val)
pred = (prob > 0.5).astype(int).ravel()

loss = log_loss(y_val, prob, labels=[0,1,2,3])
print("Log loss: {}".format(loss))

score = metrics.accuracy_score(y_val, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_val, pred,
                                    target_names=["A","B","C","D"]))

print("confusion matrix:")
print(metrics.confusion_matrix(y_val, pred))