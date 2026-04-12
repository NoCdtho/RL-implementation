from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment'
    ,'urgent'
    ,'hot'
    ,'num_failed_logins'
    ,'logged_in'
    ,'num_compromised'
    ,'root_shell'
    ,'su_attempted'
    ,'num_root'
    ,'num_file_creations'
    ,'num_shells'
    ,'num_access_files'
    ,'num_outbound_cmds'
    ,'is_host_login'
    ,'is_guest_login'
    ,'count'
    ,'srv_count'
    ,'serror_rate'
    ,'srv_serror_rate'
    ,'rerror_rate'
    ,'srv_rerror_rate'
    ,'same_srv_rate'
    ,'diff_srv_rate'
    ,'srv_diff_host_rate'
    ,'dst_host_count'
    ,'dst_host_srv_count'
    ,'dst_host_same_srv_rate'
    ,'dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate'
    ,'dst_host_srv_diff_host_rate'
    ,'dst_host_serror_rate'
    ,'dst_host_srv_serror_rate'
    ,'dst_host_rerror_rate'
    ,'dst_host_srv_rerror_rate'
    ,'attack'
    ,'level'
]

df = pd.read_csv("Ktrain.txt", names=columns)

df = df.drop(columns=["level"])
pd.set_option('display.max_column', None)

# Binarizing the target value
df['attack'] = df['attack'].apply(lambda x : 0 if x == "normal" else 1)

# performing one hot encoding 
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

# splitting feature X and target y
scaler = StandardScaler()
X = df.drop(columns=['attack'])
y = df['attack']

""" Below line calculates the mean and standard deviation of your data, and then scales all the values so they have a 
    mean of 0 and a standard deviation of 1. This helps neural networks learn much faster and more stably. So that the neural network does not get 
    distracted with large numbers it performs some operation it 
"""
X_scaled = scaler.fit_transform(X)

#converts into a mathematical list of numbers
y = np.array(y)

# Creating  sequnces of inputs for LSTM
def create_sequences(X, y, seq_length=10):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1]) #label of last step 

    return np.array(X_seq), np.array(y_seq)

# Now applying here
X_seq, y_seq = create_sequences(X_scaled, y)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, train_size=0.2, shuffle=False
)

# To see the first 10 sequence of packets
# first_sequence = X_train[0]
# readable_sequence = pd.DataFrame(first_sequence, columns=X.columns)
# print(readable_sequence)
# print("Label for the first sequnce:", y_train[0])

# saving the processed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
