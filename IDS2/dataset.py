from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# The 11 essential temporal and volumetric features for CIC-IoT2023 + the label
columns_to_keep = [
    'IAT', 'flow_duration', 'Rate', 'Srate', 'Header_Length', 
    'syn_flag_number', 'rst_count', 'Min', 'Max', 'AVG', 'Std', 'label'
]

# 1. Load the dataset
# Note: You might need to change 'CICIOT2023.csv' to the exact name of your downloaded file
df_raw = pd.read_csv("CICIOT2023.csv")

# 2. Filter down to the 10,000 important rows for the RL agent
sequential_attacks = ['DictionaryBruteForce', 'PortScan', 'OSScan', 'HostDiscovery']
direct_attacks = ['DDoS-ICMP-Flood', 'DDoS-UDP-Flood']

# Sampling to ensure data balance (Total: 10,000 rows)
df_seq = df_raw[df_raw['label'].isin(sequential_attacks)].sample(n=4000, random_state=42)
df_dir = df_raw[df_raw['label'].isin(direct_attacks)].sample(n=3000, random_state=42)
df_ben = df_raw[df_raw['label'] == 'Benign'].sample(n=3000, random_state=42)

# Combine and shuffle, then filter columns
df = pd.concat([df_seq, df_dir, df_ben]).sample(frac=1).reset_index(drop=True)
df = df[columns_to_keep]

pd.set_option('display.max_column', None)

# 3. Binarizing the target value (0 for Benign, 1 for Attack)
df['label'] = df['label'].apply(lambda x : 0 if x == "Benign" else 1)

# Note: CIC-IoT2023 core features are already numeric, so we skip the pd.get_dummies() step 
# that was required for NSL-KDD's text-based protocols.

# 4. Splitting feature X and target y
scaler = StandardScaler()
X = df.drop(columns=['label'])
y = df['label']

""" Below line calculates the mean and standard deviation of your data, and then scales all the values so they have a 
    mean of 0 and a standard deviation of 1. This helps neural networks learn much faster and more stably. So that the neural network does not get 
    distracted with large numbers it performs some operation it 
"""
X_scaled = scaler.fit_transform(X)

# converts into a mathematical list of numbers
y = np.array(y)

# 5. Creating sequences of inputs for LSTM
def create_sequences(X, y, seq_length=10):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1]) # label of last step 

    return np.array(X_seq), np.array(y_seq)

# Now applying here
X_seq, y_seq = create_sequences(X_scaled, y, seq_length=10)

# 6. Train and test split
# shuffle=False is strictly maintained so we don't break the time-series flow of the packets
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False 
)

# To see the first sequence of packets (Uncomment to debug)
# first_sequence = X_train[0]
# readable_sequence = pd.DataFrame(first_sequence, columns=X.columns)
# print(readable_sequence)
# print("Label for the first sequence:", y_train[0])

# 7. Saving the processed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)   # Added saving for test data
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)   # Added saving for test data

print(f"Data preprocessing complete! Training shape: {X_train.shape}, Testing shape: {X_test.shape}")