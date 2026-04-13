import numpy as np
import pandas as pd
from dataSet import X_scaled, X, y

X_df = pd.DataFrame(X_scaled, columns=X.columns)
y_df = pd.DataFrame(y, columns=["attacks"])

X_df.to_excel("X.xlsx", index=False)
y_df.to_excel("y.xlsx", index=False)