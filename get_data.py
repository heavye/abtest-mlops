from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import os

seed = 42
# Generate data
X, y = make_classification(n_samples = 1000, random_state=seed)

# Make a train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=seed)

# Save it
if not os.path.isdir("data"):
    os.mkdir("data")
np.savetxt("data/AdSmartABdata.csv",X_train)
np.savetxt("data/AdSmartABdata.csv",X_test)
np.savetxt("data/AdSmartABdata.csv",y_train)
np.savetxt("data/AdSmartABdata.csv",y_test)


