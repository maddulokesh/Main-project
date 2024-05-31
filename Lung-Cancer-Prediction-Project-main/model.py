# Importing Libraries:
import pandas as pd
import numpy as np
import pickle

# Reading Dataset:
dataset = pd.read_csv("survey lung cancer.csv")
label = dataset["LUNG_CANCER"]
label = np.where(label == 'YES',1,0)


# Replacing Categorical Values with Numericals
dataset.GENDER = dataset.GENDER.map({"M":1,"F":2})
dataset.LUNG_CANCER = dataset.LUNG_CANCER.map({"YES":1,"NO":2})


"""# Handling Missing Values:
features = ['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH','SWALLOWING_DIFFICULTY','CHEST_PAIN','LUNG_CANCER']
for feature in features:
    dataset[feature] = dataset[feature].fillna(dataset[feature].median())"""


# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# After feature importance:
X = dataset[['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH','SWALLOWING_DIFFICULTY','CHEST_PAIN','LUNG_CANCER']]

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(dataset,label)

# RandomForestClassifier:
from sklearn.linear_model import LogisticRegression
Model1 = LogisticRegression(solver='lbfgs', max_iter=10000)
logi= Model1.fit(X_train,Y_train)

# Creating a pickle file for the classifier
filename = 'lung.pkl'
pickle.dump(logi, open(filename, 'wb'))