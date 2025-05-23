import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load and preprocess
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income']
df = pd.read_csv(url, names=cols, skipinitialspace=True, na_values='?').dropna()
df = df[df['sex'].isin(['Male', 'Female'])]
df['income'] = df['income'].map({'>50K':1, '<=50K':0})
df['sex'] = df['sex'].map({'Male':1, 'Female':0})

# Bias check
X = pd.get_dummies(df.drop('income', axis=1), drop_first=True)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = LogisticRegression(max_iter=200).fit(X_train, y_train)
pred = model.predict(X_test)
test_sex = X_test['sex']
print("Male acc:", accuracy_score(y_test[test_sex==1], pred[test_sex==1]))
print("Female acc:", accuracy_score(y_test[test_sex==0], pred[test_sex==0]))

# Simple mitigation: oversample females
df_min = df[df['sex'] == 0]
df_maj = df[df['sex'] == 1]
df_up = pd.concat([df_maj, resample(df_min, replace=True, n_samples=len(df_maj), random_state=1)])

X2 = pd.get_dummies(df_up.drop('income', axis=1), drop_first=True)
y2 = df_up['income']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=1)
model2 = LogisticRegression(max_iter=200).fit(X_train2, y_train2)
pred2 = model2.predict(X_test2)
test_sex2 = X_test2['sex_1']

print("Post-mitigation Male acc:", accuracy_score(y_test2[test_sex2==1], pred2[test_sex2==1]))
print("Post-mitigation Female acc:", accuracy_score(y_test2[test_sex2==0], pred2[test_sex2==0]))