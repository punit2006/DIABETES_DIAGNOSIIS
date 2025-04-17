import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#Total no. of columns in the dataset.
DIABETES_df.columns

#Information about the dataset.
DIABETES_df.info()

DIABETES_df.describe()
DIABETES_df.isnull()
DIABETES_df.isnull().sum()

DIABETES_df_copy = DIABETES_df.copy(deep = True)
DIABETES_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = DIABETES_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0,np.NaN)

#Showing the count of NANs
print(DIABETES_df_copy.isnull().sum())

#Plotting the data distribution plots on HISTOGRAM
p = DIABETES_df.hist(figsize = (20,20))

#Aiming to impute NAN values for the columns in accordance with their distribution.
DIABETES_df_copy['Glucose'].fillna(DIABETES_df_copy['Glucose'].mean(), inplace = True)
DIABETES_df_copy['BloodPressure'].fillna(DIABETES_df_copy['BloodPressure'].mean(), inplace = True)
DIABETES_df_copy['SkinThickness'].fillna(DIABETES_df_copy['SkinThickness'].median(), inplace = True)
DIABETES_df_copy['Insulin'].fillna(DIABETES_df_copy['Insulin'].median(), inplace = True)
DIABETES_df_copy['BMI'].fillna(DIABETES_df_copy['BMI'].median(), inplace = True)
DIABETES_df_copy

#Plotting the distribution after removing the NAN values.
p = DIABETES_df_copy.hist(figsize = (20,20))

#Checking the balance of the data by plotting the count of the outcomes by their values.
color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = DIABETES_df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(DIABETES_df.Outcome.value_counts())
p = DIABETES_df.Outcome.value_counts().plot(kind = "bar")

X = DIABETES_df.drop(columns = 'Outcome', axis = 1)
Y = DIABETES_df['Outcome']
print(X)

DIABETES_df_copy.head()
y = DIABETES_df_copy.Outcome
y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 42, stratify = y)
print(X.shape, X_train.shape, X_test.shape)

train_df, test_df = train_test_split(DIABETES_df, test_size=0.1, random_state=42)
train_df_labels = train_df["Outcome"].copy()                           
train_df= train_df.drop("Outcome", axis=1) 
num_pipeline = Pipeline([('std_scaler', StandardScaler()), ])

train_prepared = num_pipeline.fit_transform(train_df)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_prepared, train_df_labels)
cross_val_score(sgd_clf, train_prepared, train_df_labels, cv= 3, scoring='accuracy')

prediction = sgd_clf.predict(train_prepared)
print("SGD Accuracy of Classifier: ", sgd_clf.score(train_prepared, train_df_labels))

model = LogisticRegression()
model.fit(train_prepared, train_df_labels)
from sklearn.model_selection import cross_val_score
cross_val_score(model, train_prepared, train_df_labels, cv= 3, scoring='accuracy')

prediction = model.predict(train_prepared)
print("LR Accuracy of Classifier: ", model.score(train_prepared, train_df_labels))

poly_kernel_svm_clf = Pipeline([ ("scaler", StandardScaler()), 
                                ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))]) 

poly_kernel_svm_clf.fit(train_prepared, train_df_labels)
print("SVM Accuracy of Classifier: ", poly_kernel_svm_clf.score(train_prepared, train_df_labels))

test_scores = []
train_scores = []

for i in range(1, 15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

train_scores
test_scores
G = train_scores
G

max_train_score = max(G)
train_scores_ind = [i for i, v in enumerate(G) if v == max_train_score]
print('Max train score {} % and k ={}'.format(max_train_score*100,list(map(lambda x: x+1,train_scores_ind))))

max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k ={}'.format(max_test_score*100,list(map(lambda x: x+1,test_scores_ind))))

knn = KNeighborsClassifier(11)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

y_pred = knn.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu', fmt = 'g')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

param_grid = {'n_neighbors' :np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid,cv = 5)
knn_cv.fit(X,y)

print("Best Score: " + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

from sklearn.model_selection import cross_val_predict 
y_train_pred = cross_val_predict(poly_kernel_svm_clf, train_prepared, train_df_labels, cv=3)

confusion_matrix(train_df_labels, y_train_pred)

from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision Score:',precision_score(train_df_labels, y_train_pred))
print('Recall Score:',recall_score(train_df_labels, y_train_pred))
print('F1 Score:',f1_score(train_df_labels, y_train_pred))

import numpy as np
from sklearn.linear_model import LogisticRegression 

try:
    a = int(input("Enter No. of Pregnancies: "))
    b = int(input("Enter Glucose Level: "))
    c = int(input("Enter Blood Pressure: "))
    d = int(input("Enter Skin Thickness: "))
    e = int(input("Enter Insulin Level: "))
    f = float(input("Enter BMI: "))
    g = float(input("Enter Diabetes Pedigree Function: "))
    h = int(input("Enter Age: "))

    input_data = (a, b, c, d, e, f, g, h)
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    print("Reshaped Input Data:", input_data_reshaped)

    
    print("\nPrediction Result:", prediction)

    if prediction[0] == 0:
        print('The person is not diabetic')
    else: 
        print('The person is diabetic')
except ValueError as e:
    print(f"Input Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/punit/Downloads/DIABETESDIAGNOSIS/DIABETES.csv')

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Matrix Heatmap")

plt.show()


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df = pd.read_csv('C:/Users/punit/Downloads/DIABETESDIAGNOSIS/DIABETES.csv')

features = df.columns[:-1]

min_max_scaler = MinMaxScaler()
df_min_max = df.copy()
df_min_max[features] = min_max_scaler.fit_transform(df[features])

standard_scaler = StandardScaler()
df_standard = df.copy()
df_standard[features] = standard_scaler.fit_transform(df[features])

robust_scaler = RobustScaler()
df_robust = df.copy()
df_robust[features] = robust_scaler.fit_transform(df[features])

print("Min-Max Scaled Data Sample:\n", df_min_max.head())
print("\nZ-Score Standardized Data Sample:\n", df_standard.head())
print("\nRobust Scaled Data Sample:\n", df_robust.head())

df_min_max.to_csv("min_max_scaled.csv", index=False)
df_standard.to_csv("z_score_standardized.csv", index=False)
df_robust.to_csv("robust_scaled.csv", index=False)


import pandas as pd

df = pd.read_csv('C:/Users/punit/Downloads/DIABETESDIAGNOSIS/DIABETES.csv')

summary_stats = pd.DataFrame({
    "Mean": df.mean(),
    "Median": df.median(),
    "Mode": df.mode().iloc[0], 
    "Variance": df.var(),
    "Standard Deviation": df.std(),
    "Min": df.min(),
    "Max": df.max()
})

print(summary_stats)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/punit/Downloads/DIABETESDIAGNOSIS/DIABETES.csv')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, palette="Set2")

plt.title("Box Plot of Features in the Dataset", fontsize=14)
plt.xticks(rotation=45) 
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/punit/Downloads/DIABETESDIAGNOSIS/DIABETES.csv')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Glucose"], y=df["BMI"], hue=df["Outcome"], palette="coolwarm", alpha=1)

plt.xlabel("Glucose Level")
plt.ylabel("BMI")
plt.title("Scatter Plot of Glucose vs BMI (Caolored by Outcome)")

plt.legend(title="Diabetes Outcome", labels=["No Diabetes", "Diabetes"])
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/punit/Downloads/DIABETESDIAGNOSIS/DIABETES.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(12, 10))
sns.pairplot(df, hue="Outcome", palette="coolwarm", diag_kind="kde")

plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the dataset
df = pd.read_csv('/content/DIABETES.csv')
df = df[df["Insulin"] >= 16]
X = df.drop(['Outcome', 'Insulin'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "SGD Classifier": SGDClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(),
    "SVM (Poly Kernel)": SVC(kernel='poly', degree=3, coef0=1, C=5),
    "KNN (k=11)": KNeighborsClassifier(n_neighbors=11)
}

# Store results
results = []

# Evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

# Create results DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\nModel Comparison:\n")
print(results_df)

# Plotting results
plt.figure(figsize=(12, 6))
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind='bar', figsize=(12, 6), ylim=(0.6, 1.0), colormap='coolwarm')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define Base Models
base_models = [
    ('sgd', SGDClassifier(loss='log_loss', random_state=42)),  # Use log-loss to enable probabilities
    ('lr', LogisticRegression()),
    ('svm_poly', SVC(kernel='poly', degree=3, coef0=1, C=5, probability=True)),
    ('knn', KNeighborsClassifier(n_neighbors=11))
]

# Define Meta-Model
meta_model = LogisticRegression()

# Create Stacking Classifier with 'auto' stack method
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, stack_method='auto')

# Train
stacking_clf.fit(X_train, y_train)

# Predict
y_pred = stacking_clf.predict(X_test)

# Evaluate
print(f"Stacking Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Stacking Model Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define base models
model1 = ('sgd', SGDClassifier(loss='log_loss', random_state=42))  # log_loss allows probability estimates
model2 = ('lr', LogisticRegression())
model3 = ('svm_poly', SVC(kernel='poly', degree=3, coef0=1, C=5, probability=True))
model4 = ('knn', KNeighborsClassifier(n_neighbors=11))

# Create VotingClassifier
# Use 'soft' voting if all models support predict_proba (which they do now with SGDClassifier using 'log_loss')
voting_clf = VotingClassifier(
    estimators=[model1, model2, model3, model4],
    voting='soft'  # Change to 'hard' if you want majority class vote instead
)

# Fit the model
voting_clf.fit(X_train, y_train)

# Predict
y_pred = voting_clf.predict(X_test)

# Evaluate
print(f"Voting Classifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Voting Classifier Classification Report:\n", classification_report(y_test, y_pred))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('/content/DIABETES.csv')

# Filter out unreliable insulin values (e.g., less than 16)
df = df[df["Insulin"] >= 16]

# Features and target
X = df.drop(["Outcome", "Insulin"], axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = [
    ('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier(n_neighbors=11)),
    ('svm', SVC(kernel='poly', degree=3, coef0=1, C=5, probability=True))
]

# Stacking Classifier
stacking = StackingClassifier(estimators=models, final_estimator=LogisticRegression())
stacking.fit(X_train_scaled, y_train)
y_pred_stack = stacking.predict(X_test_scaled)

# Voting Classifier
voting = VotingClassifier(estimators=models, voting='soft')
voting.fit(X_train_scaled, y_train)
y_pred_vote = voting.predict(X_test_scaled)

# Evaluate function
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

# Collect results
results = [
    evaluate_model("Stacking Classifier", y_test, y_pred_stack),
    evaluate_model("Voting Classifier", y_test, y_pred_vote)
]

# Display
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="F1-Score", ascending=False))
