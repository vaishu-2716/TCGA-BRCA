# Import necessary libraries

import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Import machine learning models

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Load dataset
data = pd.read_csv('somatic_mutation.csv')

# Feature selection
X = data[[col for col in data.columns if col.startswith('mu_')]]
y = data['vital.status']

selector = SelectKBest(score_func=f_classif, k=50)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
warnings.filterwarnings("ignore")

# Model training and evaluation
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(f"KNN Accuracy: {knn_acc:.2%}")

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.2%}")

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"Decision Tree Accuracy: {dt_acc:.2%}")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression Accuracy: {lr_acc:.2%}")

ann = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
ann.fit(X_train, y_train)
ann_acc = accuracy_score(y_test, ann.predict(X_test))
print(f"MLP Neural Network Accuracy: {ann_acc:.2%}")

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test))
print(f"Naive Bayes Accuracy: {nb_acc:.2%}")

et = ExtraTreesClassifier(n_estimators=200, random_state=42)
et.fit(X_train, y_train)
et_acc = accuracy_score(y_test, et.predict(X_test))
print(f"Extra Trees Accuracy: {et_acc:.2%}")

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"Gradient Boosting Accuracy: {gb_acc:.2%}")

ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test))
print(f"AdaBoost Accuracy: {ada_acc:.2%}")

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
print(f"XGBoost Accuracy: {xgb_acc:.2%}")

lgb = LGBMClassifier(random_state=42, verbose=-1)
lgb.fit(X_train, y_train)
lgb_acc = accuracy_score(y_test, lgb.predict(X_test))
print(f"LightGBM Accuracy: {lgb_acc:.2%}")

ridge = RidgeClassifier()
ridge.fit(X_train, y_train)
ridge_acc = accuracy_score(y_test, ridge.predict(X_test))
print(f"Ridge Classifier Accuracy: {ridge_acc:.2%}")

lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
lasso.fit(X_train, y_train)
lasso_acc = accuracy_score(y_test, lasso.predict(X_test))
print(f"Lasso Regression Accuracy: {lasso_acc:.2%}")

sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
sgd_acc = accuracy_score(y_test, sgd.predict(X_test))
print(f"SGD Classifier Accuracy: {sgd_acc:.2%}")

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_acc = accuracy_score(y_test, lda.predict(X_test))
print(f"Linear Discriminant Analysis Accuracy: {lda_acc:.2%}")

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_acc = accuracy_score(y_test, qda.predict(X_test))
print(f"Quadratic Discriminant Analysis Accuracy: {qda_acc:.2%}")

pa = PassiveAggressiveClassifier(random_state=42)
pa.fit(X_train, y_train)
pa_acc = accuracy_score(y_test, pa.predict(X_test))
print(f"Passive Aggressive Classifier Accuracy: {pa_acc:.2%}")

bag = BaggingClassifier(random_state=42)
bag.fit(X_train, y_train)
bag_acc = accuracy_score(y_test, bag.predict(X_test))
print(f"Bagging Classifier Accuracy: {bag_acc:.2%}")

voting = VotingClassifier(estimators=[
    ('lr', lr), ('rf', rf), ('knn', knn)
], voting='soft')
voting.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting.predict(X_test))
print(f"Voting Classifier Accuracy: {voting_acc:.2%}")

cat = CatBoostClassifier(verbose=0, random_state=42)
cat.fit(X_train, y_train)
cat_acc = accuracy_score(y_test, cat.predict(X_test))
print(f"CatBoost Accuracy: {cat_acc:.2%}")

print("\n")


all_scores = {
    'KNN': knn_acc,
    'Random Forest': rf_acc,
    'Decision Tree': dt_acc,
    'Logistic Regression': lr_acc,
    'ANN (MLP)': ann_acc,
    'Naive Bayes': nb_acc,
    'Extra Trees': et_acc,
    'Gradient Boosting': gb_acc,
    'AdaBoost': ada_acc,
    'XGBoost': xgb_acc,
    'LightGBM': lgb_acc,
    'Ridge': ridge_acc,
    'Lasso': lasso_acc,
    'SGD': sgd_acc,
    'LDA': lda_acc,
    'QDA': qda_acc,
    'Passive Aggressive': pa_acc,
    'Bagging': bag_acc,
    'Voting Classifier': voting_acc,
    'CatBoost': cat_acc
}


best_model = max(all_scores, key=all_scores.get)
print(f"Best Model: {best_model} with Accuracy: {all_scores[best_model]:.2%}")