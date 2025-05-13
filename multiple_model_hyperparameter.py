import mlflow
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,50],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
scores = []

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("MultipleModels_Mlflow1")

with mlflow.start_run(run_name="MultipleModels"):
    for model_name, mp in model_params.items():
     clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
     clf.fit(iris.data, iris.target)
     results=pd.DataFrame(clf.cv_results_)
    #export results dataframe to excel
    #  results.to_excel(f"{model_name}_results1.xlsx", index=False)
     for i, params in enumerate(clf.cv_results_["params"]):
        with mlflow.start_run(run_name=f"run_{i}{model_name}", nested=True):
            acc = clf.cv_results_["mean_test_score"][i]
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", acc)
     scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
     })
     with mlflow.start_run(run_name=f"best_model_{model_name}",nested=True):
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("best_score", clf.best_score_)
        
        # Train the best model with optimal parameters
        best_model = mp['model']
        best_model.set_params(**clf.best_params_)
        best_model.fit(iris.data, iris.target)
        
        # Register the model in MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=f"models/{model_name}",
            registered_model_name=f"Iris-{model_name}"
        )
    df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
    print(df)