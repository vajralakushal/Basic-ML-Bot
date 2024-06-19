#import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import concurrent.futures
import re #for lightgbm

# Suppress Convergence Warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define hyperparameter grids for classification models
classification_param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, solver='saga', tol=1e-3),
        'params': {
            'C': [0.01, 0.1, 1],
            'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
        },
        'package': 'sklearn.linear_model'
    },
    'Decision Tree Classifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None] + list(range(10, 51, 10)),
            'min_samples_split': range(2, 11)
        },
        'package': 'sklearn.tree'
    },
    'Random Forest Classifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': range(10, 201, 10),
            'max_depth': [None] + list(range(10, 51, 10)),
            'min_samples_split': range(2, 11)
        },
        'package': 'sklearn.ensemble'
    },
    'Gradient Boosting Classifier': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': range(10, 201, 10),
            'learning_rate': [0.001, 0.01, 0.1, 0.3],
            'max_depth': range(3, 11)
        },
        'package': 'sklearn.ensemble'
    },
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'package': 'sklearn.svm'
    },
    'KNeighbors Classifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': range(3, 13),
            'weights': ['uniform', 'distance']
        },
        'package': 'sklearn.neighbors'
    },
    'Gaussian NB': {
        'model': GaussianNB(),
        'params': {},
        'package': 'sklearn.naive_bayes'
    },
    'MLP Classifier': {
        'model': MLPClassifier(max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        },
        'package': 'sklearn.neural_network'
    },
    'XGB Classifier': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': range(50, 201, 50),
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': range(3, 11)
        },
        'package': 'xgboost'
    }
    # ,
    # 'CatBoost Classifier': {
    #     'model': CatBoostClassifier(verbose=0),
    #     'params': {
    #         'iterations': range(100, 301, 50),
    #         'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #         'depth': range(3, 11)
    #     },
    #     'package': 'catboost'
    # },
    # 'LGBM Classifier': {
    #     'model': LGBMClassifier(),
    #     'params': {
    #         'n_estimators': range(50, 201, 50),
    #         'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #         'max_depth': range(3, 11)
    #     },
    #     'package': 'lightgbm'
    # }
}

# Define hyperparameter grids for regression models
regression_param_grids = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {},
        'package': 'sklearn.linear_model'
    },
    'Decision Tree Regressor': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [None] + list(range(10, 51, 10)),
            'min_samples_split': range(2, 11)
        },
        'package': 'sklearn.tree'
    },
    'Random Forest Regressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': range(10, 201, 10),
            'max_depth': [None] + list(range(10, 51, 10)),
            'min_samples_split': range(2, 11)
        },
        'package': 'sklearn.ensemble'
    },
    'Gradient Boosting Regressor': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': range(10, 201, 10),
            'learning_rate': [0.001, 0.01, 0.1, 0.3],
            'max_depth': range(3, 11)
        },
        'package': 'sklearn.ensemble'
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'package': 'sklearn.svm'
    },
    'KNeighbors Regressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': range(3, 13),
            'weights': ['uniform', 'distance']
        },
        'package': 'sklearn.neighbors'
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {},
        'package': 'sklearn.linear_model'
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {},
        'package': 'sklearn.linear_model'
    },
    'MLP Regressor': {
        'model': MLPRegressor(max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        },
        'package': 'sklearn.neural_network'
    },
    'XGB Regressor': {
        'model': XGBRegressor(),
        'params': {
            'n_estimators': range(50, 201, 50),
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': range(3, 11)
        },
        'package': 'xgboost'
    }
    # ,
    # 'CatBoost Regressor': {
    #     'model': CatBoostRegressor(verbose=0),
    #     'params': {
    #         'iterations': range(100, 301, 50),
    #         'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #         'depth': range(3, 11)
    #     },
    #     'package': 'catboost'
    # },
    # 'LGBM Regressor': {
    #     'model': LGBMRegressor(),
    #     'params': {
    #         'n_estimators': range(50, 201, 50),
    #         'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #         'max_depth': range(3, 11)
    #     },
    #     'package': 'lightgbm'
    # }
}

# in the past, logistic regression took WAY too long. sometimes it's better just to see the other models ASAP
# would love for someone to correct me if that's a bad approach
def run_model_with_timeout(grid_search, X_scaled, y, timeout):
    def model_fit():
        grid_search.fit(X_scaled, y)
        return grid_search

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(model_fit)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            return None

# Need to load the data and get the input and outputs
def load_data(file_path, target_column):
    data = pd.read_csv(file_path)
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    data = data.dropna()
    #clean up any qualitative columns and assign them codes
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    #unique_values = {col: data[col].unique().tolist() for col in non_numeric_columns}
    
    mappings = {}
    for col in non_numeric_cols:
        codes, uniques = pd.factorize(data[col])
        data[col] = codes + 1  # Start codes from 1 instead of 0
        mappings[col] = {unique: code + 1 for code, unique in enumerate(uniques)}
    
    x = data.drop(columns=[target_column])
    y = data[target_column]
    print(data)
    return x, y

#run the models, whether they are Classification or Regression models

def run_classification_models(X, y):
    results = {}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for name, model_info in classification_param_grids.items():
        model = model_info['model']
        params = model_info['params']
        grid_search = GridSearchCV(model, params, cv=kf, scoring='accuracy', n_jobs=-1, error_score='raise')
        result = run_model_with_timeout(grid_search, X_scaled, y, 300)
        if result:
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            y_pred = best_model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            results[name] = {
                'best_score': best_score,
                'best_params': best_params,
                'accuracy': accuracy
            }
        else:
            results[name] = {
                'best_score': None,
                'best_params': None,
                'accuracy': None
            }
    return results

def run_regression_models(X, y):
    results = {}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for name, model_info in regression_param_grids.items():
        model = model_info['model']
        params = model_info['params']
        grid_search = GridSearchCV(model, params, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
        result = run_model_with_timeout(grid_search, X_scaled, y, 300)
        if result:
            best_model = grid_search.best_estimator_
            best_score = -grid_search.best_score_  # Convert negative MSE to positive
            best_params = grid_search.best_params_
            y_pred = best_model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            results[name] = {
                'best_score': best_score,
                'best_params': best_params,
                'mse': mse
            }
        else:
            results[name] = {
                'best_score': None,
                'best_params': None,
                'mse': None
            }
    return results

def main(file_path, target_column):
    X, y = load_data(file_path, target_column)
    
    if y.nunique() <= 10:  # Simple heuristic to decide between classification and regression
        print("Running classification models...")
        results = run_classification_models(X, y)
        best_model_info = None
        best_model_accuracy = 0
        
        for name, result in results.items():
            if result['accuracy'] is not None:
                accuracy = result['accuracy']
                if 0.90 <= accuracy <= 0.98:
                    best_model_info = (name, result)
                    break
                elif accuracy > best_model_accuracy:
                    best_model_info = (name, result)
                    best_model_accuracy = accuracy

        if best_model_info:
            model_name, result = best_model_info
            print(f"Selected Model: {model_name}")
            print(f"Best Hyperparameters = {result['best_params']}, Best Score = {result['best_score']:.4f}, Accuracy = {result['accuracy']:.4f}")

            # Generate Python file for the selected model
            with open(f"{model_name.replace(' ', '_')}_model.py", "w") as f:
                f.write(f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from {classification_param_grids[model_name]['package']} import {model_name.replace(' ', '')}
import re

def load_data(file_path, target_column):
    data = pd.read_csv(file_path)
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    data = data.dropna()
    #clean up any qualitative columns and assign them codes
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    mappings = {{}}
    for col in non_numeric_cols:
        codes, uniques = pd.factorize(data[col])
        data[col] = codes + 1  # Start codes from 1 instead of 0
        mappings[col] = {{unique: code + 1 for code, unique in enumerate(uniques)}}
    
    x = data.drop(columns=[target_column])
    y = data[target_column]
    #print(data)
    return x, y

def main(file_path, target_column):
    X, y = load_data(file_path, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = {model_name.replace(' ', '')}(**{result['best_params']})
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {{accuracy:.4f}}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file_path> <target_column>")
    else:
        file_path = sys.argv[1]
        target_column = sys.argv[2]
        main(file_path, target_column)
""")
        else:
            print("No suitable model found.")
    else:
        print("Running regression models...")
        results = run_regression_models(X, y)
        best_model_info = None
        best_model_mse = float('inf')

        for name, result in results.items():
            if result['mse'] is not None:
                mse = result['mse']
                if mse < best_model_mse:
                    best_model_info = (name, result)
                    best_model_mse = mse

        if best_model_info:
            model_name, result = best_model_info
            print(f"Selected Model: {model_name}")
            print(f"Best Hyperparameters = {result['best_params']}, Best Score = {result['best_score']:.4f}, MSE = {result['mse']:.4f}")

            # Generate Python file for the selected model
            with open(f"{model_name.replace(' ', '_')}_model.py", "w") as f:
                f.write(f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from {regression_param_grids[model_name]['package']} import {model_name.replace(' ', '')}
import re

def load_data(file_path, target_column):
    data = pd.read_csv(file_path)
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    data = data.dropna()
    #clean up any qualitative columns and assign them codes
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    mappings = {{}}
    for col in non_numeric_cols:
        codes, uniques = pd.factorize(data[col])
        data[col] = codes + 1  # Start codes from 1 instead of 0
        mappings[col] = {{unique: code + 1 for code, unique in enumerate(uniques)}}
    
    x = data.drop(columns=[target_column])
    y = data[target_column]
    #print(data)
    return x, y

def main(file_path, target_column):
    X, y = load_data(file_path, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = {model_name.replace(' ', '')}(**{result['best_params']})
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {{mse:.4f}}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file_path> <target_column>")
    else:
        file_path = sys.argv[1]
        target_column = sys.argv[2]
        main(file_path, target_column)
""")
        else:
            print("No suitable model found.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file_path> <target_column>")
    else:
        file_path = sys.argv[1]
        target_column = sys.argv[2]
        main(file_path, target_column)