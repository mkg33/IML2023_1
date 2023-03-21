import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def fit1(X, y, lam):
    
    w = np.zeros((21,))
    clf = Ridge(alpha = lam).fit(X,y)
    w = clf.coef_
    assert w.shape == (21,)
    return w

def calculate_RMSE(w, X, y):
    RMSE = 0
    
    y_pred = np.dot(X,w)
    RMSE = mean_squared_error(y, y_pred)**0.5
    assert np.isscalar(RMSE)
    return RMSE

def min_RMSE(X, y, lambdas):
    store_RMSE = np.zeros((len(lambdas),))
    
    for i in range(len(lambdas)):
        w = fit1(X,y,lambdas[i])
        store_RMSE[i] = calculate_RMSE(w, X, y)
    
    for i in range(len(store_RMSE)):
        if store_RMSE[i] == np.min(store_RMSE):
            print("lambda_opt = ", lambdas[i])
            return i
        
    return i

def transform_data(X):
    X_transformed = np.zeros((700, 21))
    
    X_transformed[:, 0:5] = X
    X_transformed[:, 5:10] = X**2
    X_transformed[:, 10:15] = np.exp(X)
    X_transformed[:, 15:20] = np.cos(X)
    X_transformed[:, 20] = 1
    
    assert X_transformed.shape == (700, 21)
    return X_transformed

def fit(X,y, lambdas):
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    RMSE = calculate_RMSE(w, X_transformed, y)
    i = min_RMSE(X_transformed, y, lambdas)
    reg = Ridge(alpha = lambdas[i])
    reg = reg.fit(X_transformed, y)
    w = reg.coef_
    assert w.shape == (21,)
    return w
    

if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns = ["Id", "y"])
    
    X = data.to_numpy()
    lambdas = np.arange(0, 1, 0.0001)
    w = fit(X,y,lambdas)
    
    np.savetxt("./results.csv", w, fmt="%.12f")
    
    
