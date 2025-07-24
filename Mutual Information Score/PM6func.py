def catconsep(df):
    cat = list(df.columns[df.dtypes=='object'])
    con = list(df.columns[df.dtypes!='object'])
    return cat, con

def replacer(df):
    cat, con = catconsep(df)
    for i in df:
        if i in cat:
            md = df[i].mode()[0]
            df[i] = df[i].fillna(md)
        else:
            mn = df[i].mean()
            df[i] = df[i].fillna(mn)
    print("Null Values filled")


def evaluate_model(xtrain, xtest, ytrain, ytest, model):
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    tr_mse = mean_squared_error(ytrain,ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_r2 = r2_score(ytrain,ypred_tr)
    # Print metrics for training data
    print("Metrics of Train Data")
    print(f"MSE: {tr_mse:.2f}")
    print(f"MAE: {tr_mae:.2f}")
    print(f"RMSE: {tr_rmse:.2f}")
    print(f"R2: {tr_r2:.2f}")
    print("\n==============================\n")
    ts_mse = mean_squared_error(ytest,ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_r2 = r2_score(ytest,ypred_ts)
    # Print metrics for testing data
    print("Metrics of Test Data")
    print(f"MSE: {ts_mse:.2f}")
    print(f"MAE: {ts_mae:.2f}")
    print(f"RMSE: {ts_rmse:.2f}")
    print(f"R2: {ts_r2:.2f}")