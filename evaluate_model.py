import pandas as pd
from prophet import Prophet
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def load_test_data(test_path, split_date):
    df = pd.read_csv(test_path)
    split_date = pd.to_datetime(split_date)
    df_test = df[pd.to_datetime(df['din_instante']) >= split_date]
    dummies = pd.get_dummies(df['Estacao'], prefix='Estacao')
    dummies_test = pd.get_dummies(df_test['Estacao'], drop_first=True)
    df_prophet_test = pd.concat([df_test[['din_instante', 'val_geracao']], dummies_test], axis=1)
    return df_prophet_test, dummies

def evaluate_model(model, df_test, dummies):
    future = model.make_future_dataframe(periods=len(df_test))
    future = pd.concat([future, dummies.reindex(future.index, fill_value=0)], axis=1)
    forecast = model.predict(future)

    plt.figure(figsize=(14, 7))
    plt.plot(df_test['din_instante'], df_test['val_geracao'], label='Real')
    plt.plot(df_test['din_instante'], forecast['yhat'].iloc[-len(df_test):], label='Previsão Prophet', color='red')
    plt.fill_between(df_test['din_instante'], forecast['yhat'].iloc[-len(df_test):], alpha=0.1,  color='red')
    plt.title('Previsão de Geração de Energia Solar com Prophet')
    plt.xlabel('Data')
    plt.ylabel('Geração de Energia Solar')
    plt.legend()
    plt.show()

    mae = mean_absolute_error(df_test['target'], forecast['yhat'].iloc[-len(df_test):])
    mse = mean_squared_error(df_test['target'], forecast['yhat'].iloc[-len(df_test):])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(df_test['target'], forecast['yhat'].iloc[-len(df_test):])

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')

def main():
    model_path = 'models/prophet_model.pkl'
    test_data_path = 'Data/dataframe.csv'
    
    model = joblib.load(model_path)
    df_test, dummies_test = load_test_data(test_data_path, '2023-01-01')
    evaluate_model(model, df_test, dummies_test)

if __name__ == "__main__":
    main()
