import pandas as pd
from prophet import Prophet
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_test_data(data_path, split_date):
    #Lendo o dataframe
    df = pd.read_csv(data_path)
    #Definindo os dados de teste
    split_date = pd.to_datetime(split_date)
    df_test = df[pd.to_datetime(df['din_instante']) >= split_date]

    #Separando Dummies para usar no modelo
    dummies = pd.get_dummies(df['Estacao'], prefix='Estacao')
    dummies_test = pd.get_dummies(df_test['Estacao'], drop_first=True)

    df_prophet_test = pd.concat([df_test[['din_instante', 'val_geracao']], dummies_test], axis=1)
    return df_prophet_test, dummies

def evaluate_model(model, df_test, dummies):
    # Fazendo previsões
    future = model.make_future_dataframe(periods=len(df_test))
    future = pd.concat([future, dummies.reindex(future.index, fill_value=0)], axis=1)
    forecast = model.predict(future)

    # Calculando erros para o modelo Prophet
    mae = mean_absolute_error(df_test['val_geracao'], forecast['yhat'].iloc[-len(df_test):])
    mse = mean_squared_error(df_test['val_geracao'], forecast['yhat'].iloc[-len(df_test):])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(df_test['val_geracao'], forecast['yhat'].iloc[-len(df_test):])

    #Plotando gráfico
    plt.figure(figsize=(14, 7))
    plt.plot(df_test['din_instante'], df_test['val_geracao'], label='Real')
    plt.plot(df_test['din_instante'], forecast['yhat'].iloc[-len(df_test):], label='Previsão Prophet', color='red')
    plt.fill_between(df_test['din_instante'], forecast['yhat'].iloc[-len(df_test):], alpha=0.1,  color='red')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    metrics_text = (f"MAE: {mae:.2f}\n"
                f"MSE: {mse:.2f}\n"
                f"RMSE: {rmse:.2f}\n"
                f"MAPE: {mape:.2%}")
    plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Previsão de Geração de Energia Solar com Prophet')
    plt.xlabel('Data')
    plt.ylabel('Geração de Energia Solar')
    plt.legend()
    plt.show()

def main():
    #Caminho para o modelo
    model_path = '../models/prophet_model.pkl'
    #Caminho para o dataframe
    data_path = '../Data/dataframe.csv'
    #Carregando o modelo
    model = joblib.load(model_path)

    df_test, dummies_test = load_test_data(data_path, '2023-01-01')
    evaluate_model(model, df_test, dummies_test)

if __name__ == "__main__":
    main()
