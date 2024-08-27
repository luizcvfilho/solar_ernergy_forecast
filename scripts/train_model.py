import pandas as pd
from prophet import Prophet
from joblib import dump

def load_and_prepare_data(data_path, split_date):
    #Lendo o dataframe
    df = pd.read_csv(data_path)

    #Definindo os dados de treino
    split_date = pd.to_datetime(split_date)
    df_train = df[pd.to_datetime(df['din_instante']) < split_date]

    #Separando Dummies para usar no modelo
    dummies_train = pd.get_dummies(df_train['Estacao'], prefix='Estacao')

    df_prophet_train = pd.concat([df_train, dummies_train], axis=1)
    df_prophet_train.rename(columns={'din_instante': 'ds', 'val_geracao': 'y'}, inplace=True)
    return df_prophet_train, dummies_train

def train_model(df_prophet_train, dummies_train):
    # Criando e definindo o modelo
    model_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0
    )
    model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    #Adicionado cada variável dummy como um regressor exógeno
    for col in dummies_train.columns:
        model_prophet.add_regressor(col)

    #Ajustando o modelo
    model_prophet.fit(df_prophet_train)
    return model_prophet

def main():
    #Caminho para o dataframe
    data_path = '../Data/dataframe.csv'  
    #Chamando a função
    df_prophet_train, dummies_train = load_and_prepare_data(data_path, '2023-01-01')
    #Salvando o Modelo
    model = train_model(df_prophet_train, dummies_train)
    dump(model, '../models/prophet_model.pkl')

    print("Modelo treinado e salvo com sucesso!")

if __name__ == "__main__":
    main()
