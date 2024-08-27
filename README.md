# Projeto de Previsão de Energia Solar

Este projeto é focado na criação e ajuste de modelos de previsão de energia solar, utilizando técnicas de machine learning para prever a geração de energia com base em dados históricos.

## Sumário

- [Introdução](#introdução)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura do Projeto](#estrutura-do-projeto)

## Introdução

A previsão precisa da geração de energia solar é crucial para o planejamento e operação de sistemas de energia renovável. Este projeto busca criar modelos preditivos que possam ser usados por operadores de redes de energia, desenvolvedores de projetos solares e outros stakeholders para otimizar a gestão e distribuição de energia solar.

## Instalação

Para instalar e configurar o ambiente do projeto, siga os passos abaixo:

1. **Clone o repositório:**

    ```bash
    git clone https://github.com/luizcvfilho/solar_ernergy_forecast.git
    ```

2. **Crie e ative um ambiente virtual:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

1. **Preparação dos Dados:**

   Certifique-se de que os dados estão no formato correto e localizados no diretório `Data/`. O projeto inclui scripts para o pré-processamento dos dados, como a limpeza e normalização.

2. **Treinamento do Modelo:**

   Para treinar o modelo, altere o caminho presente no script principal de treinamento para o desejado e depois o execute:

   ```bash
   python train_model.py
    ```

    Isso gerará um modelo treinado baseado nos dados fornecidos

3. **Avaliação do Modelo:**

    Após o treinamento, você pode avaliar o desempenho do modelo alterando o caminho no script de avaliação e o executando:

   ```bash
   python evaluate_model.py
    ```

## Estrutura do Projeto

A estrutura do projeto é organizada da seguinte maneira:

```bash
   projeto-previsao-energia-solar/
    │
    ├── Data/                   # Diretório para armazenar os dados brutos e processados
    ├── models/                 # Modelo treinado
    ├── scripts/
    │   ├── train_model.py      # Script principal para treinamento do modelo
    │   ├── evaluate_model.py   # Script para avaliação do modelo
    │ 
    ├── project.ipynb           # Notebook com a análise e manipulação dos dados
    ├── requirements.txt        # Arquivo de instalações do projeto
    └── README.md               # Este arquivo
```
