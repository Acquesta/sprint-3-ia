# Previsão de evasão no pós-venda Ford

Sprint 3 de Inteligência Artificial & Machine Learning (FIAP) no Challenge Ford 2026, Desafio 2: retenção de clientes no pós-venda. O trabalho faz parte do CarSync, o nosso app com o assistente de voz Fordinho.

| Integrante | RM |
|---|---|
| Alexia Ramalho | 558385 |
| Enzo Real | 557943 |
| Gustavo Pasquini | 555454 |
| Hellen Silva | 559008 |
| Lorenzo Acquesta | 557397 |

O trabalho completo, já executado, está no notebook [`Sprint3_Churn_PosVenda.ipynb`](Sprint3_Churn_PosVenda.ipynb). Este README resume a solução.

## Problema

Depois que a garantia acaba, muitos donos de Ford deixam de fazer revisão na concessionária e passam a usar oficinas independentes. No nosso Business Model Canvas, propomos mirar as campanhas de retenção só nos clientes que um modelo preditivo aponta com risco de evasão, e aqui construímos esse modelo.

Para cada cliente, o modelo estima a probabilidade de ele não fazer nenhum serviço na rede Ford nos 12 meses seguintes. É uma classificação binária supervisionada, com classes desbalanceadas (29,4% de evasores).

## Dados

Como não temos acesso ao CRM da Ford, geramos a base com [`gerar_dados.py`](gerar_dados.py): 10.000 clientes com dados de concessionária, telemetria, app CarSync e pesquisa NPS. Depois colocamos na base a sujeira comum em CRM: 100 linhas duplicadas, valores ausentes, texto fora do padrão, valores impossíveis e uma coluna que vaza o alvo. Os pesos do gerador são premissas nossas, sem base em números de mercado.

## Preparação

- Tiramos as duplicatas, os identificadores e a coluna `motivo_saida_pesquisa`, que só existe para quem já evadiu e entregaria a resposta ao modelo.
- Padronizamos o texto das categorias (73 grafias viraram 17 valores) e trocamos por vazio 309 valores impossíveis, como km de 999.999 e NPS de 99. Os extremos plausíveis ficaram, porque a regra do IQR marcaria centenas de clientes reais como outliers.
- Criamos seis variáveis: idade do carro, km por ano, garantia ativa, NPS respondido, modelo fora de linha e uso do app.
- Separamos 80% para treino e 20% para teste, com estratificação. Na seleção, tiramos `ano_modelo` e `km_atual` (repetem a idade do carro) e `idade_cliente` e `aceita_whatsapp` (sem relação com a evasão).
- Imputação pela mediana ou moda, `log1p` nas variáveis assimétricas, padronização e one-hot, ajustados só no treino dentro de um `Pipeline`.

## Modelos

Comparamos os quatro algoritmos vistos em aula com validação cruzada estratificada de 5 folds, usando a PR-AUC como métrica principal por causa do desbalanceamento. As 80 configurações testadas ficaram registradas no MLflow.

| Modelo | Configurações | Melhor configuração | PR-AUC no CV |
|---|---|---|---|
| Regressão Logística | 16 | `C=0.1`, penalidade L1 | 0,734 |
| Random Forest | 24 | profundidade 16, folhas com 5 ou mais clientes | 0,721 |
| KNN | 16 | 61 vizinhos ponderados pela distância | 0,713 |
| Árvore de Decisão | 24 | profundidade 8, folhas com 100 ou mais clientes | 0,669 |

## Resultados no teste

2.000 clientes que o treino nunca viu, com limiar de 0,5 (exceto na última linha).

| Modelo | PR-AUC | ROC-AUC | F1 | Recall | Precisão |
|---|---|---|---|---|---|
| Regressão Logística | 0,751 | 0,872 | 0,650 | 0,584 | 0,732 |
| KNN | 0,727 | 0,857 | 0,619 | 0,533 | 0,739 |
| Random Forest | 0,726 | 0,861 | 0,637 | 0,565 | 0,730 |
| Árvore de Decisão | 0,686 | 0,841 | 0,635 | 0,596 | 0,679 |
| Regressão Logística com limiar 0,32 | 0,751 | 0,872 | 0,683 | 0,761 | 0,620 |

A linha de base, que sempre responde "ficou", tem PR-AUC de 0,294 e ROC-AUC de 0,5.

## Modelo escolhido

Ficamos com a Regressão Logística com penalidade L1, `C=0.1` e limiar de 0,32. Ela teve a maior PR-AUC no CV e no teste. A Random Forest ficou só 0,013 atrás no CV, dentro do desvio entre folds, e para esse empate a regra que definimos antes de olhar o teste manda ficar com o modelo mais simples. Os coeficientes viram odds ratios fáceis de explicar (sem garantia, as chances de sair são 2,6 vezes maiores), e o modelo não mostra sobreajuste.

Os fatores que mais pesam são o tempo desde a última visita, a distância até a concessionária, a garantia, as revisões feitas na rede e a nota do NPS.

## Uso no CarSync e deploy

Ordenando os clientes pelo risco, os 20% primeiros da lista reúnem 53% dos evasores. Entre os clientes de prioridade alta no teste, 78,8% de fato evadiram, contra 6,4% nos de prioridade baixa.

Propomos um job diário (DAG no Airflow) que pontua a base e alimenta o dashboard da concessionária, e uma API que o Fordinho consulta antes de conversar com o motorista, para oferecer o agendamento da revisão a quem está em risco alto. Os dois usam o modelo registrado no MLflow com o alias `@champion`.

## Como rodar

O projeto usa Python 3.12.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook Sprint3_Churn_PosVenda.ipynb
```

Os dados já estão em `data/`, e o notebook roda o `gerar_dados.py` sozinho se o arquivo não existir. Para ver os experimentos depois de rodar o notebook:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Onde está cada item da rubrica

| Item | Onde está |
|---|---|
| 1. Compreensão do problema | Notebook, seção 1 |
| 2. Preparação dos dados | Notebook, seção 2 |
| 3. Desenvolvimento dos modelos | Notebook, seção 3, e experimentos no MLflow |
| 4. Avaliação e comparação | Notebook, seções 4 e 5, e as tabelas deste README |
| 6. Conclusão | Notebook, seção 6, e as seções "Modelo escolhido" e "Uso no CarSync e deploy" deste README |
