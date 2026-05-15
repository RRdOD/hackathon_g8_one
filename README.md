# 🚀 Previsão de Churn em Serviço de Streaming – Hackathon No Country

**Problema de negócio:** Uma empresa de streaming perde cerca de 25% dos clientes anualmente, mas não consegue identificar quais clientes estão em risco nem os motivos da saída.

**Solução:** Pipeline completo de machine learning que prevê a probabilidade de cancelamento de cada cliente, permitindo ações de retenção direcionadas.

**Impacto potencial:** Com o modelo, é possível concentrar esforços nos 20% clientes de maior risco, recuperando uma parcela significativa da receita que seria perdida.

## 📊 Resultados Principais

| Métrica               | Valor        |
|-----------------------|--------------|
| F1‑Score (Churn)      | **0,943**    |
| AUC‑ROC               | **0,995**    |
| Recall (detecção)     | **96,4%**    |
| Precisão (alertas)    | **92,3%**    |
| Acurácia geral        | 97,1%        |

Apenas 3,6% dos clientes que cancelam não são detectados (54 falsos negativos em 6.000 testes).

### 🔍 O que leva um cliente a cancelar?

A análise de importância de features e os padrões MNAR revelaram os principais fatores:

- **Baixo engajamento** (tempo de sessão 57% abaixo da média de quem fica)
- **Muitos dias sem acessar** (clientes com +34 dias de inatividade têm 52% de chance de churn)
- **Não avaliar conteúdo** (clientes que não deixam avaliação cancelam 2,2× mais)
- **Contrato mensal** vs. anual (maior flexibilidade para sair)
- **Visualizações baixas** (clientes com poucas visualizações têm perfil de risco)

## 🧠 O que está no notebook

1. **Diagnóstico de dados faltantes (MNAR)**  
   Testes estatísticos (Chi‑Square, ANOVA, KS) mostram que a ausência de avaliações não é aleatória — é um sinal de desengajamento.

2. **Engenharia de features**  
   Criação de scores de engajamento, risco e flags de inatividade que capturam o comportamento do cliente.

3. **Modelagem preditiva**  
   Random Forest com otimização de hiperparâmetros (GridSearch) e seleção de features (RFE). O modelo foi calibrado com Isotonic Regression para produzir probabilidades confiáveis.

4. **Validação rigorosa (5 testes)**  
   Validação cruzada estável (AUC 0,9956 ± 0,0003), teste de overfitting, análise de sensibilidade a ruído e data drift, e investigação detalhada de erros.

5. **Preparação para deploy**  
   Artefatos prontos (modelo serializado, threshold ótimo, encoders) e sugestão de monitoramento contínuo (KS‑test, limites de alerta).

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.x-green)
![Scikit‑learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.x-lightblue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red)


## 🚀 Como executar

1. Clone o repositório e instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib

2. Abra o notebook Diário_de_Bordo_Rafael_Dias.ipynb e execute todas as células.

3. Para fazer previsões com o modelo já treinado:
```
import joblib
modelo = joblib.load('modelo_churn.joblib')
# X_novo deve ser um DataFrame com as mesmas 25 features
probabilidade = modelo.predict_proba(X_novo)[:, 1]
```
Autor: Rafael Dias
Contato: [LinkedIn](https://www.linkedin.com/in/rafael-dias-datascience/)
