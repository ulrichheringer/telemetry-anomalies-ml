# telemetry-anomalies-ml

Identifica pontos anômalos em CPU, RAM, latência e tráfego de API com Isolation Forest treinado só em observações normais.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: Isolation Forest (contamination 0,05), ajustado apenas em 1.350 linhas normais do treino; holdout com 500 linhas e 50 anomalias conhecidas — precisão **0,6338**, recall **0,9000**, F1 **0,7438**. São medições do gerador sintético, com quatro padrões de anomalia definidos, e não afirmam desempenho em telemetria real.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
