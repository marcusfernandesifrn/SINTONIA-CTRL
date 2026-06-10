# 📘 RED — SINTONIA-CTRL

**Recurso Educacional Digital** para a disciplina *Modelagem e Sistemas Lineares*  
Curso de Engenharia de Energia — IFRN Campus Natal-Central (CNAT)  
Autor: Marcus V A Fernandes · marcus.fernandes@ifrn.edu.br · v1.0 · 2026

---

## Estrutura do repositório

```
red_msl/
├── streamlit_app.py               ← Entrypoint principal (roteador)
├── requirements.txt
├── README.md
└── modulos/
    ├── __init__.py
    ├── sinais_e_sistemas_lineares.py
    ├── transformada_de_laplace.py
    ├── dinamica_sistemas_ordem_1.py
    ├── dinamica_sistemas_ordem_2.py
    ├── realimentacao_malha_fechada.py
    ├── realimentacao_lgr.py
    ├── estabilidade_realimentacao.py
    ├── resposta_frequencia.py
    ├── criterio_nyquist.py
    └── espaco_de_estados.py
```

## Módulos

| # | Módulo | Ícone |
|---|---|---|
| 1 | Sinais e Sistemas Lineares | 📡 |
| 2 | Transformada de Laplace | 🌀 |
| 3.1 | Dinâmica — Sistemas de Ordem 1 | 📈 |
| 3.2 | Dinâmica — Sistemas de Ordem 2 | 📊 |
| 4.1 | Análise com Realimentação — Malha Fechada | 🔄 |
| 4.2 | Análise com Realimentação — Perturbação e LGR | 📍 |
| 5 | Estabilidade com Realimentação | ⚖️ |
| 6.1 | Resposta em Frequência — Diagramas de Bode | 📉 |
| 6.2 | Resposta em Frequência — Critério de Nyquist | 🔁 |
| 7 | Análise no Espaço de Estados | 🧮 |

## Deploy no Streamlit Cloud

1. Faça o upload ou push deste repositório no GitHub
2. No Streamlit Cloud, configure:
   - O Streamlit Cloud detecta `streamlit_app.py` automaticamente — nenhuma configuração necessária.
3. A pasta `modulos/` **não deve** ser renomeada para `pages/`
   (o nome `pages/` ativa o sistema legado de navegação do Streamlit)

## Execução local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
