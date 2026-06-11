# 🎛️ SINTONIA — Sistemas de Controle

**Sistemas Interativos para Teoria e Otimização no Nível de Interpretação e Aprendizagem**

> Recurso Educacional Digital (RED) de acesso livre para o estudo de
> Sistemas de Controle e Modelagem de Sistemas Lineares.

---

## 📌 Sobre o SINTONIA

O **SINTONIA** é um RED desenvolvido para apoiar o ensino e a aprendizagem
de Sistemas de Controle e Modelagem de Sistemas Lineares em cursos de Engenharia.
O material é organizado em módulos progressivos — de fundamentos de sinais e sistemas
até representação no espaço de estados — com ênfase em compreensão visual e exploração
paramétrica por meio de **exploradores interativos**.

Cada módulo combina teoria com equações e exemplos analíticos, figuras geradas
numericamente e exploradores com controles deslizantes e campos de entrada que
permitem observar o efeito de parâmetros em tempo real, sem necessidade de
reexecução.

---

## 🎓 Indicações de Uso

O SINTONIA pode ser utilizado como material de apoio em qualquer disciplina de
Engenharia que aborde Sistemas de Controle ou Modelagem de Sistemas Lineares.
Aplicações identificadas pelo autor:

| Disciplina | Curso | Instituição |
|---|---|---|
| *Modelagem e Sistemas Lineares* | Engenharia de Energia | IFRN — Campus Natal-Central (CNAT) |
| *Sistemas de Controle 1* | Engenharia Elétrica | UFRN |

O material é igualmente adequado a outros cursos de Engenharia (Mecânica,
Mecatrônica, de Controle e Automação, Química, entre outros) que contemplem
conteúdos de sistemas lineares, funções de transferência, resposta temporal,
resposta em frequência e espaço de estados.

---

## 🗂️ Módulos

| Módulo | Tema | Ícone | Exploradores |
|---|---|:---:|:---:|
| 1 | Sinais e Sistemas Lineares | 📡 | 1 |
| 2 | Transformada de Laplace | 🌀 | 1 |
| 3.1 | Dinâmica — Sistemas de Ordem 1 | 📈 | 5 |
| 3.2 | Dinâmica — Sistemas de Ordem 2 | 📊 | 6 |
| 4.1 | Análise com Realimentação — Malha Fechada | 🔄 | 3 |
| 4.2 | Análise com Realimentação — Perturbação e LGR | 📍 | 3 |
| 5 | Estabilidade com Realimentação | ⚖️ | 2 |
| 6.1 | Resposta em Frequência — Diagramas de Bode | 📉 | 12 |
| 6.2 | Resposta em Frequência — Critério de Nyquist | 🔁 | 3 |
| 7 | Análise no Espaço de Estados | 🧮 | 3 |
| **Total** | | | **39** |

---

## 📁 Estrutura do Repositório

```
SINTONIA-CTRL/
├── streamlit_app.py          ← Entrypoint principal e página inicial
├── requirements.txt          ← Dependências Python
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

> **Atenção:** a pasta de módulos **não deve** ser renomeada para `pages/`.
> O nome `pages/` ativa o sistema legado de navegação do Streamlit e
> causa conflito com o roteador baseado em `st.navigation()`.

---

## 🚀 Deploy no Streamlit Cloud

1. Faça o push deste repositório para o GitHub (repositório público ou privado)
2. Acesse [share.streamlit.io](https://share.streamlit.io) e crie um novo app
3. Selecione o repositório, branch `main` e arquivo principal `streamlit_app.py`
4. O Streamlit Cloud detecta `streamlit_app.py` automaticamente — nenhuma
   configuração adicional é necessária

---

## 💻 Execução Local

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/SINTONIA-CTRL.git
cd SINTONIA-CTRL

# Instalar dependências
pip install -r requirements.txt

# Executar
streamlit run streamlit_app.py
```

---

## 📦 Dependências

```
streamlit >= 1.36.0
numpy >= 1.24.0
matplotlib >= 3.7.0
scipy >= 1.11.0
plotly >= 5.18.0
sympy >= 1.12
control >= 0.9.0
pandas >= 2.0.0
```

---

## ✍️ Autoria

| | |
|---|---|
| **Autor** | Marcus V A Fernandes |
| **E-mail** | marcus.fernandes@ifrn.edu.br |
| **Vínculo** | Diretoria de Indústria — IFRN Campus Natal-Central (CNAT) |
| **Versão** | v1.0 |
| **Ano** | 2026 |

---

## 📄 Licença

Este material é disponibilizado como Recurso Educacional Digital de acesso livre.
Uso, adaptação e redistribuição são permitidos para fins educacionais, com a
devida atribuição de autoria.

---

*SINTONIA — Sistemas Interativos para Teoria e Otimização no Nível de Interpretação e Aprendizagem*
