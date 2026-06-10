"""
RED — Modelagem e Sistemas Lineares
streamlit_app.py — Entrypoint principal
Streamlit >= 1.36 — st.navigation com funções Python como páginas
"""

import streamlit as st
import importlib

st.set_page_config(
    page_title="RED — Modelagem e Sistemas Lineares",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Utilitário: importa (ou recarrega) módulo e chama run() ──────────────────
def _rodar(nome_modulo: str):
    import sys
    if nome_modulo in sys.modules:
        mod = importlib.reload(sys.modules[nome_modulo])
    else:
        mod = importlib.import_module(nome_modulo)
    mod.run()

# ── Funções de página ────────────────────────────────────────────────────────
def pagina_sinais():  _rodar("modulos.sinais_e_sistemas_lineares")
def pagina_laplace(): _rodar("modulos.transformada_de_laplace")
def pagina_ord1():    _rodar("modulos.dinamica_sistemas_ordem_1")
def pagina_ord2():    _rodar("modulos.dinamica_sistemas_ordem_2")
def pagina_mf():      _rodar("modulos.realimentacao_malha_fechada")
def pagina_lgr():     _rodar("modulos.realimentacao_lgr")
def pagina_estab():   _rodar("modulos.estabilidade_realimentacao")
def pagina_bode():    _rodar("modulos.resposta_frequencia")
def pagina_nyquist(): _rodar("modulos.criterio_nyquist")
def pagina_ss():      _rodar("modulos.espaco_de_estados")

# ── Definição das páginas com url_path explícito ─────────────────────────────
PG_HOME   = st.Page(lambda: _home(),         title="Página Inicial",              icon="📘", default=True, url_path="home")
PG_SINAIS = st.Page(pagina_sinais,           title="Sinais e Sistemas Lineares",  icon="📡", url_path="sinais")
PG_LAP    = st.Page(pagina_laplace,          title="Transformada de Laplace",     icon="🌀", url_path="laplace")
PG_ORD1   = st.Page(pagina_ord1,             title="Sistemas de Ordem 1",         icon="📈", url_path="ordem1")
PG_ORD2   = st.Page(pagina_ord2,             title="Sistemas de Ordem 2",         icon="📊", url_path="ordem2")
PG_MF     = st.Page(pagina_mf,               title="Malha Fechada — Ordem 1 e 2", icon="🔄", url_path="malha-fechada")
PG_LGR    = st.Page(pagina_lgr,              title="Perturbação e LGR",           icon="📍", url_path="lgr")
PG_ESTAB  = st.Page(pagina_estab,            title="Critério de Routh-Hurwitz",   icon="⚖️", url_path="estabilidade")
PG_BODE   = st.Page(pagina_bode,             title="Diagramas de Bode",           icon="📉", url_path="bode")
PG_NYQ    = st.Page(pagina_nyquist,          title="Critério de Nyquist",         icon="🔁", url_path="nyquist")
PG_SS     = st.Page(pagina_ss,               title="Análise no Espaço de Estados", icon="🧮", url_path="estados")

# ── Navegação ─────────────────────────────────────────────────────────────────
_nav = st.navigation(
    {
        "🏠 Início":                   [PG_HOME],
        "📡 Sinais e Sistemas":        [PG_SINAIS],
        "🌀 Transformada de Laplace":  [PG_LAP],
        "📈 Dinâmica no Tempo":        [PG_ORD1, PG_ORD2],
        "🔄 Análise com Realimentação":[PG_MF, PG_LGR],
        "⚖️ Estabilidade":             [PG_ESTAB],
        "📉 Resposta em Frequência":   [PG_BODE, PG_NYQ],
        "🧮 Espaço de Estados":        [PG_SS],
    },
    position="sidebar",
    expanded=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem; font-weight: 800; line-height: 1.18; margin: 0 0 0.5rem;
}
.hero-sub  { font-size: 1.02rem; opacity: .72; max-width: 640px; margin-bottom: .55rem; }
.meta-line { font-size: .82rem; opacity: .50; margin-top: .4rem; }
.red-badge {
    display: inline-block;
    background: linear-gradient(135deg,#3d8ef0 0%,#6c47ff 100%);
    color: #fff; font-size: .7rem; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px; margin-right: .5rem; vertical-align: middle;
}
.stat-row {
    display: flex; gap: 2.5rem; margin: 1.4rem 0 2rem; padding: 1rem 0;
    border-top: 1px solid rgba(128,128,128,.15);
    border-bottom: 1px solid rgba(128,128,128,.15);
}
.stat-item { text-align: center; }
.stat-num   { font-size: 1.7rem; font-weight: 700; color: #3d8ef0; }
.stat-label { font-size: .68rem; text-transform: uppercase; letter-spacing: .09em; opacity: .48; }

.mod-card {
    border: 1.5px solid rgba(128,128,128,.18); border-radius: 14px;
    padding: 1.05rem 1.15rem .85rem;
    transition: border-color .18s, box-shadow .18s, transform .12s;
}
.mod-card:hover {
    border-color: #3d8ef0;
    box-shadow: 0 4px 18px rgba(61,142,240,.13);
    transform: translateY(-2px);
}
.mod-num  { font-size: .64rem; font-weight: 700; letter-spacing: .12em;
            text-transform: uppercase; opacity: .38; margin-bottom: .3rem; }
.mod-icon { font-size: 1.4rem; margin-bottom: .2rem; display: block; }
.mod-title { font-size: .95rem; font-weight: 700; margin-bottom: .15rem; }
.mod-sub   { font-size: .75rem; opacity: .48; font-style: italic; margin-bottom: .3rem; }
.mod-desc  { font-size: .79rem; opacity: .62; line-height: 1.55; margin-bottom: .5rem; }
.tag {
    display: inline-block; font-size: .67rem; padding: 2px 7px;
    border-radius: 4px; background: rgba(61,142,240,.10);
    color: #3d8ef0; margin: 2px 2px 0 0; font-weight: 500;
}
.exp-group-title {
    font-size: .8rem; font-weight: 700; opacity: .6;
    margin: .85rem 0 .2rem; letter-spacing: .03em;
}
.page-footer {
    margin-top: 3rem; padding: 1.2rem 0 .5rem;
    border-top: 1px solid rgba(128,128,128,.14);
    text-align: center; font-size: .79rem; opacity: .48; line-height: 1.9;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEÚDO DA HOME
# ═══════════════════════════════════════════════════════════════════════════════
def _home():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero">
  <h1>📘 Modelagem e<br>Sistemas Lineares</h1>
  <p class="hero-sub">
    <span class="red-badge">RED</span>
    Recurso Educacional Digital — material didático interativo com exploradores
    de parâmetros, simulações numéricas e fórmulas LaTeX para o curso de
    Engenharia de Energia do IFRN-CNAT.
  </p>
  <p class="meta-line">
    🎓 IFRN — CNAT &nbsp;·&nbsp; 🏛️ Engenharia de Energia &nbsp;·&nbsp;
    👤 Marcus V A Fernandes &nbsp;·&nbsp;
    ✉️ marcus.fernandes@ifrn.edu.br &nbsp;·&nbsp; v1.0 · 2026
  </p>
</div>
<div class="stat-row">
  <div class="stat-item"><div class="stat-num">7</div><div class="stat-label">Módulos</div></div>
  <div class="stat-item"><div class="stat-num">10</div><div class="stat-label">Submódulos</div></div>
  <div class="stat-item"><div class="stat-num">34</div><div class="stat-label">Exploradores</div></div>
  <div class="stat-item"><div class="stat-num">60+</div><div class="stat-label">Figuras</div></div>
  <div class="stat-item"><div class="stat-num">100%</div><div class="stat-label">Online</div></div>
</div>
""", unsafe_allow_html=True)

    # ── Sobre ─────────────────────────────────────────────────────────────────
    st.markdown("### 📖 Sobre este RED")
    st.markdown("""
Este **Recurso Educacional Digital** cobre a disciplina *Modelagem e Sistemas Lineares*
do curso de Engenharia de Energia do IFRN-CNAT, com módulos progressivos — de fundamentos
de sinais até representação em espaço de estados — com ênfase em compreensão visual e
exploração paramétrica. Cada módulo combina **teoria** com equações e exemplos analíticos,
**figuras** geradas numericamente e **exploradores interativos** com controles deslizantes e
campos de entrada para observar o efeito de parâmetros em tempo real, sem necessidade de
reexecução.
""")

    # ── Índice completo ───────────────────────────────────────────────────────
    with st.expander("📋 Índice geral com acesso direto", expanded=False):
        st.caption("Clique em qualquer item para acessar diretamente o conteúdo.")
        st.markdown("---")

        # ── Módulo 1 ──────────────────────────────────────────────────────────
        st.markdown("#### 📡 1 · Sinais e Sistemas Lineares")
        st.page_link(PG_SINAIS, label="Ir para o módulo →", icon="📡")
        st.markdown("""
- **1.1** Definição de sinal e sistema — grandezas contínuas e discretas
- **1.2** Sistemas Lineares e Invariantes no Tempo (LCIT)
- **1.3** Operações com sinais: deslocamento temporal, inversão, escalonamento
- **1.4** Propriedades: linearidade, invariância no tempo, causalidade, memória, estabilidade
- **1.5** Resposta ao impulso e representação por convolução: $y(t)=h(t)*u(t)$
- **1.6** Diagrama de blocos: conexão em série, paralelo e realimentação
- **1.7** Relação entre diagrama de blocos e função de transferência
- 🎛️ Explorador: operações com sinais — deslocamento, escala, inversão
""")
        st.markdown("---")

        # ── Módulo 2 ──────────────────────────────────────────────────────────
        st.markdown("#### 🌀 2 · Transformada de Laplace")
        st.page_link(PG_LAP, label="Ir para o módulo →", icon="🌀")
        st.markdown("""
- **2.1** Definição bilateral e unilateral — integral de Bromwich
- **2.2** Tabela de pares: impulso, degrau, rampa, exponencial, senoide, cosseno
- **2.3** Propriedades: linearidade, deslocamento temporal e em frequência, escalamento
- **2.4** Propriedades: derivação, integração no tempo; teoremas do valor inicial e final
- **2.5** Convolução no tempo ↔ multiplicação em $s$
- **2.6** Função de transferência $H(s)=Y(s)/U(s)$ — polos, zeros e ganho
- **2.7** Estabilidade via posição dos polos no plano $s$
- **2.8** Realizações: forma direta, cascata, paralela e atraso de transporte (Padé)
- **2.9** Sistemas não-lineares e linearização por série de Taylor
- **2.10** Expansão em frações parciais: raízes reais distintas, complexas conjugadas e repetidas
- 🎛️ Explorador: linearização por expansão de Taylor (ordem 1 a 5)
""")
        st.markdown("---")

        # ── Módulo 3 ──────────────────────────────────────────────────────────
        st.markdown("#### 📈 3 · Dinâmica no Domínio do Tempo")
        st.markdown("##### 3.1 · Sistemas de Ordem 1")
        st.page_link(PG_ORD1, label="Ir para o módulo →", icon="📈")
        st.markdown("""
- **3.1.1** Função de transferência: grau relativo, ganho DC $k/a$, constante de tempo $\\tau=1/a$
- **3.1.2** Resposta ao degrau: componente forçada (regime permanente) e natural (transitório)
- **3.1.3** Especificações temporais: $y(\\infty)$, $T_r$, $T_s$ — identificação experimental
- **3.1.4** Exemplos físicos: circuito RL, circuito RC, massa-amortecedor, inércia rotacional, sistema térmico e hidráulico
- **3.1.5** Efeito do zero: fase mínima, cancelamento polo-zero e fase não-mínima
- **3.1.6** Sistemas com polo no SPD (instável) e polo na origem (integrador)
- **3.1.7** Polo de malha aberta vs. polo de malha fechada com realimentação proporcional
- 🎛️ Explorador 1: resposta ao degrau — controles deslizantes $k$, $a$, $k_r$
- 🎛️ Explorador 2: plano $s$ + degrau — controles deslizantes $k$ e $a$
- 🎛️ Explorador 3: sistema com zero — controles deslizantes $k$, $a$, $b$
- 🎛️ Explorador 4: sistema instável — velocidade de divergência
- 🎛️ Explorador 5: polo MA vs. polo MF — integrador com realimentação
""")
        st.markdown("##### 3.2 · Sistemas de Ordem 2")
        st.page_link(PG_ORD2, label="Ir para o módulo →", icon="📊")
        st.markdown("""
- **3.2.1** Forma canônica: $\\xi$ (amortecimento), $\\omega_n$ (frequência natural), $k$ (ganho)
- **3.2.2** Polos complexos conjugados: $s_{1,2}=-\\xi\\omega_n\\pm j\\omega_n\\sqrt{1-\\xi^2}$
- **3.2.3** Regimes: subamortecido ($0<\\xi<1$), criticamente amortecido ($\\xi=1$), sobreamortecido ($\\xi>1$), oscilatório puro ($\\xi=0$), instável ($\\xi<0$)
- **3.2.4** Especificações: ultrapassagem $UP(\\%)$, instante de pico $T_p$, tempo de subida $T_r$, tempo de acomodação $T_s$ — fórmulas analíticas
- **3.2.5** Localização dos polos: circunferência de raio $\\omega_n$, ângulo $\\theta=\\arccos(\\xi)$
- **3.2.6** Parametrização direta por $\\sigma=\\xi\\omega_n$ e $\\omega_d=\\omega_n\\sqrt{1-\\xi^2}$
- **3.2.7** Exemplos físicos: massa-mola-amortecedor e circuito RLC série
- **3.2.8** Efeito de polo adicional e zero adicional (fase mínima e não-mínima)
- **3.2.9** Sistema oscilatório: $H(s)=k/(s^2+k)$, polos em $\\pm j\\sqrt{k}$
- 🎛️ Explorador 1: regimes de amortecimento — controles deslizantes $\\xi$, $\\omega_n$
- 🎛️ Explorador 2: plano $s$ + especificações de desempenho
- 🎛️ Explorador 3: efeito de $\\xi$, $\\omega_n$ e $k$ na resposta
- 🎛️ Explorador 4: parâmetros $\\sigma$ e $\\omega_d$ — polos complexos diretos
- 🎛️ Explorador 5: polo e zero adicionais com ativação/desativação independente
- 🎛️ Explorador 6: sistema oscilatório — controle deslizante $k$
""")
        st.markdown("---")

        # ── Módulo 4 ──────────────────────────────────────────────────────────
        st.markdown("#### 🔄 4 · Análise de Sistemas com Realimentação")
        st.markdown("##### 4.1 · Malha Fechada — Ordem 1 e 2")
        st.page_link(PG_MF, label="Ir para o módulo →", icon="🔄")
        st.markdown("""
- **4.1.1** Estrutura de malha fechada: $H_{MF}(s)=G(s)/[1+G(s)]$ — diagrama de blocos completo
- **4.1.2** Tipo do sistema ($\\nu$): número de polos na origem de $G(s)$
- **4.1.3** Constantes de erro estático: posição $K_p$, velocidade $K_v$, aceleração $K_a$
- **4.1.4** Erro em regime permanente: degrau $e_{rp}=k_r/(1+K_p)$, rampa $k_r/K_v$, parábola $k_r/K_a$
- **4.1.5** Planta 1ª ordem tipo 0: polo MF em $s=-(a+k)$, $\\tau_{MF}=1/(a+k)$, $e_{rp}=a/(a+k)$
- **4.1.6** Planta 1ª ordem tipo 1: sistema MF de 2ª ordem com $\\omega_n=\\sqrt{k}$, $\\xi=a/(2\\sqrt{k})$
- **4.1.7** Planta 2ª ordem tipo 0: $\\omega_n^{MF}=\\omega_n\\sqrt{1+k}$, $\\xi^{MF}=\\xi/\\sqrt{1+k}$ — compromisso erro × amortecimento
- **4.1.8** Planta 2ª ordem tipo 1: MF de 3ª ordem, $e_{rp}(\\text{rampa})=1/k$
- 🎛️ Explorador 1: tipo do sistema, tipo de entrada e ganho $k$
- 🎛️ Explorador 2: planta 1ª ordem — entrada, $k$ e atraso puro
- 🎛️ Explorador 3: planta 2ª ordem — entrada, $k$ e atraso puro
""")
        st.markdown("##### 4.2 · Perturbação e Lugar Geométrico das Raízes")
        st.page_link(PG_LGR, label="Ir para o módulo →", icon="📍")
        st.markdown("""
- **4.2.1** Plantas de ordem superior (3ª ordem tipo 1): ganho crítico $k_{crit}=a_1 a_2(a_1+a_2)$ via Routh
- **4.2.2** Perturbação $D(s)$ na entrada da planta: superposição $H_R(s)$ e $H_D(s)$
- **4.2.3** Erro de perturbação em regime permanente: $e_{rp,D}=-1/(a+k)$
- **4.2.4** Estratégias de rejeição: ganho alto, integrador em $C(s)$, realimentação de estado
- **4.2.5** Lugar Geométrico das Raízes (LGR): definição, equação característica $D(s)+kN(s)=0$
- **4.2.6** Regras de construção: partida/chegada, assíntotas, eixo real, cruzamento imaginário
- **4.2.7** Quadro 4×4 — 16 sistemas (graus $n=1$ a $4$, quatro configurações de zeros)
- **4.2.8** Interpretação do LGR para projeto de controladores
- 🎛️ Explorador 1: estabilidade de ordem superior — $k$ e atraso
- 🎛️ Explorador 2: seguimento de referência vs. rejeição de perturbação
- 🎛️ Explorador 3: LGR interativo — insira $N(s)/D(s)$ e $k_{max}$
""")
        st.markdown("---")

        # ── Módulo 5 ──────────────────────────────────────────────────────────
        st.markdown("#### ⚖️ 5 · Estabilidade de Sistemas com Realimentação")
        st.page_link(PG_ESTAB, label="Ir para o módulo →", icon="⚖️")
        st.markdown("""
- **5.1** Conceitos de equilíbrio: estável (assintótico), marginalmente estável e instável
- **5.2** Estabilidade BIBO vs. assintótica (Lyapunov) — condições para sistemas LTI
- **5.3** Classificação pela posição dos polos: SPE, eixo imaginário (simples e repetido), SPD
- **5.4** Critério de Routh-Hurwitz: construção da tabela, interpretação da 1ª coluna
- **5.5** Caso especial 1: zero isolado na 1ª coluna — substituição por $\\varepsilon\\to0^+$
- **5.6** Caso especial 2: linha inteira de zeros — polinômio auxiliar $P(s)$ e $P'(s)$
- **5.7** Exemplo 1: $D_{MF}(s)=s^3+10s^2+31s+1030$ — 2 raízes no SPD
- **5.8** Exemplo 2: zero na primeira coluna — $s^5+2s^4+3s^3+6s^2+5s+3$
- **5.9** Exemplo 3: linha de zeros — $s^5+7s^4+6s^3+42s^2+8s+56$ (marginalmente estável)
- **5.10** Exemplo 4: região de estabilidade — $G(s)=1/[s(s+7)(s+11)]$, $k_{crit}=1386$
- 🎛️ Explorador 1: polos MF em função de $k$ — 🟢 estável / 🟡 marginal / 🔴 instável
- 🎛️ Explorador 2: região de estabilidade no plano $(k,\\,a_2)$ com ponto marcável
""")
        st.markdown("---")

        # ── Módulo 6 ──────────────────────────────────────────────────────────
        st.markdown("#### 📉 6 · Resposta em Frequência de Sistemas")
        st.markdown("##### 6.1 · Diagramas de Bode")
        st.page_link(PG_BODE, label="Ir para o módulo →", icon="📉")
        st.markdown("""
- **6.1.1** Resposta senoidal em regime permanente: $y_{rp}(t)=A|H(j\\omega)|\\sin(\\omega t+\\angle H(j\\omega))$
- **6.1.2** Representação em dB: $|H(j\\omega)|_{dB}=20\\log_{10}|H(j\\omega)|$
- **6.1.3** Fator elementar 1 — $s$: +20 dB/déc, fase +90° constante
- **6.1.4** Fator elementar 2 — $1/s$: −20 dB/déc, fase −90° constante
- **6.1.5** Fator elementar 3 — $(s+a)$: patamar $20\\log a$, +20 dB/déc após $\\omega=a$, fase 0° a +90°
- **6.1.6** Fator elementar 4 — $1/(s+a)$: patamar $-20\\log a$, −20 dB/déc após $\\omega=a$, fase 0° a −90°
- **6.1.7** Fator elementar 5 — $(s+b)/(s+a)$: lead ($b>a$) ou lag ($b<a$)
- **6.1.8** Fator elementar 6 — $1/[(s+a_1)(s+a_2)]$: −40 dB/déc, fase 0° a −180°
- **6.1.9** Frequência de corte $\\omega_c$ (−3 dB) e banda passante $\\omega_{BW}$ vs. $\\xi$
- **6.1.10** Margem de fase $\\phi_m=180°+\\angle G(j\\omega_{gc})$ e margem de ganho $G_m=-|G(j\\omega_{pc})|_{dB}$
- **6.1.11** Bode de sistemas de 1ª ordem: assíntotas, $\\omega_c=a$, fase de 0° a −90°
- **6.1.12** Bode de sistemas de 2ª ordem: pico de ressonância $\\omega_r=\\omega_n\\sqrt{1-2\\xi^2}$, $M_r=k/(2\\xi\\sqrt{1-\\xi^2})$
- **6.1.13** Diagrama de Nichols: ponto crítico $(−180°,\\,0\\,\\text{dB})$, leitura de margens
- **6.1.14** Filtros Butterworth: passa-baixa, passa-alta, passa-faixa, rejeita-faixa — $n\\times20$ dB/déc
- 🎛️ Explorador 1–6: um explorador por fator elementar (tabs individuais com controle deslizante)
- 🎛️ Explorador 7: margens $\\phi_m$ e $G_m$ — insira $N(s)/D(s)$
- 🎛️ Explorador 8: Bode 1ª ordem — controles deslizantes $k$ e $a$
- 🎛️ Explorador 9: Bode 2ª ordem — controles deslizantes $k$, $\\xi$, $\\omega_n$
- 🎛️ Explorador 10: Nichols — múltiplos valores de $\\xi$ (seleção múltipla)
- 🎛️ Explorador 11: filtros Butterworth — tipo e ordem (1–10)
- 🎛️ Explorador 12: explorador geral de Bode com $\\omega_c$, $\\phi_m$, $G_m$ automáticos
""")
        st.markdown("##### 6.2 · Critério de Nyquist")
        st.page_link(PG_NYQ, label="Ir para o módulo →", icon="🔁")
        st.markdown("""
- **6.2.1** Mapeamento de contornos e Princípio do Argumento: $N=Z-P$
- **6.2.2** Contribuições de zeros e polos no ângulo total percorrido
- **6.2.3** Contorno de Nyquist no plano $s$: sentido horário, envolve todo o SPD
- **6.2.4** Diagrama completo: curva principal ($\\omega>0$), conjugada ($\\omega<0$) e fechamento
- **6.2.5** Comportamento por tipo do sistema ($\\omega\\to0$): tipo 0 → finito; tipo 1 → $\\infty\\angle{-90°}$
- **6.2.6** Critério: $Z=N+P=0$ para estabilidade; $Z>0$ → $Z$ polos no SPD
- **6.2.7** Tabela de casos práticos: $P=0$ e $P>0$
- **6.2.8** Desvio em polos sobre o eixo imaginário: semicírculo $\\varepsilon\\to0$, convenção padrão
- **6.2.9** Efeito por tipo de polo: simples $1/s$ (180°), duplo $1/s^2$ (360°), par $\\pm j\\omega_0$
- **6.2.10** Margem de fase $\\phi_m=180°+\\angle G(j\\omega_{gc})H(j\\omega_{gc})$ no diagrama de Nyquist
- **6.2.11** Margem de ganho $G_m=1/|G(j\\omega_{pc})H(j\\omega_{pc})|$ — distância até $(-1,j0)$
- **6.2.12** Comparação sincronizada Nyquist × Bode × LGR com marcos de fase coloridos
- 🎛️ Explorador 1: efeito do ganho $K$ — curva escala 🟢 estável / 🟡 marginal / 🔴 instável
- 🎛️ Explorador 2: comparação sincronizada Nyquist × Bode × LGR
- 🎛️ Explorador 3: Nyquist interativo — diagnóstico automático $P$, $N$, $Z$, $\\phi_m$, $G_m$
""")
        st.markdown("---")

        # ── Módulo 7 ──────────────────────────────────────────────────────────
        st.markdown("#### 🧮 7 · Análise de Sistemas no Espaço de Estados")
        st.page_link(PG_SS, label="Ir para o módulo →", icon="🧮")
        st.markdown("""
- **7.1** Descrição interna vs. externa: equivalência quando controlável e observável
- **7.2** Equações de estado: $\\dot{\\mathbf{x}}=A\\mathbf{x}+B\\mathbf{u}$, $\\mathbf{y}=C\\mathbf{x}+D\\mathbf{u}$
- **7.3** Dimensões das matrizes $A\\,(n\\times n)$, $B\\,(n\\times p)$, $C\\,(q\\times n)$, $D\\,(q\\times p)$
- **7.4** Diagrama de blocos: integradores, realimentação de estado, transmissão direta
- **7.5** Realização CCF (forma canônica do controlador): última linha de $A$ = coef. do denominador
- **7.6** Realização OCF (forma canônica do observador): transposta da CCF
- **7.7** Realização em cascata e paralela (frações parciais) — forma modal (diagonal)
- **7.8** Solução via Laplace: $\\mathbf{X}(s)=(sI-A)^{-1}\\mathbf{x}(0)+(sI-A)^{-1}B\\mathbf{U}(s)$
- **7.9** Decomposição: resposta de entrada nula (zero-input) + estado nulo (zero-state)
- **7.10** Função de transferência: $H(s)=C(sI-A)^{-1}B+D$
- **7.11** Matriz de transição $e^{At}$: série de potências, propriedades, cálculo via Cayley-Hamilton
- **7.12** Transformação de similaridade $\\bar{A}=PAP^{-1}$: invariância de autovalores e FT
- **7.13** Controlabilidade: matriz $W_c=[B\\;AB\\;\\cdots\\;A^{n-1}B]$, critério $\\text{rank}(W_c)=n$
- **7.14** Observabilidade: matriz $W_o=[C^T\\;(CA)^T\\;\\cdots\\;(CA^{n-1})^T]^T$, critério $\\text{rank}(W_o)=n$
- **7.15** Cancelamento polo-zero: modos não controláveis ou não observáveis ocultos na FT
- 🎛️ Explorador 1: insira $A,B,C,D$ — resposta ao degrau, impulso, trajetória no espaço de estados
- 🎛️ Explorador 2: conversor $H(s)\\to(A,B,C,D)$ — insira $N(s)/D(s)$
- 🎛️ Explorador 3: conversor $(A,B,C,D)\\to H(s)$ — insira matrizes
""")

    # ── Cards de módulos ──────────────────────────────────────────────────────
    st.markdown("### 🗂️ Módulos do curso")
    st.caption("Clique em **Abrir** para acessar o módulo ou use o menu lateral.")

    CARDS = [
        ("MOD 01","📡","Sinais e Sistemas Lineares","",
         "Fundamentos LTI: superposição, causalidade, estabilidade BIBO, "
         "convolução e diagramas de blocos em série, paralelo e realimentação.",
         ["LCIT","Convolução","Diagramas de blocos","Superposição"], PG_SINAIS),
        ("MOD 02","🌀","Transformada de Laplace","",
         "Tabela de pares, propriedades, frações parciais e função de transferência. "
         "Realizações de sistemas e linearização por série de Taylor.",
         ["Laplace","Frações parciais","Linearização","FT"], PG_LAP),
        ("MOD 03","📈","Dinâmica no Domínio do Tempo","Sistemas de Ordem 1",
         "Resposta ao degrau: $y(\\infty)$, $\\tau$, $T_r$, $T_s$. Polo/zero e fase mínima. "
         "Exemplos: RL, RC, massa-amortecedor, inércia, térmico, hidráulico.",
         ["Ordem 1","Degrau","Polo/Zero","Fase mínima"], PG_ORD1),
        ("MOD 03","📊","Dinâmica no Domínio do Tempo","Sistemas de Ordem 2",
         "Coeficiente $\\xi$ e frequência $\\omega_n$. Especificações $UP\\%$, $T_p$, $T_r$, $T_s$. "
         "Polos e zeros adicionais com ativação independente. Sistema oscilatório.",
         ["Ordem 2","Amortecimento","UP%","Polos complexos"], PG_ORD2),
        ("MOD 04","🔄","Análise com Realimentação","Malha Fechada — Ordem 1 e 2",
         "$H_{MF}=G/(1+G)$. Tipo do sistema e constantes $K_p$, $K_v$, $K_a$. "
         "Efeito de $k$ nos polos de MF e compromisso erro × amortecimento.",
         ["Malha fechada","Erro","Tipo do sistema","Polos MF"], PG_MF),
        ("MOD 04","📍","Análise com Realimentação","Perturbação e LGR",
         "Plantas de ordem superior, rejeição de perturbação e $e_{rp,D}=-1/(a+k)$. "
         "LGR com quadro 4×4 de 16 sistemas e explorador interativo.",
         ["LGR","Perturbação","Ordem superior","$k_{crit}$"], PG_LGR),
        ("MOD 05","⚖️","Estabilidade com Realimentação","Critério de Routh-Hurwitz",
         "Critério de Routh-Hurwitz com casos $\\varepsilon$ e polinômio auxiliar. "
         "4 exemplos numéricos e região de estabilidade no plano $(k,a_2)$.",
         ["Routh-Hurwitz","$k_{crit}$","Região de estabilidade","Marginal"], PG_ESTAB),
        ("MOD 06","📉","Resposta em Frequência","Diagramas de Bode",
         "6 fatores elementares, margens $\\phi_m$ e $G_m$, pico de ressonância. "
         "Diagrama de Nichols e filtros Butterworth (PB, PA, PF, RF).",
         ["Bode","Margens","Nichols","Filtros"], PG_BODE),
        ("MOD 06","🔁","Resposta em Frequência","Critério de Nyquist",
         "Princípio do Argumento $N=Z-P$, contorno de Nyquist, desvio em polos. "
         "Comparação sincronizada Nyquist × Bode × LGR com marcos de fase.",
         ["Nyquist","$N=Z-P$","Margens","Mapeamento"], PG_NYQ),
        ("MOD 07","🧮","Espaço de Estados","",
         "Equações $\\dot{x}=Ax+Bu$. Realizações CCF/OCF, matriz $e^{At}$ via Cayley-Hamilton. "
         "Controlabilidade $W_c$, observabilidade $W_o$ e conversores FT↔SS.",
         ["Estado","Controlabilidade","Observabilidade","$e^{At}$"], PG_SS),
    ]

    for row_start in range(0, len(CARDS), 3):
        cols = st.columns(3, gap="medium")
        for ci, card in enumerate(CARDS[row_start:row_start+3]):
            num, icon, title, sub, desc, tags, pg = card
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
            sub_html  = f'<div class="mod-sub">↳ {sub}</div>' if sub else ""
            with cols[ci]:
                st.markdown(f"""
<div class="mod-card">
  <div class="mod-num">{num}</div>
  <span class="mod-icon">{icon}</span>
  <div class="mod-title">{title}</div>
  {sub_html}
  <div class="mod-desc">{desc}</div>
  <div style="margin-top:.4rem">{tags_html}</div>
</div>""", unsafe_allow_html=True)
                st.page_link(pg, label=f"Abrir — {title}" + (f" · {sub}" if sub else ""),
                             use_container_width=True)

    # ── Exploradores ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎛️ Exploradores interativos")
    st.markdown(
        "Os **exploradores** são o diferencial deste RED — controles deslizantes, "
        "menus de seleção e campos de entrada com atualização em tempo real, "
        "sem necessidade de reexecução."
    )

    EXP_GROUPS = [
        ("📡 Sinais e Sistemas", PG_SINAIS, [
            "Operações com sinais — deslocamento, escala, inversão",
        ]),
        ("🌀 Transformada de Laplace", PG_LAP, [
            "Linearização por Taylor — ordem 1 a 5",
        ]),
        ("📈 Dinâmica — Ordem 1", PG_ORD1, [
            "Resposta ao degrau — controles $k$, $a$, $k_r$",
            "Plano $s$ + degrau — controles $k$ e $a$",
            "Sistema com zero — controles $k$, $a$, $b$",
            "Sistema instável — velocidade de divergência",
            "Polo MA vs. MF — integrador com realimentação",
        ]),
        ("📊 Dinâmica — Ordem 2", PG_ORD2, [
            "Regimes de amortecimento — $\\xi$, $\\omega_n$",
            "Plano $s$ + especificações de desempenho",
            "Efeito de $\\xi$, $\\omega_n$ e $k$ na resposta",
            "Parâmetros $\\sigma$ e $\\omega_d$ diretos",
            "Polo e zero adicionais — ativação independente",
            "Sistema oscilatório — controle deslizante $k$",
        ]),
        ("🔄 Realimentação — MF", PG_MF, [
            "Tipo do sistema, entrada e ganho $k$",
            "Planta 1ª ordem — entrada, $k$ e atraso",
            "Planta 2ª ordem — entrada, $k$ e atraso",
        ]),
        ("📍 Realimentação — LGR", PG_LGR, [
            "Estabilidade de ordem superior — $k$ e atraso",
            "Seguimento vs. rejeição de perturbação",
            "LGR interativo — $N(s)/D(s)$ e $k_{max}$",
        ]),
        ("⚖️ Estabilidade", PG_ESTAB, [
            "Polos MF vs. ganho $k$ — 🟢/🟡/🔴",
            "Região de estabilidade — plano $(k,\\,a_2)$",
        ]),
        ("📉 Bode", PG_BODE, [
            "6 fatores elementares — tabs individuais",
            "Margens $\\phi_m$ e $G_m$ — insira $N(s)/D(s)$",
            "Bode 1ª ordem — controles $k$ e $a$",
            "Bode 2ª ordem — controles $k$, $\\xi$, $\\omega_n$",
            "Nichols — múltiplos $\\xi$ com seleção",
            "Filtros Butterworth — tipo e ordem (1–10)",
            "Explorador geral — $\\omega_c$, $\\phi_m$, $G_m$ automáticos",
        ]),
        ("🔁 Nyquist", PG_NYQ, [
            "Efeito do ganho $K$ — 🟢/🟡/🔴",
            "Nyquist × Bode × LGR sincronizados",
            "Nyquist interativo — $P$, $N$, $Z$, $\\phi_m$, $G_m$",
        ]),
        ("🧮 Espaço de Estados", PG_SS, [
            "Explorador SS — insira $A,B,C,D$",
            "Conversor $H(s)\\to SS$",
            "Conversor $SS\\to H(s)$",
        ]),
    ]

    half = (len(EXP_GROUPS) + 1) // 2
    col_l, col_r = st.columns(2)
    for col, grupo in zip([col_l, col_r], [EXP_GROUPS[:half], EXP_GROUPS[half:]]):
        with col:
            for gtitle, pg, items in grupo:
                st.markdown(f'<p class="exp-group-title">{gtitle}</p>',
                            unsafe_allow_html=True)
                for item in items:
                    st.page_link(pg, label=item, use_container_width=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="page-footer">
  Modelagem e Sistemas Lineares &nbsp;·&nbsp;
  Engenharia de Energia &nbsp;·&nbsp; CNAT — IFRN<br>
  Autor: Marcus V A Fernandes &nbsp;·&nbsp;
  marcus.fernandes@ifrn.edu.br &nbsp;·&nbsp; v1.0 · 2026
</div>
""", unsafe_allow_html=True)


# ── Executar ──────────────────────────────────────────────────────────────────
_nav.run()
