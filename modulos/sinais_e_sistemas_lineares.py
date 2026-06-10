"""
📡 Sinais e Sistemas Lineares
Disciplina: Modelagem e Sistemas Lineares
Curso: Engenharia de Energia
Instituição: IFRN — Campus Natal-Central (CNAT)
Autor: Marcus V A Fernandes · marcus.fernandes@ifrn.edu.br · v1.0
"""
        
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import lti, step as sc_step
import plotly.graph_objects as go
import warnings
        
def run():
        
        warnings.filterwarnings("ignore")
        
        # ── Configuração da Página ────────────────────────────────────────────────────
        st.set_page_config(
            page_title="Sinais e Sistemas Lineares",
            page_icon="📡",
            layout="wide",
        )
        
        # ── Estilo global de figuras ──────────────────────────────────────────────────
        plt.rcParams.update({
            "figure.dpi": 120,
            "axes.grid": True,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "lines.linewidth": 1.6,
            "font.family": "serif",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "grid.alpha": 0.3,
        })
        
        # ── Paleta de cores ───────────────────────────────────────────────────────────
        COR = dict(
            sinal    = "royalblue",
            saida    = "crimson",
            natural  = "seagreen",
            instavel = "crimson",
            marginal = "darkorange",
            impulso  = "#222222",
            degrau   = "royalblue",
            rampa    = "crimson",
            parabola = "seagreen",
            senoide  = "#7d3c98",
            bloco_f  = "#FFFFFF",
            bloco_e  = "#000000",
        )
        
        # ── Helpers de plotagem ───────────────────────────────────────────────────────
        def estilo(ax, xlabel="t", ylabel="Amplitude"):
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        def bloco(ax, x, y, w, h, txt, fc=None, ec=None, fs=8):
            fc = fc or COR["bloco_f"]; ec = ec or COR["bloco_e"]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.04", facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
            ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4)
        
        def ponto(ax, x, y):
            ax.plot(x, y, "o", color=COR["bloco_e"], ms=4, zorder=5)
        
        # ── CSS responsivo — injetado uma única vez ───────────────────────────────────
        # No desktop (>768 px) cada figura fica centralizada com largura controlada.
        # Em telas estreitas (≤768 px, mobile/vertical) a figura ocupa 100% da tela.
        st.markdown("""
        <style>
        /* Wrapper responsivo para figuras matplotlib */
        .fig-wrap {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .fig-wrap > div {
            width: 100%;          /* mobile-first: 100 % */
        }
        
        /* Desktop: limitar pela variável --fw definida inline no elemento */
        @media (min-width: 769px) {
            .fig-wrap > div {
                width: var(--fw, 65%);
                max-width: var(--fw, 65%);
            }
        }
        
        /* Garantir que a imagem dentro do st.pyplot preencha o wrapper */
        .fig-wrap img,
        .fig-wrap [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # ── Helper de exibição responsivo ────────────────────────────────────────────
        def show_fig(fig, width_frac=0.65):
            """
            Renderiza `fig` de forma responsiva:
              • Desktop (>768 px): figura centralizada com largura = width_frac × 100 %
              • Mobile  (≤768 px): figura expande para 100 % da tela automaticamente
            width_frac: 0.0–1.0
            """
            import io, base64
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.get_dpi())
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            pct = f"{int(width_frac * 100)}%"
            st.markdown(
                f'<div class="fig-wrap">'
                f'<div style="--fw:{pct}">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;display:block;"/>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        
        # ── Helpers de álgebra de blocos ──────────────────────────────────────────────
        def _blk(ax, x, y, w, h, t, fs=8):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x-w/2, y-h/2), w, h, boxstyle="round,pad=0.04",
                facecolor=COR["bloco_f"], edgecolor=COR["bloco_e"], lw=1.3, zorder=3))
            ax.text(x, y, t, ha="center", va="center", fontsize=fs, zorder=4)
        
        def _a(ax, x1, y1, x2, y2):
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=COR["bloco_e"],
                                lw=1.3, shrinkA=0, shrinkB=0), zorder=2)
        
        def _l(ax, x1, y1, x2, y2):
            ax.plot([x1,x2],[y1,y2], color=COR["bloco_e"], lw=1.3, zorder=2)
        
        def _s(ax, x, y, r=0.22):
            ax.add_patch(mpatches.Circle((x,y), r, fc="white",
                          ec=COR["bloco_e"], lw=1.3, zorder=3))
        
        def _d(ax, x, y):
            ax.plot(x, y, "o", color=COR["bloco_e"], ms=4, zorder=5)
        
        def _t(ax, x, y, t, fs=8, ha="center"):
            ax.text(x, y, t, ha=ha, va="center", fontsize=fs, zorder=5)
        
        def _pm2(ax, x, y, s, c):
            ax.text(x, y, s, ha="center", va="center", fontsize=9,
                    color=c, fontweight="bold", zorder=6)
        
        def _setup(ax, titulo):
            ax.set_aspect("equal")
            ax.set_xlim(0, 11); ax.set_ylim(-1.4, 1.6); ax.axis("off")
            ax.set_title(titulo, fontsize=9, fontweight="bold", pad=4)
            ax.axvline(5.5, color="#CCCCCC", lw=1, ls="--", zorder=0)
            ax.text(5.5, 0.0, "⟺", ha="center", va="center", fontsize=14, color="dimgray")
            ax.text(0.1, 1.52, "Original:",    ha="left", va="top", fontsize=7, color="dimgray")
            ax.text(5.7, 1.52, "Equivalente:", ha="left", va="top", fontsize=7, color="dimgray")
        
        wb = 1.3; hb = 0.40; r = 0.22
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CABEÇALHO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.title("📡 Sinais e Sistemas Lineares")
        st.caption("Modelagem e Sistemas Lineares · Engenharia de Energia · IFRN-CNAT · Marcus V A Fernandes")
        st.markdown("---")
        
        # ── Índice ────────────────────────────────────────────────────────────────────
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown("""
        **[1. Definições Fundamentais](#1-defini-es-fundamentais)**
        - Conceito de sinal e sistema
        - Sistemas SISO e MIMO
        - Diagrama de sistema genérico
        
        **[2. Tamanho de um Sinal](#2-tamanho-de-um-sinal)**
        - 2.1 Energia do sinal
        - 2.2 Potência média do sinal
        - Comparativo: sinal de energia vs. sinal de potência
        
        **[3. Natureza e Periodicidade](#3-natureza-e-periodicidade)**
        - 3.1 Natureza do sinal (analógico/digital, contínuo/discreto)
        - 3.2 Periodicidade e período fundamental
        - 3.3 Causalidade (causal, não-causal, anti-causal) e determinismo
        
        **[4. Operações sobre Sinais](#4-opera-es-sobre-sinais)**
        - 4.1 Deslocamento temporal (atraso e avanço)
        - 4.2 Escalamento temporal (compressão e expansão)
        - 4.3 Reversão temporal
        - 🎛️ Explorador interativo de operações
        
        **[5. Funções Pares e Ímpares](#5-fun-es-pares-e-mpares)**
        - Decomposição par/ímpar de um sinal arbitrário
        - Propriedades do produto
        
        **[6. Funções Singulares](#6-fun-es-singulares)**
        - 6.1 Degrau unitário $u(t)$
        - 6.2 Impulso de Dirac $\\delta(t)$ — propriedade de amostragem
        - 6.3 Rampa $r(t)$ e parábola $p(t)$
        
        **[7. Funções Exponenciais e Frequência Complexa](#7-fun-es-exponenciais-e-frequ-ncia-complexa)**
        - Variável de frequência complexa $s = \\sigma + j\\omega$
        - Casos de $e^{st}$: constante, monotônica, senoidal, amortecida
        - Senoide amortecida e envelope exponencial
        
        **[8. Sistemas — Definições e Classificação](#8-sistemas-defini-es-e-classifica-o)**
        - Classificação: linear, LIT, causal, estável (BIBO), SISO/MIMO
        - 8.1 Decomposição da resposta (entrada nula + estado nulo)
        - 8.2 Sistemas LIT e Transformada de Laplace
        
        **[9. Princípio da Superposição](#9-princ-pio-da-superposi-o)**
        - Aditividade e homogeneidade
        - Verificação numérica
        
        **[10. Estabilidade](#10-estabilidade)**
        - Estável, instável e marginalmente estável
        - Localização dos pólos no plano complexo
        
        **[11. Sistemas de Controle](#11-sistemas-de-controle)**
        - 11.1 Malha aberta
        - 11.2 Malha fechada (realimentação negativa)
        - 11.3 Objetivos: resposta transitória, estacionária e estabilidade
        - 11.4 Sinais de entrada padrão (impulso, degrau, rampa, parábola, senoidal)
        
        **[12. Diagramas de Blocos](#12-diagramas-de-blocos)**
        - 12.1 Cascata (série): $G_{eq} = G_1 \\cdot G_2$
        - 12.2 Paralelo: $G_{eq} = G_1 \\pm G_2$
        - 12.3 Realimentação: $T(s) = G / (1 \\pm GF)$
        - 12.4 Álgebra de blocos — regras de movimento de pontos de soma e ramificação
        - Exemplo numérico: redução de malha fechada
        
        **[13. Referências](#13-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1 — DEFINIÇÕES FUNDAMENTAIS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Definições Fundamentais")
        
        st.markdown(r"""
        Um **sinal** é uma representação de dados ou informação. Os sinais são processados por **sistemas**,
        que os modificam ou extraem informações.
        
        Um **sistema** transforma sinais de **entrada** em sinais de **saída**:
        
        $$x_1(t),\ldots,x_j(t) \;\xrightarrow{\text{Sistema}}\; y_1(t),\ldots,y_k(t)$$
        
        Sistemas com múltiplas entradas e saídas são chamados **MIMO** (*Multiple-Input Multiple-Output*);
        com uma entrada e uma saída, **SISO**.
        
        Um sistema pode ser implementado em **hardware** (componentes físicos) ou **software** (algoritmo).
        """)
        
        # Diagrama MIMO — figura quadrada/compacta, centrada em 45 % da tela
        fig_mimo, ax_mimo = plt.subplots(figsize=(4.5, 2.2))
        ax_mimo.set_xlim(-0.5, 10.5); ax_mimo.set_ylim(-0.3, 4.3)
        ax_mimo.set_aspect("equal"); ax_mimo.axis("off")
        ax_mimo.set_title(r"Sistema MIMO — $j$ entradas, $k$ saídas",
                          fontsize=10, fontweight="bold", pad=7)
        ax_mimo.add_patch(mpatches.FancyBboxPatch(
            (2.9, 0.28), 2.2, 3.44, boxstyle="round,pad=0.06",
            facecolor=COR["bloco_f"], edgecolor=COR["bloco_e"], lw=1.6, zorder=3))
        ax_mimo.text(4.0, 2.0, "Sistema", ha="center", va="center", fontsize=10, zorder=4)
        for lbl, y in [(r"$x_1(t)$", 3.5), (r"$x_2(t)$", 2.5), (r"$x_j(t)$", 0.5)]:
            ax_mimo.text(0.3, y, lbl, ha="center", va="center", fontsize=9)
            ax_mimo.annotate("", xy=(2.9, y), xytext=(1.0, y),
                arrowprops=dict(arrowstyle="->", color=COR["bloco_e"], lw=1.4, shrinkA=0, shrinkB=0))
        ax_mimo.text(0.3, 1.5, r"$\vdots$", ha="center", va="center", fontsize=13)
        for lbl, y in [(r"$y_1(t)$", 3.5), (r"$y_2(t)$", 2.5), (r"$y_k(t)$", 0.5)]:
            ax_mimo.annotate("", xy=(7.1, y), xytext=(5.1, y),
                arrowprops=dict(arrowstyle="->", color=COR["bloco_e"], lw=1.4, shrinkA=0, shrinkB=0))
            ax_mimo.text(7.7, y, lbl, ha="center", va="center", fontsize=9)
        ax_mimo.text(9.7, 1.5, r"$\vdots$", ha="center", va="center", fontsize=13)
        plt.tight_layout()
        show_fig(fig_mimo, 0.45)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2 — TAMANHO DE UM SINAL
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Tamanho de um Sinal")
        
        st.markdown(r"""
        ### 2.1 Energia
        
        Sinais que **tendem a zero** quando $t \to \pm\infty$ possuem energia finita:
        
        $$E_x = \int_{-\infty}^{+\infty} |x(t)|^2 \, dt < \infty$$
        
        ### 2.2 Potência
        
        Sinais que **persistem no tempo** possuem potência média finita:
        
        $$P_x = \lim_{T \to \infty} \frac{1}{T} \int_{-T/2}^{+T/2} |x(t)|^2 \, dt < \infty$$
        
        > **Nota:** um sinal não pode ser simultaneamente de energia e de potência. Sinais periódicos são
        > de potência ($E_x = \infty$, $P_x < \infty$); pulsos isolados são de energia ($E_x < \infty$, $P_x = 0$).
        """)
        
        fig_tam, axs_tam = plt.subplots(1, 2, figsize=(6, 2.4))
        t1 = np.linspace(0, 10, 300)
        y_lim = np.exp(-(t1 - 3)**2 / 0.5)
        axs_tam[0].plot(t1, y_lim, color=COR["sinal"], lw=1.6)
        axs_tam[0].set_title(r"Energia  ($E_x < \infty$, $P_x = 0$)", fontsize=8)
        axs_tam[0].fill_between(t1, y_lim, alpha=0.15, color=COR["sinal"])
        axs_tam[0].set_yticks([]); estilo(axs_tam[0])
        t2 = np.linspace(0, 10, 1000)
        y_pot = (0.3*np.sin(2*np.pi*0.4*t2) + 0.2*np.cos(2*np.pi*1.1*t2)
                 + 0.15*np.sin(2*np.pi*2.3*t2))
        axs_tam[1].plot(t2, y_pot, color=COR["sinal"], lw=1.6)
        axs_tam[1].set_title(r"Potência  ($P_x < \infty$, $E_x = \infty$)", fontsize=8)
        axs_tam[1].set_yticks([]); estilo(axs_tam[1])
        plt.suptitle("Classificação pelo Tamanho do Sinal", fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig_tam, 0.60)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3 — NATUREZA, PERIODICIDADE E CAUSALIDADE
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Natureza e Periodicidade")
        
        st.markdown(r"""
        ### 3.1 Natureza do Sinal
        
        Dois eixos classificam um sinal: o **domínio** (eixo do tempo) e a **amplitude**.
        
        | | **Analógico** (amplitude contínua) | **Digital** (amplitude quantizada) |
        |---|---|---|
        | **Contínuo** ($t \in \mathbb{R}$) | $x(t)$: grandeza física | $x(t)$: onda quadrada |
        | **Discreto** ($n \in \mathbb{Z}$) | $x[n]$: amostras reais | $x[n]$: sequência de bits |
        
        ### 3.2 Periodicidade
        
        Um sinal é **periódico** se existe $T_0 > 0$ tal que:
        
        $$x(t) = x(t + T_0), \quad \forall\, t$$
        
        O menor $T_0$ que satisfaz essa condição é o **período fundamental**.
        """)
        
        # Natureza — grade 2×2 compacta
        fig_nat, axs_nat = plt.subplots(2, 2, figsize=(5.5, 3.5))
        fig_nat.suptitle("Natureza dos Sinais", fontsize=9, fontweight="bold")
        xc = np.linspace(0, 4, 400); xd = np.arange(0, 4.25, 0.25)
        sq_c = np.where((np.floor(xc) % 2) == 0, 1.0, -1.0)
        sq_d = np.where((np.floor(xd) % 2) == 0, 1.0, -1.0)
        axs_nat[0,0].set_title("Analógico", fontsize=8, fontweight="bold")
        axs_nat[0,1].set_title("Digital",   fontsize=8, fontweight="bold")
        axs_nat[0,0].set_ylabel("Contínuo", fontsize=8, fontweight="bold", labelpad=6)
        axs_nat[1,0].set_ylabel("Discreto", fontsize=8, fontweight="bold", labelpad=6)
        axs_nat[0,0].plot(xc, xc**2/8, "k", lw=1.6)
        axs_nat[0,0].set_xlabel("$t$", fontsize=7); axs_nat[0,0].set_yticks([])
        axs_nat[0,0].spines[["right","top"]].set_visible(False)
        axs_nat[0,1].plot(xc, sq_c, "k", lw=1.6)
        axs_nat[0,1].set_xlabel("$t$", fontsize=7); axs_nat[0,1].set_yticks([])
        axs_nat[0,1].spines[["right","top"]].set_visible(False)
        axs_nat[1,0].stem(xd, xd/4, linefmt="k-", markerfmt="ko", basefmt="k-")
        axs_nat[1,0].set_xlabel("$n$", fontsize=7)
        axs_nat[1,0].spines[["right","top"]].set_visible(False)
        axs_nat[1,1].stem(xd, sq_d, linefmt="k-", markerfmt="ko", basefmt="k-")
        axs_nat[1,1].set_xlabel("$n$", fontsize=7)
        axs_nat[1,1].spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig_nat, 0.48)
        
        # Periodicidade — dois painéis lado a lado
        fig_per, axs_per = plt.subplots(1, 2, figsize=(6, 2.4))
        fig_per.suptitle("Periodicidade de Sinais", fontsize=9, fontweight="bold")
        t_per = np.linspace(-5, 5, 1000)
        y_per = np.sin(np.pi*t_per) + 0.35*np.sin(3*np.pi*t_per)
        axs_per[0].plot(t_per, y_per, color=COR["sinal"], lw=1.6)
        axs_per[0].set_title("Periódico", fontsize=8)
        axs_per[0].axvline(-1, color="gray", ls="--", lw=1)
        axs_per[0].axvline( 1, color="gray", ls="--", lw=1)
        axs_per[0].annotate("", xy=(1,-1.45), xytext=(-1,-1.45),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2))
        axs_per[0].text(0, -1.75, "$T_0$", ha="center", fontsize=9)
        axs_per[0].set_yticks([]); axs_per[0].set_xlabel("$t$", fontsize=8)
        axs_per[0].spines[["right","top"]].set_visible(False); axs_per[0].set_xlim(-4,4)
        t_aper = np.linspace(0, 8, 600)
        y_aper = np.exp(-0.4*t_aper)*np.sin(3*t_aper)
        axs_per[1].plot(t_aper, y_aper, color="coral", lw=1.6)
        axs_per[1].set_title("Aperiódico", fontsize=8)
        axs_per[1].set_yticks([]); axs_per[1].set_xlabel("$t$", fontsize=8)
        axs_per[1].spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig_per, 0.55)
        
        st.markdown(r"""
        ### 3.3 Causalidade e Determinismo
        
        | Tipo | Condição | Exemplo |
        |---|---|---|
        | **Causal** | $x(t) = 0$ para $t < 0$ | Resposta a uma excitação |
        | **Não-causal** | $x(t) \neq 0$ para $t < 0$ | Sinal modelado matematicamente |
        | **Anti-causal** | $x(t) = 0$ para $t \geq 0$ | Componente passada de um sistema |
        
        Um sinal **determinístico** é completamente conhecido matematicamente; um sinal **aleatório**
        não pode ser predito com precisão e é descrito por métodos estatísticos.
        """)
        
        # Causalidade — três painéis lado a lado
        t_caus = np.linspace(-4, 4, 1000)
        fig_caus, axs_caus = plt.subplots(1, 3, figsize=(6.5, 2.2))
        fig_caus.suptitle("Causalidade de Sinais", fontsize=9, fontweight="bold")
        y_c = np.where(t_caus >= 0, np.exp(-t_caus)*np.sin(4*t_caus), 0)
        axs_caus[0].plot(t_caus, y_c, color=COR["sinal"], lw=1.6)
        axs_caus[0].set_title("Causal\n$x(t)=0,\\;t<0$", fontsize=8)
        axs_caus[0].axvline(0, color="k", lw=0.8, ls="--")
        axs_caus[0].fill_betweenx([-1.1,1.1], -4, 0, alpha=0.06, color="gray")
        axs_caus[0].set_yticks([]); axs_caus[0].set_xlabel("$t$", fontsize=8)
        axs_caus[0].spines[["right","top"]].set_visible(False); axs_caus[0].set_ylim(-1.1,1.1)
        y_nc = np.exp(-0.3*np.abs(t_caus))*np.sin(3*t_caus)
        axs_caus[1].plot(t_caus, y_nc, color="seagreen", lw=1.6)
        axs_caus[1].set_title("Não-causal\n$x(t)\\neq0,\\;t<0$", fontsize=8)
        axs_caus[1].axvline(0, color="k", lw=0.8, ls="--")
        axs_caus[1].set_yticks([]); axs_caus[1].set_xlabel("$t$", fontsize=8)
        axs_caus[1].spines[["right","top"]].set_visible(False); axs_caus[1].set_ylim(-1.1,1.1)
        y_ac = np.where(t_caus < 0, np.exp(0.5*t_caus)*np.cos(4*t_caus), 0)
        axs_caus[2].plot(t_caus, y_ac, color="darkorange", lw=1.6)
        axs_caus[2].set_title("Anti-causal\n$x(t)=0,\\;t\\geq0$", fontsize=8)
        axs_caus[2].axvline(0, color="k", lw=0.8, ls="--")
        axs_caus[2].fill_betweenx([-1.1,1.1], 0, 4, alpha=0.06, color="gray")
        axs_caus[2].set_yticks([]); axs_caus[2].set_xlabel("$t$", fontsize=8)
        axs_caus[2].spines[["right","top"]].set_visible(False); axs_caus[2].set_ylim(-1.1,1.1)
        plt.tight_layout()
        show_fig(fig_caus, 0.62)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4 — OPERAÇÕES SOBRE SINAIS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Operações sobre Sinais")
        
        st.markdown(r"""
        | Operação | Expressão | Efeito |
        |---|:---|---|
        | Atraso | $\phi(t) = x(t-T),\quad T>0$ | Desloca para a direita |
        | Avanço | $\phi(t) = x(t+T),\quad T>0$ | Desloca para a esquerda |
        | Compressão | $\phi(t) = x(at),\quad a>1$ | Acelera o sinal |
        | Expansão | $\phi(t) = x(t/a),\quad a>1$ | Desacelera o sinal |
        | Reversão temporal | $\phi(t) = x(-t)$ | Espelhamento no eixo do tempo |
        | Inversão de amplitude | $\phi(t) = -x(t)$ | Espelhamento no eixo da amplitude |
        """)
        
        # 4.1 / 4.2 / 4.3 lado a lado com texto + figura em colunas ──────────────────
        st.markdown("### 4.1 Deslocamento Temporal")
        st.markdown(r"""
        Substituir $t$ por $t-T$ desloca o sinal $T$ unidades para a direita (atraso);
        substituir por $t+T$, para a esquerda (avanço).
        """)
        
        def sinal_demo(x, pp=2, ph=1, ur=0.2, dr=0.5):
            return ph * np.where(x < pp,
                np.exp(-(x-pp)**2/(2*ur**2)),
                np.exp(-(x-pp)**2/(2*dr**2)))
        
        T_desloc = 1.5
        x_d = np.linspace(-1, 6, 500)
        
        # Deslocamento: figura vertical contida em coluna estreita
        fig_desl, axs_desl = plt.subplots(3, 1, figsize=(4.0, 4.0), sharex=True)
        fig_desl.suptitle("Deslocamento Temporal", fontsize=9, fontweight="bold")
        for ax, (sig, tit, cor) in zip(axs_desl, [
            (sinal_demo(x_d),            "$x(t)$",            "k"),
            (sinal_demo(x_d - T_desloc), "$x(t-T)$ — atraso", COR["degrau"]),
            (sinal_demo(x_d + T_desloc), "$x(t+T)$ — avanço", COR["saida"]),
        ]):
            ax.plot(x_d, sig, color=cor, lw=1.6)
            ax.axvline(0, color="k", lw=0.7)
            ax.spines[["right","top","left"]].set_visible(False)
            ax.set_yticks([])
            ax.text(0.02, 0.88, tit, transform=ax.transAxes, fontsize=8, va="top")
        axs_desl[-1].set_xlabel("$t$", fontsize=8)
        plt.tight_layout()
        show_fig(fig_desl, 0.40)
        
        st.markdown("### 4.2 Escalamento Temporal")
        st.markdown(r"""
        $$\phi(t) = x(at),\quad a>1 \quad\Rightarrow\quad \text{compressão (mais rápido)}$$
        
        $$\phi(t) = x\!\left(\tfrac{t}{a}\right),\quad a>1 \quad\Rightarrow\quad \text{expansão (mais lento)}$$
        """)
        
        def tri(x, s1, s2):
            return np.where(x < 0, s1*x+1, s2*x+1)
        
        x_e = np.linspace(-3, 3, 500)
        fig_esc, axs_esc = plt.subplots(3, 1, figsize=(4.0, 4.0), sharex=True)
        fig_esc.suptitle("Escalamento Temporal", fontsize=9, fontweight="bold")
        for ax, (sig, tit) in zip(axs_esc, [
            (tri(x_e, 1.5, -1),    "$x(t)$ — original"),
            (tri(x_e, 3,   -2),    "$x(t/2)$ — expansão ($a=2$)"),
            (tri(x_e, 0.75, -0.5), "$x(2t)$ — compressão ($a=2$)"),
        ]):
            ax.plot(x_e, np.clip(sig, -0.3, 1.3), "k", lw=1.6)
            ax.axvline(0, color="k", lw=0.7); ax.axhline(0, color="k", lw=0.7)
            ax.spines[["right","top","left"]].set_visible(False)
            ax.set_yticks([]); ax.set_ylim(-0.45, 1.45)
            ax.text(0.02, 0.95, tit, transform=ax.transAxes, fontsize=8, va="top")
        axs_esc[-1].set_xlabel("$t$", fontsize=8)
        plt.tight_layout()
        show_fig(fig_esc, 0.40)
        
        st.markdown("### 4.3 Reversão Temporal")
        st.markdown(r"""
        $$\phi(t) = x(-t)$$
        
        Espelha o sinal em torno do eixo $t=0$. Não confundir com a
        **inversão de amplitude** $\phi(t)=-x(t)$, que espelha em torno do eixo da amplitude.
        """)
        
        x_r = np.linspace(-6, 6, 1200)
        def f_rev(x):
            y = np.zeros_like(x)
            y = np.where((x > -5) & (x <= 0), x+5, y)
            y = np.where((x >= 0) & (x < 2),  2.5*np.sqrt(np.maximum(x,0))-3.5, y)
            return y
        
        fig_rev, axs_rev = plt.subplots(1, 2, figsize=(5.5, 2.4))
        for ax, (xarg, tit) in zip(axs_rev, [(x_r, "$x(t)$"), (-x_r, "$x(-t)$ — reversão")]):
            ax.plot(x_r, f_rev(xarg), "k", lw=1.6)
            ax.set_title(tit, fontsize=8)
            ax.axvline(0, color="k", lw=0.7); ax.axhline(0, color="k", lw=0.7)
            ax.spines[["right","top"]].set_visible(False)
            ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(-4, 6)
        plt.suptitle("Reversão Temporal", fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig_rev, 0.50)
        
        # ── EXPLORADOR INTERATIVO ─────────────────────────────────────────────────────
        st.markdown("### 🎛️ Explorador de Operações sobre Sinais")
        col1, col2 = st.columns([1, 2])
        with col1:
            op = st.selectbox("Operação", ["Deslocamento temporal", "Escalamento temporal", "Reversão temporal"])
            if op == "Deslocamento temporal":
                t0 = st.slider("Atraso $t_0$", -4.0, 4.0, 1.5, 0.1)
            elif op == "Escalamento temporal":
                a = st.slider("Fator $a$", 0.25, 4.0, 2.0, 0.25)
            sinal_base = st.selectbox("Sinal base", ["Pulso retangular", "Rampa", "Senoide"])
        
        t = np.linspace(-6, 6, 1000)
        
        def make_signal(t, tipo):
            if tipo == "Pulso retangular":
                return ((t >= -1) & (t <= 1)).astype(float)
            elif tipo == "Rampa":
                return np.where(t >= 0, t*np.exp(-t), 0.0)
            else:
                return np.where(np.abs(t) <= 4, np.sin(2*t)*np.exp(-0.2*np.abs(t)), 0.0)
        
        x_orig = make_signal(t, sinal_base)
        if op == "Deslocamento temporal":
            x_mod = make_signal(t - t0, sinal_base);  label_mod = f"x(t − {t0:.1f})"
        elif op == "Escalamento temporal":
            x_mod = make_signal(a * t, sinal_base);   label_mod = f"x({a:.2f}·t)"
        else:
            x_mod = make_signal(-t, sinal_base);       label_mod = "x(−t)"
        
        with col2:
            fig_exp = go.Figure()
            fig_exp.add_trace(go.Scatter(x=t, y=x_orig, name="x(t) original",
                                         line=dict(color="#3d8ef0", width=2, dash="dash")))
            fig_exp.add_trace(go.Scatter(x=t, y=x_mod, name=label_mod,
                                         line=dict(color="#ef4444", width=2.5)))
            fig_exp.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20),
                                  legend=dict(orientation="h", y=1.1),
                                  xaxis_title="t", yaxis_title="Amplitude")
            st.plotly_chart(fig_exp, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5 — FUNÇÕES PARES E ÍMPARES
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Funções Pares e Ímpares")
        
        st.markdown(r"""
        - **Par:** $x_e(t) = x_e(-t)$ — simetria em relação ao eixo de amplitude.
        - **Ímpar:** $x_o(t) = -x_o(-t)$ — antissimetria em relação ao eixo de amplitude.
        
        Qualquer sinal pode ser decomposto em componentes par e ímpar:
        
        $$x_e(t) = \frac{x(t)+x(-t)}{2} \qquad x_o(t) = \frac{x(t)-x(-t)}{2}$$
        
        **Propriedades do produto:** par × par = par; ímpar × ímpar = par; par × ímpar = ímpar.
        """)
        
        t_pi = np.linspace(-4, 4, 800)
        x_pi = np.where(t_pi >= 0, np.exp(-t_pi), 0.3*np.exp(t_pi))
        xe_pi = (x_pi + x_pi[::-1]) / 2
        xo_pi = (x_pi - x_pi[::-1]) / 2
        
        fig_pi, axs_pi = plt.subplots(1, 3, figsize=(6.5, 2.4), sharex=True, sharey=True)
        for ax, (y, tit, cor) in zip(axs_pi, [
            (x_pi,  "$x(t)$ — original", COR["sinal"]),
            (xe_pi, "$x_e(t)$ — par",   "seagreen"),
            (xo_pi, "$x_o(t)$ — ímpar", COR["saida"]),
        ]):
            ax.plot(t_pi, y, color=cor, lw=1.6)
            ax.set_title(tit, fontsize=8)
            ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
            ax.spines[["right","top"]].set_visible(False)
            ax.set_yticks([]); ax.set_xlabel("$t$", fontsize=7)
        plt.suptitle("Componentes Par e Ímpar", fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig_pi, 0.60)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6 — FUNÇÕES SINGULARES
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Funções Singulares")
        
        st.markdown(r"""
        Funções utilizadas na análise de sistemas. Cada uma possui um **ponto singular na origem**
        e é nula nos demais pontos. São relacionadas entre si por integração ou diferenciação sucessiva.
        
        ### 6.1 Degrau Unitário $u(t)$
        
        $$u(t) = \begin{cases} 1 & t \ge 0 \\ 0 & t < 0 \end{cases}$$
        
        Representa o instante em que um sinal ou fonte é **ligado**. Útil para descrever sinais causais.
        
        ### 6.2 Impulso de Dirac $\delta(t)$
        
        Definido pela sua **propriedade de amostragem**:
        
        $$\int_{-\infty}^{+\infty} \phi(t)\,\delta(t-T)\,dt = \phi(T)$$
        
        Pode ser entendido como o limite de um pulso retangular de largura $\epsilon \to 0$ e altura $1/\epsilon$,
        mantendo área unitária. Relação com o degrau:
        
        $$\delta(t) = \frac{du(t)}{dt} \qquad u(t) = \int_{-\infty}^{t} \delta(\tau)\,d\tau$$
        
        ### 6.3 Rampa $r(t)$ e Parábola $p(t)$
        
        $$r(t) = t\,u(t) \qquad p(t) = \frac{t^2}{2}\,u(t)$$
        
        Obtidas por integração sucessiva do degrau. São usadas como **entradas padrão** para avaliar o
        erro estacionário de sistemas de controle.
        """)
        
        h = 0.001; t_sg = np.arange(0, 3, h); atr = 1.0; eps = 2*h
        degrau   = np.where(t_sg >= atr, 1.0, 0.0)
        impulso  = np.where(np.abs(t_sg - atr) <= eps, 1.0/(2*eps), 0.0)
        rampa    = np.where(t_sg >= atr, t_sg - atr, 0.0)
        parabola = np.where(t_sg >= atr, 0.5*(t_sg - atr)**2, 0.0)
        
        # Funções singulares: 4 painéis empilhados — coluna estreita
        fig_sg, axs_sg = plt.subplots(4, 1, figsize=(4.0, 5.5), sharex=True)
        fig_sg.suptitle("Funções Singulares (atraso $t_0=1$)", fontsize=9, fontweight="bold")
        for ax, (y, cor, lb) in zip(axs_sg, [
            (impulso,  COR["impulso"],  r"Impulso $\delta(t-1)$"),
            (degrau,   COR["degrau"],   r"Degrau $u(t-1)$"),
            (rampa,    COR["rampa"],    r"Rampa $r(t-1)$"),
            (parabola, COR["parabola"], r"Parábola $p(t-1)$"),
        ]):
            ax.plot(t_sg, y, color=cor, lw=1.6, label=lb)
            ax.set_ylabel("Amp.", fontsize=7)
            ax.legend(loc="upper left", fontsize=7)
            ax.spines[["right","top"]].set_visible(False)
        axs_sg[-1].set_xlabel("Tempo (s)", fontsize=8)
        plt.tight_layout()
        show_fig(fig_sg, 0.38)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7 — FUNÇÕES EXPONENCIAIS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("7. Funções Exponenciais e Frequência Complexa")
        
        st.markdown(r"""
        A variável $s = \sigma + j\omega$ (**frequência complexa**) unifica todos os tipos de sinal na forma:
        
        $$e^{st} = e^{(\sigma + j\omega)t} = e^{\sigma t}(\cos\omega t + j\sin\omega t)$$
        
        | Condição | Sinal resultante |
        |:---|---|
        | $s = 0$ | Constante |
        | $\omega = 0,\ \sigma > 0$ | Exponencial crescente $e^{\sigma t}$ |
        | $\omega = 0,\ \sigma < 0$ | Exponencial decrescente $e^{-|\sigma|t}$ |
        | $\sigma = 0$ | Senoide pura $\cos\omega t$ |
        | $\sigma < 0$ | Senoide amortecida $e^{\sigma t}\cos\omega t$ |
        | $\sigma > 0$ | Senoide com envelope crescente |
        
        **Fórmula de Euler:** $e^{j\omega t} = \cos\omega t + j\sin\omega t$
        """)
        
        t_ex = np.arange(-5, 5, 0.001)
        exps = {
            "constante":  np.ones_like(t_ex),
            "mono_pos":   np.exp( 0.25*t_ex),
            "mono_neg":   np.exp(-0.50*t_ex),
            "senoide":    np.cos(5*t_ex),
            "amortecida": np.exp(-0.50*t_ex)*np.cos(5*t_ex),
            "crescente":  np.exp( 0.50*t_ex)*np.cos(5*t_ex),
        }
        
        # Grade 2×3 — ocupa ~75% da tela
        fig_ex, axs_ex = plt.subplots(2, 3, figsize=(7.5, 3.8), sharex=True)
        fig_ex.suptitle(r"Funções Exponenciais — casos de $e^{st}$", fontsize=9, fontweight="bold")
        itens = [
            ("constante",  "seagreen",    r"Constante ($s=0$)",                   axs_ex[0,0]),
            ("mono_pos",   COR["degrau"], r"Monotônica crescente ($\sigma>0$)",   axs_ex[0,1]),
            ("mono_neg",   COR["saida"],  r"Monotônica decrescente ($\sigma<0$)", axs_ex[0,2]),
            ("senoide",    COR["impulso"],r"Senoide pura ($\sigma=0$)",           axs_ex[1,0]),
            ("amortecida", COR["degrau"], r"Senoide amortecida ($\sigma<0$)",     axs_ex[1,1]),
            ("crescente",  COR["saida"],  r"Senoide crescente ($\sigma>0$)",      axs_ex[1,2]),
        ]
        for nome, cor, titulo, ax in itens:
            ax.plot(t_ex, np.clip(exps[nome], -3, 3), color=cor, lw=1.6)
            ax.set_title(titulo, fontsize=7.5)
            ax.axhline(0, color="k", lw=0.5)
            ax.spines[["right","top"]].set_visible(False); ax.set_ylim(-3, 3)
        for ax in axs_ex[1]: ax.set_xlabel("Tempo (s)", fontsize=7)
        for ax in [axs_ex[0,0], axs_ex[1,0]]: ax.set_ylabel("Amplitude", fontsize=7)
        plt.tight_layout()
        show_fig(fig_ex, 0.72)
        
        # Senoide amortecida e envelope
        t_env = np.arange(-5, 5, 0.001)
        env = np.exp(-0.5*t_env); sam = env*np.cos(5*t_env)
        fig_env, ax_env = plt.subplots(figsize=(5.5, 2.6))
        ax_env.plot(t_env, env,  color="k",          lw=1.2, ls="--", label=r"Envelope $e^{-|\sigma|t}$")
        ax_env.plot(t_env, -env, color="k",          lw=1.2, ls="--")
        ax_env.plot(t_env, sam,  color=COR["degrau"],lw=1.6, label="Senoide amortecida")
        ax_env.axhline(0, color="k", lw=0.5)
        ax_env.set_title("Senoide amortecida e envelope exponencial", fontsize=9)
        ax_env.legend(fontsize=7)
        estilo(ax_env, "Tempo (s)", "Amplitude")
        plt.tight_layout()
        show_fig(fig_env, 0.55)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 8 — SISTEMAS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("8. Sistemas — Definições e Classificação")
        
        st.markdown(r"""
        O estudo de sistemas envolve três áreas: **modelagem matemática**, **análise** e **projeto**.
        
        | Classificação | Condição |
        |---|---|
        | **Linear** | Obedece ao princípio da superposição |
        | **Invariante no Tempo (LIT)** | Parâmetros constantes; atraso na entrada $\Rightarrow$ mesmo atraso na saída |
        | **Causal** | Saída em $t_0$ depende apenas de $x(t)$ para $t \le t_0$ |
        | **Sem memória** | Saída depende apenas da entrada no instante atual |
        | **Com memória (dinâmico)** | Saída depende do histórico da entrada |
        | **Estável (BIBO)** | Toda entrada limitada produz saída limitada |
        | **Tempo contínuo / discreto** | Sinais em $t \in \mathbb{R}$ ou $n \in \mathbb{Z}$ |
        | **SISO / MIMO** | Uma ou múltiplas entradas/saídas |
        
        ### 8.1 Decomposição da Resposta
        
        A resposta total de um sistema linear é a soma de duas componentes:
        
        $$\underbrace{y(t)}_{\text{total}} = \underbrace{y_{en}(t)}_{\substack{\text{entrada nula} \\ \text{(natural)}}} + \underbrace{y_{ez}(t)}_{\substack{\text{estado nulo} \\ \text{(forçada)}}}$$
        
        $y_{en}(t)$: depende apenas das **condições iniciais**. $y_{ez}(t)$: depende apenas da **entrada aplicada**.
        
        ### 8.2 Sistemas LIT
        
        Sistemas **Lineares e Invariantes no Tempo** são analisados pela **Transformada de Laplace**,
        que converte equações diferenciais em equações algébricas no domínio $s$.
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 9 — SUPERPOSIÇÃO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("9. Princípio da Superposição")
        
        st.markdown(r"""
        Propriedade **exclusiva de sistemas lineares**, combinando:
        
        - **Aditividade:** $\mathcal{S}\{x_1+x_2\} = \mathcal{S}\{x_1\} + \mathcal{S}\{x_2\}$
        - **Homogeneidade:** $\mathcal{S}\{a\,x\} = a\,\mathcal{S}\{x\}$
        
        $$\mathcal{S}\{a\,x_1(t) + b\,x_2(t)\} = a\,\mathcal{S}\{x_1(t)\} + b\,\mathcal{S}\{x_2(t)\}$$
        
        Permite analisar cada componente de entrada **separadamente** e somar os resultados —
        técnica fundamental na análise de circuitos e sistemas.
        """)
        
        t_sp = np.linspace(0, 5, 800)
        x1_sp = np.sin(2*np.pi*t_sp); x2_sp = 0.5*np.cos(4*np.pi*t_sp)
        
        fig_sp, axs_sp = plt.subplots(3, 1, figsize=(5.0, 4.8), sharex=True)
        fig_sp.suptitle(r"Princípio da Superposição — $y = 2x$", fontsize=9, fontweight="bold")
        axs_sp[0].plot(t_sp, x1_sp,         COR["degrau"], lw=1.6, label="$x_1(t)$")
        axs_sp[0].plot(t_sp, x2_sp,         COR["saida"],  lw=1.6, label="$x_2(t)$")
        axs_sp[0].plot(t_sp, x1_sp+x2_sp,   "k--",         lw=1.2, label="$x_1+x_2$")
        axs_sp[0].set_ylabel("Entradas", fontsize=7); axs_sp[0].legend(fontsize=7)
        axs_sp[1].plot(t_sp, 2*x1_sp, COR["degrau"], lw=1.6, label="$y_1=2x_1$")
        axs_sp[1].plot(t_sp, 2*x2_sp, COR["saida"],  lw=1.6, label="$y_2=2x_2$")
        axs_sp[1].set_ylabel("Saídas indiv.", fontsize=7); axs_sp[1].legend(fontsize=7)
        axs_sp[2].plot(t_sp, 2*(x1_sp+x2_sp),   "k-",         lw=2.0, label="$2(x_1+x_2)$")
        axs_sp[2].plot(t_sp, 2*x1_sp+2*x2_sp,   "darkorange",  lw=1.4, ls="--", label="$y_1+y_2$")
        axs_sp[2].set_ylabel("Saída total", fontsize=7)
        axs_sp[2].set_xlabel("Tempo (s)", fontsize=8); axs_sp[2].legend(fontsize=7)
        for ax in axs_sp: ax.spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig_sp, 0.48)
        
        erro_max = np.max(np.abs(2*(x1_sp+x2_sp) - (2*x1_sp+2*x2_sp)))
        st.info(f"Erro máximo entre as duas formas de calcular: {erro_max:.2e}  (≈ 0 → superposição válida)")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 10 — ESTABILIDADE
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("10. Estabilidade")
        
        st.markdown(r"""
        Para um sistema **LIT**, a estabilidade é determinada pelo comportamento da **resposta natural** $y_n(t)$:
        
        | Tipo | Condição | Localização dos pólos |
        |---|---|---|
        | **Estável** | $y_n(t) \to 0$ quando $t \to \infty$ | Semiplano esquerdo aberto ($\sigma < 0$) |
        | **Instável** | $y_n(t) \to \infty$ quando $t \to \infty$ | Semiplano direito ($\sigma > 0$) |
        | **Marginalmente estável** | $y_n(t)$ oscila com amplitude constante | Eixo imaginário ($\sigma = 0$) |
        
        Os **pólos** são as raízes do denominador da função de transferência $H(s)$. Para sistemas de controle,
        apenas a **resposta forçada** $y_f(t)$ permanece em regime permanente:
        
        $$y(t) = \underbrace{y_n(t)}_{\to\,0\,(\text{estável})} + y_f(t)$$
        """)
        
        t_est = np.linspace(0, 6, 4000)
        fig_est, axs_est = plt.subplots(3, 1, figsize=(5.0, 4.5), sharex=True)
        fig_est.suptitle(r"Estabilidade de Sistemas LIT — $y_n(t)$", fontsize=9, fontweight="bold")
        for ax, (y, cor, tit) in zip(axs_est, [
            (np.exp(-1.5*t_est)*np.cos(10*t_est), COR["natural"],  r"Estável — $y_n(t)\to 0$"),
            (np.exp( 0.6*t_est)*np.cos(10*t_est), COR["instavel"], r"Instável — $y_n(t)\to\infty$"),
            (np.cos(10*t_est),                     COR["marginal"], r"Marginalmente estável"),
        ]):
            ax.plot(t_est, np.clip(y, -2.5, 2.5), color=cor, lw=1.6)
            ax.axhline(0, color="k", lw=0.6, ls="--")
            ax.set_ylabel("$y_n(t)$", fontsize=7); ax.set_title(tit, fontsize=8)
            ax.spines[["right","top"]].set_visible(False); ax.set_ylim(-2.5, 2.5)
        axs_est[-1].set_xlabel("Tempo (s)", fontsize=8); axs_est[-1].set_xlim(0, 5)
        plt.tight_layout()
        show_fig(fig_est, 0.48)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 11 — SISTEMAS DE CONTROLE
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("11. Sistemas de Controle")
        
        st.markdown(r"""
        Um **sistema de controle** reúne subsistemas e processos para obter uma **resposta desejada**
        com desempenho especificado.
        
        ### 11.1 Malha Aberta
        
        A saída **não é medida** nem realimentada. O controlador opera sem correção de erro —
        simples, mas vulnerável a perturbações e incertezas do modelo.
        """)
        
        fig_ma, ax_ma = plt.subplots(figsize=(6.5, 1.6))
        ax_ma.set_xlim(-0.5, 11.5); ax_ma.set_ylim(-0.7, 0.7); ax_ma.axis("off")
        ax_ma.set_title("Sistema em Malha Aberta", fontsize=9, fontweight="bold", pad=4)
        cx = [0.9, 3.3, 5.7, 8.1]; w_b = 1.8; h_b = 0.46
        x_rd = cx[1] - w_b/2 - 1.3
        ax_ma.text(x_rd, 0, "Resposta\ndesejada", ha="center", va="center", fontsize=7.5)
        ax_ma.annotate("", xy=(cx[1]-w_b/2, 0), xytext=(x_rd+0.65, 0),
            arrowprops=dict(arrowstyle="->", color=COR["bloco_e"], lw=1.4, shrinkA=0, shrinkB=0))
        for x, lb in zip(cx[1:], ["Controlador", "Atuador", "Processo"]):
            bloco(ax_ma, x, 0, w_b, h_b, lb)
        for i in range(len(cx[1:])-1):
            ax_ma.annotate("", xy=(cx[i+2]-w_b/2, 0), xytext=(cx[i+1]+w_b/2, 0),
                arrowprops=dict(arrowstyle="->", color=COR["bloco_e"], lw=1.4, shrinkA=0, shrinkB=0))
        ax_ma.annotate("", xy=(9.8, 0), xytext=(cx[-1]+w_b/2, 0),
            arrowprops=dict(arrowstyle="->", color=COR["bloco_e"], lw=1.4, shrinkA=0, shrinkB=0))
        ax_ma.text(10.2, 0, "Resposta\nreal", ha="center", va="center", fontsize=7.5)
        plt.tight_layout()
        show_fig(fig_ma, 0.65)
        
        st.markdown(r"""
        ### 11.2 Malha Fechada (Realimentação)
        
        A saída é **medida e comparada** com a referência. O **erro** $E(t) = R(t) - Y(t)$ corrige o sistema
        continuamente, compensando perturbações e variações do processo.
        
        - **Sensor:** mede o sinal de saída e o retorna ao comparador.
        - **Erro:** diferença entre a referência $R(t)$ e a saída medida $Y(t)$.
        
        **Vantagens:** rejeição de perturbações, robustez a incertezas do modelo e redução do erro estacionário.
        """)
        
        fig_mf, ax_mf = plt.subplots(figsize=(9.0, 3.6))
        ax_mf.set_aspect("equal")
        ax_mf.set_xlim(-0.8, 14.5); ax_mf.set_ylim(-2.8, 1.6); ax_mf.axis("off")
        ax_mf.set_title("Sistema em Malha Fechada", fontsize=9, fontweight="bold", pad=4)
        
        def _arr_mf(ax, x1,y1,x2,y2,lb="",ldy=0.13):
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=COR["bloco_e"],lw=1.3,shrinkA=0,shrinkB=0))
            if lb: ax.text((x1+x2)/2,(y1+y2)/2+ldy,lb,ha="center",fontsize=7,zorder=5)
        
        def _lin_mf(ax,x1,y1,x2,y2): ax.plot([x1,x2],[y1,y2],color=COR["bloco_e"],lw=1.3,zorder=2)
        def _dot_mf(ax,x,y): ax.plot(x,y,"o",color=COR["bloco_e"],ms=5,zorder=5)
        def _som_mf(ax,x,y,r=0.28): ax.add_patch(mpatches.Circle((x,y),r,fc="white",ec=COR["bloco_e"],lw=1.4,zorder=3))
        def _pm_mf(ax,x,y,s,c): ax.text(x,y,s,ha="center",va="center",fontsize=9,color=c,fontweight="bold",zorder=6)
        
        w2=1.8; h2=0.46; rs=0.28
        xrd=-0.3;xs1=1.0;xct=3.1;xat=5.3;xsp=7.2;xpr=9.3;xdt=11.2
        yfb=-1.8;xsr=9.3;xsn=5.3
        
        ax_mf.text(xrd,0,"Resposta\ndesejada",ha="center",va="center",fontsize=7.5)
        _arr_mf(ax_mf,xrd+0.55,0,xs1-rs,0)
        _som_mf(ax_mf,xs1,0)
        _pm_mf(ax_mf,xs1-rs-0.16,+0.22,"+","seagreen"); _pm_mf(ax_mf,xs1+0.16,-rs-0.18,"−","crimson")
        _arr_mf(ax_mf,xs1+rs,0,xct-w2/2,0,"erro",ldy=0.14)
        bloco(ax_mf,xct,0,w2,h2,"Controlador")
        _arr_mf(ax_mf,xct+w2/2,0,xat-w2/2,0)
        bloco(ax_mf,xat,0,w2,h2,"Atuador")
        _arr_mf(ax_mf,xat+w2/2,0,xsp-rs,0)
        _som_mf(ax_mf,xsp,0)
        _pm_mf(ax_mf,xsp-rs-0.16,+0.22,"+","seagreen"); _pm_mf(ax_mf,xsp+0.16,rs+0.16,"+","seagreen")
        ax_mf.annotate("perturbação",xy=(xsp,rs),xytext=(xsp,1.15),
            arrowprops=dict(arrowstyle="->",color="gray",lw=1,shrinkA=0,shrinkB=0),
            fontsize=7,color="gray",ha="center",va="bottom")
        _arr_mf(ax_mf,xsp+rs,0,xpr-w2/2,0)
        bloco(ax_mf,xpr,0,w2,h2,"Processo")
        _lin_mf(ax_mf,xpr+w2/2,0,xdt,0); _dot_mf(ax_mf,xdt,0)
        _arr_mf(ax_mf,xdt,0,13.0,0)
        ax_mf.text(13.15,0,"Saída\nreal",ha="left",va="center",fontsize=7.5)
        _lin_mf(ax_mf,xdt,0,xdt,yfb); _arr_mf(ax_mf,xdt,yfb,xsr+rs,yfb)
        _som_mf(ax_mf,xsr,yfb)
        _pm_mf(ax_mf,xsr+rs+0.16,yfb+0.22,"+","seagreen"); _pm_mf(ax_mf,xsr+0.16,yfb-rs-0.18,"+","seagreen")
        ax_mf.annotate("ruído de\nmedição",xy=(xsr,yfb-rs),xytext=(xsr,yfb-1.15),
            arrowprops=dict(arrowstyle="->",color="gray",lw=1,shrinkA=0,shrinkB=0),
            fontsize=7,color="gray",ha="center",va="top")
        _arr_mf(ax_mf,xsr-rs,yfb,xsn+w2/2,yfb)
        bloco(ax_mf,xsn,yfb,w2,h2,"Sensor")
        _lin_mf(ax_mf,xsn-w2/2,yfb,xs1,yfb); _arr_mf(ax_mf,xs1,yfb,xs1,-rs)
        ax_mf.text((xdt+xsr)/2+0.3,yfb-0.28,"Saída\nmensurada",ha="center",fontsize=6.5,color="dimgray")
        plt.tight_layout()
        show_fig(fig_mf, 0.82)
        
        st.markdown(r"""
        ### 11.3 Objetivos de um Sistema de Controle
        
        **Objetivos funcionais:** amplificação de potência, controle remoto, adaptação da forma da
        entrada e compensação de perturbações.
        
        **Objetivos de análise e projeto:**
        
        | Objetivo | Descrição |
        |---|---|
        | **Resposta transitória** | Comportamento antes do regime permanente |
        | **Resposta estacionária** | Valor final da saída em relação à referência |
        | **Estabilidade** | Resposta natural tende a zero em regime |
        """)
        
        wn_tr, z_tr = 5.0, 0.5
        sys_tr = lti([wn_tr**2], [1, 2*z_tr*wn_tr, wn_tr**2])
        t_tr, y_tr = sc_step(sys_tr)
        
        fig_tr, ax_tr = plt.subplots(figsize=(5.5, 3.0))
        ax_tr.plot(t_tr, y_tr, color=COR["saida"], lw=1.8, label="Saída $y(t)$")
        ax_tr.axhline(1, color="gray", ls="--", lw=1, label="Referência")
        ax_tr.axhline(1.02, color="seagreen", ls=":", lw=1)
        ax_tr.axhline(0.98, color="seagreen", ls=":", lw=1, label="Banda ±2%")
        t_lim = 1.4
        ax_tr.axvspan(0,     t_lim,    alpha=0.07, color="royalblue")
        ax_tr.axvspan(t_lim, t_tr[-1], alpha=0.07, color="seagreen")
        y_final = float(y_tr[-1]); x_err = t_tr[-1]*0.88
        ax_tr.annotate("", xy=(x_err, y_final), xytext=(x_err, 1.0),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=1.2))
        ax_tr.text(x_err+0.06, (y_final+1.0)/2, "Erro\nestacionário",
                   fontsize=7, color="purple", ha="left", va="center")
        ax_tr.text(t_lim/2, 0.20, "Transitória", ha="center", fontsize=7.5, color="royalblue", fontweight="bold")
        ax_tr.text(t_lim+(t_tr[-1]-t_lim)*0.38, 0.20, "Estacionária", ha="center", fontsize=7.5, color="seagreen", fontweight="bold")
        ax_tr.set_xlabel("Tempo (s)", fontsize=8); ax_tr.set_ylabel("Amplitude", fontsize=8)
        ax_tr.set_title("Resposta transitória e estacionária ao degrau", fontsize=9, fontweight="bold")
        ax_tr.legend(fontsize=7, loc="lower right")
        ax_tr.set_xlim(0, t_tr[-1]); ax_tr.spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig_tr, 0.55)
        
        st.markdown(r"""
        ### 11.4 Sinais de Entrada para Análise
        
        | Entrada | Expressão | Uso principal |
        |---|---|---|
        | Impulso | $\delta(t)$ | Resposta transitória |
        | Degrau | $u(t)$ | Transitória + erro estacionário |
        | Rampa | $t\,u(t)$ | Erro de velocidade |
        | Parábola | $\dfrac{t^2}{2}u(t)$ | Erro de aceleração |
        | Senoidal | $\cos(\omega t)$ | Resposta em frequência |
        """)
        
        h_ent=0.001; t_ent=np.arange(0,4,h_ent); atr_ent=0.5; eps_ent=3*h_ent
        s_imp = np.where(np.abs(t_ent-atr_ent)<=eps_ent, 1.0/(2*eps_ent), 0.0)
        s_deg = np.where(t_ent>=atr_ent, 1.0, 0.0)
        s_ram = np.where(t_ent>=atr_ent, t_ent-atr_ent, 0.0)
        s_par = np.where(t_ent>=atr_ent, 0.5*(t_ent-atr_ent)**2, 0.0)
        s_sen = np.cos(5*np.pi*t_ent)
        
        fig_ent, axs_ent = plt.subplots(1, 5, figsize=(8.5, 2.4))
        fig_ent.suptitle("Sinais de Entrada para Análise de Desempenho", fontsize=9, fontweight="bold")
        for ax, y, nome, cor in zip(axs_ent,
            [s_imp, s_deg, s_ram, s_par, s_sen],
            [r"Impulso $\delta(t)$", r"Degrau $u(t)$", r"Rampa $r(t)$",
             r"Parábola $p(t)$", r"Senoidal"],
            [COR["impulso"],COR["degrau"],COR["rampa"],COR["parabola"],COR["senoide"]]):
            ax.plot(t_ent, np.clip(y,-1.5,3.5), color=cor, lw=1.6)
            ax.set_title(nome, fontsize=7.5)
            ax.set_xlabel("$t$", fontsize=7); ax.axhline(0, color="k", lw=0.5)
            ax.spines[["right","top"]].set_visible(False)
        axs_ent[0].set_ylabel("Amplitude", fontsize=7)
        plt.tight_layout()
        show_fig(fig_ent, 0.80)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 12 — DIAGRAMAS DE BLOCOS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("12. Diagramas de Blocos")
        
        st.markdown(r"""
        Representam sistemas como interconexão de subsistemas, cada um descrito por sua
        **função de transferência** $G(s)$ — razão entre a Transformada de Laplace da saída e da entrada,
        com condições iniciais nulas.
        
        ### 12.1 Cascata (Série)
        
        $$G_{eq}(s) = G_1(s) \cdot G_2(s) \cdots G_n(s)$$
        
        ### 12.2 Paralelo
        
        $$G_{eq}(s) = G_1(s) \pm G_2(s) \pm \cdots \pm G_n(s)$$
        
        ### 12.3 Realimentação (Malha Fechada)
        
        $$T(s) = \frac{Y(s)}{R(s)} = \frac{G(s)}{1 \pm G(s)\,F(s)}$$
        
        $G(s)$: função de transferência da malha direta; $F(s)$: da malha de realimentação.
        O sinal **$-$** no somador → **realimentação negativa** → denominador $1+G(s)F(s)$.
        O sinal **$+$** no somador → **realimentação positiva** → denominador $1-G(s)F(s)$.
        """)
        
        # Cascata
        fig_cas, ax_cas = plt.subplots(figsize=(5.5, 1.5))
        ax_cas.set_xlim(0,10); ax_cas.set_ylim(-0.6, 0.7); ax_cas.axis("off")
        ax_cas.set_title("Cascata (Série)", fontsize=9, fontweight="bold", pad=3)
        ax_cas.text(0.5, 0.98, r"$G_{eq}(s)=G_1(s)\cdot G_2(s)\cdots G_n(s)$",
                    transform=ax_cas.transAxes, ha="center", va="top", fontsize=8)
        W=1.5; H=0.42; cxc=[1.8,4.2,7.8]
        ax_cas.text(0.3, 0, "$X(s)$", ha="center", va="center", fontsize=8)
        _a(ax_cas, 0.6, 0, cxc[0]-W/2, 0)
        for x, lb in zip(cxc, ["$G_1(s)$","$G_2(s)$","$G_n(s)$"]):
            _blk(ax_cas, x, 0, W, H, lb)
        _a(ax_cas, cxc[0]+W/2, 0, cxc[1]-W/2, 0)
        _a(ax_cas, cxc[1]+W/2, 0, cxc[1]+W/2+0.5, 0)
        ax_cas.text(6.3, 0, r"$\cdots$", ha="center", va="center", fontsize=13)
        _a(ax_cas, 6.65, 0, cxc[2]-W/2, 0)
        _a(ax_cas, cxc[2]+W/2, 0, 9.4, 0)
        ax_cas.text(9.7, 0, "$Y(s)$", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        show_fig(fig_cas, 0.55)
        
        # Paralelo
        fig_par2, ax_par2 = plt.subplots(figsize=(5.5, 3.0))
        ax_par2.set_xlim(0,10); ax_par2.set_ylim(-1.8,1.8); ax_par2.axis("off")
        ax_par2.set_title(r"Paralelo:  $G_{eq}(s)=G_1(s)\pm G_2(s)\pm G_3(s)$",
                          fontsize=9, fontweight="bold", pad=3)
        W2=2.2; H2=0.44; xN=1.4; xBc=5.0; xS=8.6; ys=[1.1,0.0,-1.1]
        ax_par2.text(0.3, 0, "$X(s)$", ha="center", va="center", fontsize=8)
        _l(ax_par2,0.6,0,xN,0); ponto(ax_par2,xN,0)
        for i, y in enumerate(ys):
            _l(ax_par2,xN,0,xN,y); _a(ax_par2,xN,y,xBc-W2/2,y)
            _blk(ax_par2,xBc,y,W2,H2,f"$G_{i+1}(s)$"); _l(ax_par2,xBc+W2/2,y,xS,y)
        _l(ax_par2,xS,ys[0],xS,ys[-1]); ponto(ax_par2,xS,0)
        _a(ax_par2,xS,0,9.4,0)
        ax_par2.text(9.7,0,"$Y(s)$",ha="center",va="center",fontsize=8)
        plt.tight_layout()
        show_fig(fig_par2, 0.55)
        
        # Realimentação
        fig_real, ax_real = plt.subplots(figsize=(5.5, 2.8))
        ax_real.set_aspect("equal"); ax_real.set_xlim(0,10); ax_real.set_ylim(-1.7,0.9); ax_real.axis("off")
        ax_real.set_title(r"Realimentação:  $T(s)=\dfrac{G(s)}{1+G(s)\,F(s)}$",
                          fontsize=9, fontweight="bold", pad=3)
        rs2=0.24; Wf=2.2; Hf=0.44
        xR2=0.5;xS2=1.4;xG2=5.2;xD2=8.5;xY2=9.5;yF2=-1.1
        ax_real.text(xR2,0,"$R(s)$",ha="center",va="center",fontsize=8)
        _a(ax_real,xR2+0.3,0,xS2-rs2,0)
        ax_real.add_patch(mpatches.Circle((xS2,0),rs2,fc="white",ec=COR["bloco_e"],lw=1.3,zorder=3))
        ax_real.text(xS2-rs2-0.14,0.20,"+",fontsize=9,color="seagreen",fontweight="bold")
        ax_real.text(xS2+0.14,-rs2-0.16,"−",fontsize=9,color="crimson",fontweight="bold")
        _a(ax_real,xS2+rs2,0,xG2-Wf/2,0)
        ax_real.text((xS2+rs2+xG2-Wf/2)/2,0.14,"$E(s)$",ha="center",fontsize=7.5)
        _blk(ax_real,xG2,0,Wf,Hf,"$G(s)$"); _l(ax_real,xG2+Wf/2,0,xD2,0)
        ponto(ax_real,xD2,0); _a(ax_real,xD2,0,xY2,0)
        ax_real.text(xY2+0.3,0,"$Y(s)$",ha="center",va="center",fontsize=8)
        _l(ax_real,xD2,0,xD2,yF2); _blk(ax_real,xG2,yF2,Wf,Hf,"$F(s)$")
        _a(ax_real,xD2,yF2,xG2+Wf/2,yF2); _l(ax_real,xG2-Wf/2,yF2,xS2,yF2)
        _a(ax_real,xS2,yF2,xS2,-rs2)
        plt.tight_layout()
        show_fig(fig_real, 0.55)
        
        st.markdown(r"""
        ### 12.4 Álgebra de Blocos — Regras de Movimento
        
        | Operação | Regra de equivalência |
        |---|:---|
        | Blocos em cascata | $G_{eq} = G_1 \cdot G_2$ |
        | Blocos em paralelo | $G_{eq} = G_1 \pm G_2$ |
        | Eliminar malha de realimentação | $T(s) = G\,/\,(1 \pm GF)$ |
        | Mover ponto de soma **antes** de $G$ | Inserir $G$ no ramo movido |
        | Mover ponto de soma **depois** de $G$ | Inserir $1/G$ no ramo movido |
        | Mover ponto de ramificação **antes** de $G$ | Inserir $G$ no novo ramo |
        | Mover ponto de ramificação **depois** de $G$ | Inserir $1/G$ no novo ramo |
        """)
        
        # Álgebra 1
        fig_alg1, ax_alg1 = plt.subplots(figsize=(7.0, 3.0))
        _setup(ax_alg1, "1. Mover ponto de soma para ANTES de G(s)")
        _t(ax_alg1,0.2,0.7,"$R(s)$"); _a(ax_alg1,0.65,0.7,1.10,0.7)
        _s(ax_alg1,1.32,0.7); _pm2(ax_alg1,1.32-r-0.22,0.7+0.32,"±","black")
        _a(ax_alg1,1.54,0.7,2.05,0.7); _blk(ax_alg1,2.7,0.7,wb,hb,"$G(s)$")
        _a(ax_alg1,3.35,0.7,3.85,0.7); _t(ax_alg1,4.1,0.7,"$Y(s)$")
        _t(ax_alg1,1.32,0.08,"$X(s)$",fs=7.5); _a(ax_alg1,1.32,0.27,1.32,0.48)
        _t(ax_alg1,5.7,0.7,"$R(s)$"); _a(ax_alg1,6.15,0.7,6.65,0.7)
        _blk(ax_alg1,7.3,0.7,wb,hb,"$G(s)$"); _a(ax_alg1,7.95,0.7,8.45,0.7)
        _s(ax_alg1,8.67,0.7); _pm2(ax_alg1,8.67-r-0.22,0.7+0.32,"±","black")
        _a(ax_alg1,8.89,0.7,9.4,0.7); _t(ax_alg1,9.65,0.7,"$Y(s)$")
        _t(ax_alg1,5.7,-0.42,"$X(s)$",fs=7.5); _blk(ax_alg1,7.3,-0.42,wb,hb,"$G(s)$",fs=7)
        _a(ax_alg1,6.15,-0.42,6.65,-0.42); _a(ax_alg1,7.95,-0.42,8.67,-0.42)
        _l(ax_alg1,8.67,-0.42,8.67,0.48)
        _t(ax_alg1,5.5,-1.18,"Inserir G(s) no ramo de X(s)",fs=7,ha="center")
        plt.tight_layout()
        show_fig(fig_alg1, 0.68)
        
        # Álgebra 2
        fig_alg2, ax_alg2 = plt.subplots(figsize=(7.0, 3.0))
        _setup(ax_alg2, "2. Mover ponto de soma para DEPOIS de G(s)")
        _t(ax_alg2,0.2,0.7,"$R(s)$"); _a(ax_alg2,0.65,0.7,1.10,0.7)
        _s(ax_alg2,1.32,0.7); _pm2(ax_alg2,1.32-r-0.22,0.7+0.32,"±","black")
        _a(ax_alg2,1.54,0.7,2.05,0.7); _blk(ax_alg2,2.7,0.7,wb,hb,"$G(s)$")
        _a(ax_alg2,3.35,0.7,3.85,0.7); _t(ax_alg2,4.1,0.7,"$Y(s)$")
        _blk(ax_alg2,1.32,-0.45,wb,hb,"$1/G(s)$",fs=7)
        _t(ax_alg2,1.32,-1.05,"$X(s)$",fs=7.5); _a(ax_alg2,1.32,-0.83,1.32,-0.65); _a(ax_alg2,1.32,-0.25,1.32,0.48)
        _t(ax_alg2,5.7,0.7,"$R(s)$"); _a(ax_alg2,6.15,0.7,6.65,0.7)
        _blk(ax_alg2,7.3,0.7,wb,hb,"$G(s)$"); _a(ax_alg2,7.95,0.7,8.45,0.7)
        _s(ax_alg2,8.67,0.7); _pm2(ax_alg2,8.67-r-0.22,0.7+0.32,"±","black")
        _a(ax_alg2,8.89,0.7,9.4,0.7); _t(ax_alg2,9.65,0.7,"$Y(s)$")
        _t(ax_alg2,8.67,-0.42,"$X(s)$",fs=7.5); _a(ax_alg2,8.67,-0.20,8.67,0.48)
        _t(ax_alg2,5.5,-1.18,"Remover 1/G — X(s) entra direto no somador",fs=7,ha="center")
        plt.tight_layout()
        show_fig(fig_alg2, 0.68)
        
        # Álgebra 3
        fig_alg3, ax_alg3 = plt.subplots(figsize=(7.0, 3.2))
        _setup(ax_alg3, "3. Mover ponto de ramificação para ANTES de G(s)")
        _t(ax_alg3,0.2,0.5,"$R(s)$"); _a(ax_alg3,0.65,0.5,1.15,0.5)
        _blk(ax_alg3,1.8,0.5,wb,hb,"$G(s)$")
        _l(ax_alg3,2.45,0.5,3.1,0.5); _d(ax_alg3,3.1,0.5)
        _a(ax_alg3,3.1,0.5,3.7,0.5);    _t(ax_alg3,4.15,0.5,"$R(s)G(s)$",fs=7.5)
        _l(ax_alg3,3.1,0.5,3.1,0.08);  _a(ax_alg3,3.1,0.08,3.7,0.08);  _t(ax_alg3,4.15,0.08,"$R(s)G(s)$",fs=7.5)
        _l(ax_alg3,3.1,0.5,3.1,-0.34); _a(ax_alg3,3.1,-0.34,3.7,-0.34); _t(ax_alg3,4.15,-0.34,"$R(s)G(s)$",fs=7.5)
        _t(ax_alg3,5.7,0.5,"$R(s)$")
        _l(ax_alg3,6.15,0.5,6.65,0.5); _d(ax_alg3,6.65,0.5)
        _a(ax_alg3,6.65,0.5,7.15,0.5); _blk(ax_alg3,7.8,0.5,wb,hb,"$G(s)$")
        _a(ax_alg3,8.45,0.5,9.1,0.5);   _t(ax_alg3,9.55,0.5,"$R(s)G(s)$",fs=7.5)
        _l(ax_alg3,6.65,0.5,6.65,0.08); _a(ax_alg3,6.65,0.08,7.15,0.08)
        _blk(ax_alg3,7.8,0.08,wb,hb,"$G(s)$",fs=7)
        _a(ax_alg3,8.45,0.08,9.1,0.08); _t(ax_alg3,9.55,0.08,"$R(s)G(s)$",fs=7.5)
        _l(ax_alg3,6.65,0.5,6.65,-0.34); _a(ax_alg3,6.65,-0.34,7.15,-0.34)
        _blk(ax_alg3,7.8,-0.34,wb,hb,"$G(s)$",fs=7)
        _a(ax_alg3,8.45,-0.34,9.1,-0.34); _t(ax_alg3,9.55,-0.34,"$R(s)G(s)$",fs=7.5)
        _t(ax_alg3,5.5,-1.18,"G(s) duplicado em cada ramo",fs=7,ha="center")
        plt.tight_layout()
        show_fig(fig_alg3, 0.68)
        
        # Álgebra 4
        fig_alg4, ax_alg4 = plt.subplots(figsize=(7.0, 3.4))
        _setup(ax_alg4, "4. Mover ponto de ramificação para DEPOIS de G(s)")
        _t(ax_alg4,0.2,0.6,"$R(s)$"); _a(ax_alg4,0.65,0.6,1.15,0.6)
        _blk(ax_alg4,1.8,0.6,wb,hb,"$G(s)$")
        _l(ax_alg4,2.45,0.6,3.1,0.6); _d(ax_alg4,3.1,0.6)
        _a(ax_alg4,3.1,0.6,3.7,0.6);   _t(ax_alg4,4.2,0.6,"$R(s)G(s)$",fs=7.5)
        _l(ax_alg4,3.1,0.6,3.1,0.12);  _a(ax_alg4,3.1,0.12,3.45,0.12)
        _blk(ax_alg4,4.05,0.12,wb*0.95,hb,"$1/G(s)$",fs=7)
        _a(ax_alg4,4.52,0.12,5.1,0.12); _t(ax_alg4,5.18,0.12,"$R(s)$",fs=7.5)
        _l(ax_alg4,3.1,0.6,3.1,-0.28); _a(ax_alg4,3.1,-0.28,3.45,-0.28)
        _blk(ax_alg4,4.05,-0.28,wb*0.95,hb,"$1/G(s)$",fs=7)
        _a(ax_alg4,4.52,-0.28,5.1,-0.28); _t(ax_alg4,5.18,-0.28,"$R(s)$",fs=7.5)
        _t(ax_alg4,5.7,0.6,"$R(s)$")
        _l(ax_alg4,6.15,0.6,6.65,0.6); _d(ax_alg4,6.65,0.6)
        _a(ax_alg4,6.65,0.6,7.15,0.6); _blk(ax_alg4,7.8,0.6,wb,hb,"$G(s)$")
        _a(ax_alg4,8.45,0.6,9.1,0.6);   _t(ax_alg4,9.55,0.6,"$R(s)G(s)$",fs=7.5)
        _l(ax_alg4,6.65,0.6,6.65,0.12); _a(ax_alg4,6.65,0.12,9.1,0.12); _t(ax_alg4,9.55,0.12,"$R(s)$",fs=7.5)
        _l(ax_alg4,6.65,0.6,6.65,-0.28); _a(ax_alg4,6.65,-0.28,9.1,-0.28); _t(ax_alg4,9.55,-0.28,"$R(s)$",fs=7.5)
        _t(ax_alg4,5.5,-1.22,"Inserir 1/G(s) nos ramos de saída direta",fs=7,ha="center")
        plt.tight_layout()
        show_fig(fig_alg4, 0.68)
        
        # Exemplo numérico
        st.markdown(r"#### Exemplo numérico: redução de malha fechada")
        st.markdown(r"""
        $G(s) = \dfrac{10}{s+2}$, $\quad H(s) = 1$ $\quad\Rightarrow\quad$ Realimentação negativa unitária:
        $T(s) = \dfrac{G(s)}{1+G(s)H(s)} = \dfrac{10}{s+12}$
        """)
        
        T_red = lti([10], [1, 12])
        t_red, y_red = sc_step(T_red)
        fig_red, ax_red = plt.subplots(figsize=(5.0, 2.8))
        ax_red.plot(t_red, y_red, color=COR["saida"], lw=1.8, label="$y(t)$ — malha fechada")
        ax_red.axhline(10/12, color="gray", ls="--", lw=1, label=f"Regime = {10/12:.3f}")
        ax_red.fill_between(t_red, 0.98*(10/12), 1.02*(10/12), alpha=0.15, color="seagreen", label="Banda ±2%")
        estilo(ax_red, "Tempo (s)", "Amplitude")
        ax_red.set_title(r"$G(s)=10/(s+2)$,  $H(s)=1$  $\Rightarrow$  $T(s)=10/(s+12)$", fontsize=8)
        ax_red.legend(fontsize=7); ax_red.set_xlim(0, t_red[-1])
        plt.tight_layout()
        show_fig(fig_red, 0.50)
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Ganho estático (DC)", f"{10/12:.4f}")
        col_b.metric("Constante de tempo", f"{1/12:.4f} s")
        col_c.metric("Tempo de acomodação (2%)", f"{4/12:.4f} s")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 13 — REFERÊNCIAS
        # ═══════════════════════════════════════════════════════════════════════════════
        with st.expander("13. Referências", expanded=False):
            st.markdown("""
        - **LATHI, B. P.; GREEN, R.** *Sinais e Sistemas Lineares*. 3ª ed. Oxford University Press, 2018.
        - **NISE, N. S.** *Engenharia de Sistemas de Controle*. 7ª ed. Wiley / LTC, 2017.
        - **DORF, R. C.; BISHOP, R. H.** *Sistemas de Controle Modernos*. 13ª ed. LTC, 2018.
        - **OGATA, K.** *Engenharia de Controle Moderno*. 5ª ed. Pearson, 2014.
        - **OPPENHEIM, A. V.; WILLSKY, A. S.** *Sinais e Sistemas*. 2ª ed. Prentice Hall, 2010.
        """)
        
        st.divider()
        
        st.markdown(
            "<div style='text-align:center;color:gray;font-size:12px'>"
            "Sinais e Sistemas Lineares &nbsp;·&nbsp; Modelagem e Sistemas Lineares"
            " &nbsp;·&nbsp; Engenharia de Energia &nbsp;·&nbsp; CNAT — IFRN<br>"
            "Autor: Marcus V A Fernandes &nbsp;·&nbsp; marcus.fernandes@ifrn.edu.br"
            " &nbsp;·&nbsp; v1.0"
            "</div>",
            unsafe_allow_html=True,
        )
