"""
📈 Dinâmica no Domínio do Tempo — Sistemas de Ordem 1
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
from plotly.subplots import make_subplots
import warnings
        
def run():
        
        warnings.filterwarnings("ignore")
        
        # ── Configuração da Página ────────────────────────────────────────────────────
        st.set_page_config(
            page_title="Dinâmica — Sistemas de Ordem 1",
            page_icon="📈",
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
            ref      = "steelblue",
            saida    = "crimson",
            natural  = "seagreen",
            instavel = "crimson",
            marginal = "darkorange",
            degrau   = "royalblue",
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
        
        def seta(ax, x1, y1, x2, y2, lb="", ldy=0.09):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=COR["bloco_e"],
                                lw=1.4, shrinkA=0, shrinkB=0), zorder=2)
            if lb:
                ax.text((x1+x2)/2, (y1+y2)/2+ldy, lb, ha="center", fontsize=7.5)
        
        # ── Funções de resposta ───────────────────────────────────────────────────────
        def step_response(k_v, a_v, t_arr):
            return (k_v / a_v) * (1.0 - np.exp(-a_v * t_arr))
        
        def step_response_zero(k_v, a_v, b_v, t_arr):
            sys_h = lti([k_v, k_v * b_v], [1, a_v])
            _, y = sc_step(sys_h, T=t_arr)
            return y
        
        # ── CSS responsivo + show_fig ─────────────────────────────────────────────────
        st.markdown("""
        <style>
        .fig-wrap {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .fig-wrap > div { width: 100%; }
        @media (min-width: 769px) {
            .fig-wrap > div {
                width: var(--fw, 65%);
                max-width: var(--fw, 65%);
            }
        }
        .fig-wrap img,
        .fig-wrap [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        def show_fig(fig, width_frac=0.65):
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
        
        # ── Helper: plano s padronizado ───────────────────────────────────────────────
        def make_plane_trace(polos_x, polos_y, zeros_x=None, zeros_y=None,
                             xrange=(-5, 1.5), yrange=(-1.5, 1.5),
                             polo_cor="#d62728", zero_cor="#2ca02c"):
            """Retorna figura Plotly do plano s com regiões SPE/SPD."""
            fig = go.Figure()
            fig.add_vrect(x0=xrange[0], x1=0, fillcolor="seagreen",
                          opacity=0.06, layer="below", line_width=0)
            fig.add_vrect(x0=0, x1=xrange[1], fillcolor="crimson",
                          opacity=0.06, layer="below", line_width=0)
            if zeros_x is not None:
                fig.add_trace(go.Scatter(
                    x=zeros_x, y=zeros_y, mode="markers",
                    marker=dict(symbol="circle-open", size=14,
                                color=zero_cor, line=dict(width=2.5)),
                    name="Zero"))
            fig.add_trace(go.Scatter(
                x=polos_x, y=polos_y, mode="markers",
                marker=dict(symbol="x", size=14,
                            color=polo_cor, line=dict(width=3)),
                name="Polo"))
            fig.update_layout(
                xaxis=dict(title="σ", range=list(xrange), zeroline=True,
                           zerolinecolor="black", zerolinewidth=1),
                yaxis=dict(title="jω", range=list(yrange), zeroline=True,
                           zerolinecolor="black", zerolinewidth=1),
                height=300, margin=dict(t=10, b=10, l=50, r=10),
                template="plotly_white", showlegend=True,
                legend=dict(orientation="h", y=1.08),
            )
            return fig
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CABEÇALHO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.title("📈 Dinâmica no Domínio do Tempo")
        st.subheader("Sistemas de Ordem 1")
        st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
        st.markdown("---")
        
        # ── Índice ────────────────────────────────────────────────────────────────────
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[1. Função de Transferência de 1ª Ordem](#1-fun-o-de-transfer-ncia-de-1-ordem)**
        - 1.1 Grau relativo $n^* = n - m$
        - 1.2 Formas canônicas: ganho estático $k'=k/a$ e constante de tempo $\tau=1/a$
        - 1.3 Diagrama de blocos e localização do polo no plano $s$
        
        **[2. Resposta ao Degrau — Componentes e Especificações](#2-resposta-ao-degrau-componentes-e-especifica-es)**
        - 2.1 Dedução analítica: componentes forçada $y_f$ e natural $y_n$
        - 2.2 Especificações: $y(\infty)$, $\tau$, $T_r$, $T_{s_{2\%}}$
        - 2.3 Identificação experimental dos parâmetros $a$ e $k$
        - 🎛️ Explorador interativo: sliders $k$, $a$, $k_r$
        
        **[3. Sistemas de Grau Relativo 1 — Exemplos e Efeito dos Parâmetros](#3-sistemas-de-grau-relativo-1-exemplos-e-efeito-dos-par-metros)**
        - 3.1 Exemplos físicos: RL, RC, massa-amortecedor, inércia-amortecedor, térmico, hidráulico
        - 3.2 Efeito de $k$ (ganho estático) e de $a$ (velocidade de resposta)
        - 3.3 🎛️ Explorador interativo: plano $s$ + resposta ao degrau
        
        **[4. Sistemas de Grau Relativo 0 — Efeito do Zero](#4-sistemas-de-grau-relativo-0-efeito-do-zero)**
        - 4.1 Função de transferência com zero finito: $H'(s) = k(s+b)/(s+a)$
        - 4.2 Valores notáveis $y'(0^+) = k$, $y'(\infty) = kb/a$
        - 4.3 Fase mínima ($b>0$), cancelamento polo-zero ($b=a$), fase não-mínima ($b<0$)
        - 🎛️ Explorador interativo: sliders $k$, $a$, $b$
        
        **[5. Polo no Semiplano Direito — Sistema Instável](#5-polo-no-semiplano-direito-sistema-inst-vel)**
        - 5.1 Resposta ao degrau: $y(t) = (k/a)(e^{+at}-1)$
        - 5.2 Efeito de zero adicional sobre a velocidade de divergência
        - 🎛️ Explorador interativo: slider $a$
        
        **[6. Polo na Origem — Sistema Marginalmente Estável](#6-polo-na-origem-sistema-marginalmente-est-vel)**
        - 6.1 Integrador puro (malha aberta): saída rampa $y(t) = kt$
        - 6.2 Estabilização por realimentação negativa: polo migra de $s=0$ para $s=-k$
        - 6.3 🎛️ Explorador interativo: slider $k$ — polo MA vs. polo MF
        
        **[7. Referências](#7-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1 — FUNÇÃO DE TRANSFERÊNCIA DE 1ª ORDEM
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Função de Transferência de 1ª Ordem")
        
        st.markdown(r"""
        ### 1.1 Grau relativo
        
        O **grau relativo** $n^*$ de uma função de transferência $H(s)$ é a diferença entre o
        grau do denominador (número de polos, $n$) e o grau do numerador (número de zeros, $m$):
        
        $$n^* = n - m$$
        
        Para sistemas causais e próprios, $n^* \geq 0$.
        
        ### 1.2 Formas canônicas
        
        Para sistemas de 1ª ordem ($n=1$, $n^*=1$), o único polo está em $s = -a$ e a função de
        transferência admite três representações equivalentes:
        
        $$H(s) = \frac{Y(s)}{X(s)} = \frac{k}{s+a} = \frac{k/a}{\dfrac{s}{a} + 1} = \frac{k'}{\tau s + 1}$$
        
        onde $k' = k/a$ é o **ganho estático** (DC) e $\tau = 1/a$ é a **constante de tempo**.
        
        ### 1.3 Diagrama de blocos e plano $s$
        
        O polo ocupa a posição $s = -a$ no plano complexo.
        
        > Para $a > 0$: polo no **semiplano esquerdo** → sistema **estável** ($y(t) \to$ constante).
        > Para $a < 0$: polo no **semiplano direito** → sistema **instável** ($y(t) \to \infty$).
        """)
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(8.0, 2.4))
        ax = axes1[0]
        ax.set_xlim(0, 5); ax.set_ylim(-0.6, 0.6); ax.axis("off")
        seta(ax, 0.2, 0, 1.2, 0, "X(s)")
        bloco(ax, 2.5, 0, 2.3, 0.5, r"$H(s) = \dfrac{k}{s+a}$", fs=9)
        seta(ax, 3.65, 0, 4.8, 0, "Y(s)")
        ax.set_title("Diagrama de blocos — sistema de 1ª ordem", fontsize=8.5)
        
        ax2 = axes1[1]
        ax2.set_xlim(-3, 1.5); ax2.set_ylim(-1.5, 1.5)
        ax2.axhline(0, color="k", lw=0.8); ax2.axvline(0, color="k", lw=0.8)
        ax2.fill_betweenx([-1.5, 1.5], -3,  0,   alpha=0.07, color="seagreen", label="Estável (esq.)")
        ax2.fill_betweenx([-1.5, 1.5],  0, 1.5,  alpha=0.07, color="crimson",  label="Instável (dir.)")
        ax2.plot(-1, 0, "x", color=COR["saida"], ms=12, mew=2.5, label=r"polo $s=-a$")
        ax2.text(-1, 0.22, r"$-a$", ha="center", fontsize=9, color=COR["saida"])
        ax2.set_xlabel(r"$\sigma$", fontsize=8); ax2.set_ylabel(r"$j\omega$", fontsize=8)
        ax2.set_title(r"Plano $s$ — localização do polo", fontsize=8.5)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.spines[["right", "top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig1, 0.72)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2 — RESPOSTA AO DEGRAU
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Resposta ao Degrau — Componentes e Especificações")
        
        st.markdown(r"""
        ### 2.1 Dedução analítica
        
        Para entrada degrau de amplitude $k_r$, $X(s) = k_r/s$:
        
        $$Y(s) = H(s)\cdot\frac{k_r}{s} = \frac{k\,k_r}{s\,(s+a)}$$
        
        Expandindo em frações parciais e invertendo:
        
        $$\boxed{y(t) = y_f(t) + y_n(t) = \frac{k\,k_r}{a}\bigl(1 - e^{-at}\bigr), \quad t \geq 0}$$
        
        | Componente | Expressão | Nome alternativo |
        |---|---|---|
        | $y_f(t) = k\,k_r/a$ | Resposta **forçada** | Componente de estado nulo (*zero-state*) |
        | $y_n(t) = -(k\,k_r/a)\,e^{-at}$ | Resposta **natural** | Componente de entrada nula (*zero-input*) |
        
        ### 2.2 Especificações de desempenho
        
        | Parâmetro | Símbolo | Fórmula |
        |---|---|---|
        | Valor final | $y(\infty)$ | $k\,k_r/a$ |
        | Constante de tempo | $\tau$ | $1/a$ |
        | Tempo de subida | $T_r$ | $\approx 2{,}2/a$ |
        | Tempo de acomodação 2% | $T_{s_{2\%}}$ | $\approx 4/a$ |
        
        > **Características:** sem ultrapassagem (*overshoot*), inclinação inicial $\dot{y}(0^+) = k\,k_r > 0$.
        
        ### 2.3 Identificação experimental
        
        $$a = \frac{4}{T_{s_{2\%}}} \approx \frac{2{,}2}{T_r}, \qquad k = \frac{a\cdot y(\infty)}{k_r}$$
        """)
        
        k_val2 = 1.4; a_val2 = 0.7; kr_val2 = 1.0
        t_arr2 = np.linspace(0, 12, 600)
        yinf2 = k_val2 * kr_val2 / a_val2
        tau2   = 1.0 / a_val2; Tr2 = 2.2 / a_val2; Ts2_v = 4.0 / a_val2
        yf2 = yinf2 * np.ones_like(t_arr2)
        yn2 = -yinf2 * np.exp(-a_val2 * t_arr2)
        y2  = yf2 + yn2
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(9.5, 3.4))
        ax = axes2[0]
        ax.plot(t_arr2, yf2, "--", color=COR["ref"],    lw=1.4, label=r"$y_f(t)=k\,k_r/a$ (forçada)")
        ax.plot(t_arr2, yn2, ":",  color=COR["natural"], lw=1.6, label=r"$y_n(t)=-(k\,k_r/a)e^{-at}$ (natural)")
        ax.plot(t_arr2, y2,        color=COR["degrau"],  lw=2.0, label=r"$y(t)=y_f+y_n$")
        ax.axhline(yinf2, color="gray", lw=0.8, ls="--")
        ax.axvline(tau2,  color="purple", lw=0.9, ls=":")
        ax.annotate(r"$\tau=1/a$", xy=(tau2, 0.63*yinf2), xytext=(tau2+0.7, 0.48*yinf2),
                    fontsize=8, color="purple", arrowprops=dict(arrowstyle="->", color="purple", lw=0.8))
        ax.text(11.5, yinf2+0.04, r"$y(\infty)$", ha="right", fontsize=8, color="gray")
        estilo(ax, xlabel="t (s)")
        ax.set_title(rf"Componentes — $H(s)={k_val2}/(s+{a_val2})$, degrau $k_r={kr_val2}$", fontsize=8.5)
        ax.legend(fontsize=7)
        
        ax2 = axes2[1]
        ax2.plot(t_arr2, y2, color=COR["degrau"], lw=2.0)
        for frac, cor, lb in [(1.00,"gray",""), (0.98,"brown","98%"),
                              (0.90,"olive","90%"), (0.63,"purple","63%"), (0.10,"olive","10%")]:
            ax2.axhline(frac*yinf2, color=cor, lw=0.7, ls="--")
            if lb: ax2.text(12.1, frac*yinf2, lb, va="center", fontsize=7, color=cor)
        for xv, cor, lb in [(tau2,"purple",r"$\tau$"), (Tr2,"olive",r"$T_r$"), (Ts2_v,"brown",r"$T_s$")]:
            ax2.axvline(xv, color=cor, lw=0.9, ls=":")
            ax2.text(xv+0.1, -0.18, lb, fontsize=8, color=cor)
        estilo(ax2, xlabel="t (s)")
        ax2.set_title("Especificações de desempenho", fontsize=8.5)
        ax2.set_xlim(0, 12)
        plt.tight_layout()
        show_fig(fig2, 0.88)
        
        # ── Explorador interativo seção 2 ─────────────────────────────────────────────
        st.markdown("### 🎛️ Explorador — Resposta ao Degrau")
        st.caption("Mova os sliders e observe como cada parâmetro afeta $y(\\infty)$, $\\tau$ e $T_s$.")
        
        c2a, c2b = st.columns([1, 2])
        with c2a:
            k2 = st.slider("Ganho $k$",  0.5, 5.0, 2.0, 0.1, key="k2")
            a2 = st.slider("Polo $a$",   0.2, 3.0, 0.8, 0.1, key="a2")
            kr2 = st.slider("Degrau $k_r$", 0.5, 3.0, 1.0, 0.1, key="kr2")
            yinf_e2 = k2 * kr2 / a2
            st.info(f"$y(\\infty)={yinf_e2:.3f}$ · $\\tau={1/a2:.2f}$ s · $T_s\\approx{4/a2:.2f}$ s")
        
        with c2b:
            t_e2 = np.linspace(0, 18, 700)
            y_e2 = yinf_e2 * (1 - np.exp(-a2 * t_e2))
            fig_e2 = go.Figure()
            fig_e2.add_hline(y=yinf_e2, line_dash="dash", line_color="gray",
                             annotation_text=f"y(∞)={yinf_e2:.3f}", annotation_position="top right")
            fig_e2.add_trace(go.Scatter(x=t_e2, y=y_e2, mode="lines",
                                        line=dict(color="#1f77b4", width=2.5), name="y(t)"))
            fig_e2.add_vline(x=1/a2, line_dash="dot", line_color="purple",
                             annotation_text=f"τ={1/a2:.2f}s", annotation_position="top right")
            fig_e2.add_vline(x=4/a2, line_dash="dot", line_color="brown",
                             annotation_text=f"Ts={4/a2:.2f}s", annotation_position="top right")
            fig_e2.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20),
                                 xaxis_title="t (s)", yaxis_title="y(t)",
                                 template="plotly_white", showlegend=False)
            st.plotly_chart(fig_e2, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3 — GRAU RELATIVO 1
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Sistemas de Grau Relativo 1 — Exemplos e Efeito dos Parâmetros")
        
        st.markdown(r"""
        ### 3.1 Exemplos físicos
        
        Todos os sistemas abaixo seguem $H(s) = k/(s+a)$:
        
        | Domínio | Sistema ($u \to y$) | $k$ | $a$ |
        |---|---|---|---|
        | Elétrico | Circuito RL: $V_a \to I$ | $1/L$ | $R/L$ |
        | Elétrico | Circuito RC: $V_e \to V_s$ | $1/(RC)$ | $1/(RC)$ |
        | Mecânico translacional | Massa-amortecedor: $F \to v$ | $1/M$ | $B/M$ |
        | Mecânico rotacional | Inércia-amortecedor: $\mathcal{T} \to \omega$ | $1/J$ | $B/J$ |
        | Térmico | Câmara isolada: $\dot{Q} \to T$ | $1/C_t$ | $1/(R_t C_t)$ |
        | Hidráulico | Reservatório: $Q_i \to h$ | $1/C_h$ | $1/(R_h C_h)$ |
        
        ### 3.2 Efeito dos parâmetros
        
        **Variação de $k$** (polo $a$ fixo): altera $y(\infty) = k/a$ sem afetar $\tau = 1/a$.
        
        **Variação de $a$** (ganho $k$ fixo): desloca o polo — maior $a$ → resposta mais rápida, menor $y(\infty)$.
        """)
        
        t_arr3 = np.linspace(0, 12, 600)
        a_fix3 = 0.8; k_fix3 = 1.0
        
        fig3, axes3 = plt.subplots(1, 2, figsize=(9.5, 3.2))
        ax = axes3[0]
        for kv, col in zip([1,2,3,4,5,6], plt.cm.viridis(np.linspace(0.15, 0.88, 6))):
            ax.plot(t_arr3, step_response(kv, a_fix3, t_arr3), color=col, label=f"k={kv}")
        ax.axvline(1/a_fix3, color="purple", lw=0.9, ls=":", label=rf"$\tau={1/a_fix3:.2f}$s")
        ax.set_xlim(0, 12); estilo(ax, xlabel="t (s)")
        ax.set_title(rf"Variação de $k$ (polo $a={a_fix3}$ fixo)", fontsize=8.5)
        ax.legend(ncol=2, fontsize=7)
        
        ax2 = axes3[1]
        for av, col in zip([0.4,0.8,1.2,1.6,2.0,2.5], plt.cm.plasma(np.linspace(0.15, 0.88, 6))):
            ax2.plot(t_arr3, step_response(k_fix3, av, t_arr3), color=col, label=f"a={av}")
        ax2.axhline(k_fix3, color="gray", lw=0.8, ls="--", label=r"$y(\infty)=k/a$")
        ax2.set_xlim(0, 12); estilo(ax2, xlabel="t (s)")
        ax2.set_title(rf"Variação de $a$ (ganho $k={k_fix3}$ fixo)", fontsize=8.5)
        ax2.legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig3, 0.88)
        
        # Diagrama polo-zero + resposta
        k_v3=4.0; a_v3=0.8; t_arr3b = np.linspace(0, 12, 600)
        y3b=step_response(k_v3, a_v3, t_arr3b); yinf3b=k_v3/a_v3
        Ts3_idx=np.argmax(y3b>=0.98*yinf3b); Ts3_med=t_arr3b[Ts3_idx]
        
        fig3b, axes3b = plt.subplots(1, 2, figsize=(9.5, 3.2))
        ax = axes3b[0]
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        ax.fill_betweenx([-1.5,1.5], -5, 0, alpha=0.07, color="seagreen")
        ax.fill_betweenx([-1.5,1.5],  0, 1.5, alpha=0.07, color="crimson")
        ax.plot(-a_v3, 0, "x", color=COR["saida"], ms=14, mew=3, label=rf"polo $s={-a_v3}$")
        ax.set_xlim(-5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel(r"$\sigma$", fontsize=8); ax.set_ylabel(r"$j\omega$", fontsize=8)
        ax.set_title(rf"Plano $s$: $H(s)={k_v3:.0f}/(s+{a_v3})$", fontsize=8.5)
        ax.spines[["right","top"]].set_visible(False); ax.legend(fontsize=7.5)
        
        ax2 = axes3b[1]
        ax2.plot(t_arr3b, y3b, color=COR["degrau"], lw=2.0)
        ax2.axhline(yinf3b, color="gray", lw=0.8, ls="--")
        ax2.axhline(0.98*yinf3b, color="brown", lw=0.7, ls=":")
        ax2.axvline(Ts3_med, color="brown", lw=0.9, ls=":")
        ax2.axvline(1/a_v3, color="purple", lw=0.9, ls=":")
        ax2.text(Ts3_med+0.1, 0.5, r"$T_{s_{2\%}}$"+f"={Ts3_med:.1f}s", color="brown", fontsize=7.5)
        ax2.text(1/a_v3+0.1, 0.3, rf"$\tau={1/a_v3:.2f}$s", color="purple", fontsize=7.5)
        ax2.text(11.8, yinf3b+0.15, f"$y(\\infty)={yinf3b:.1f}$", fontsize=7.5, ha="right")
        estilo(ax2, xlabel="t (s)"); ax2.set_title("Resposta ao degrau unitário", fontsize=8.5)
        plt.tight_layout()
        show_fig(fig3b, 0.88)
        
        # ── Explorador interativo seção 3 ─────────────────────────────────────────────
        st.markdown("### 3.3 🎛️ Explorador — Plano $s$ + Resposta ao Degrau")
        st.caption("Slider **azul** altera $k$ (só amplitude) · Slider **vermelho** altera $a$ (polo desloca)")
        
        c3a, c3b = st.columns([1, 2])
        with c3a:
            k3 = st.slider("Ganho $k$", 0.5, 6.0, 2.0, 0.1, key="k3")
            a3 = st.slider("Polo $a$",  0.2, 3.0, 0.8, 0.1, key="a3")
            st.info(f"polo $s={-a3:.2f}$ · $y(\\infty)={k3/a3:.3f}$ · $\\tau={1/a3:.2f}$ s · $T_s={4/a3:.2f}$ s")
        
        with c3b:
            t_e3 = np.linspace(0, 15, 600)
            y_e3 = (k3/a3) * (1 - np.exp(-a3 * t_e3))
        
            fig_e3 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s", "Resposta ao degrau unitário"))
            # plano s
            fig_e3.add_vrect(x0=-7, x1=0, fillcolor="seagreen", opacity=0.06,
                             layer="below", line_width=0, row=1, col=1)
            fig_e3.add_vrect(x0=0, x1=1.5, fillcolor="crimson", opacity=0.06,
                             layer="below", line_width=0, row=1, col=1)
            fig_e3.add_trace(go.Scatter(x=[-a3], y=[0], mode="markers",
                marker=dict(symbol="x", size=14, color="#d62728",
                            line=dict(width=3, color="#d62728")),
                name="Polo", showlegend=False), row=1, col=1)
            # resposta
            fig_e3.add_trace(go.Scatter(x=t_e3, y=y_e3, mode="lines",
                line=dict(color="#1f77b4", width=2.5), name="y(t)", showlegend=False), row=1, col=2)
            fig_e3.add_hline(y=k3/a3, line_dash="dash", line_color="gray", row=1, col=2)
        
            fig_e3.update_xaxes(title_text="σ", range=[-7, 1.5],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e3.update_yaxes(title_text="jω", range=[-1.5, 1.5],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e3.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e3.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e3.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=10),
                                 template="plotly_white")
            st.plotly_chart(fig_e3, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4 — GRAU RELATIVO 0 / EFEITO DO ZERO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Sistemas de Grau Relativo 0 — Efeito do Zero")
        
        st.markdown(r"""
        ### 4.1 Função de transferência com zero finito
        
        Adicionando um zero em $s = -b$ ($n^* = 1 - 1 = 0$):
        
        $$H'(s) = \frac{k(s+b)}{s+a} = \frac{k\,b}{s+a} + \frac{k\,s}{s+a}$$
        
        ### 4.2 Resposta ao degrau unitário ($k_r = 1$)
        
        $$\boxed{y'(t) = \frac{k\,b}{a} + k\!\left(1 - \frac{b}{a}\right)e^{-at}, \quad t\geq 0}$$
        
        Valores notáveis: $\quad y'(0^+) = k$ (independe de $b$), $\quad y'(\infty) = k\,b/a$
        
        O zero **não altera** $\tau = 1/a$.
        
        ### 4.3 Classificação pelo sinal de $b$
        
        | Posição do zero | Característica | Comportamento |
        |---|---|---|
        | $b > 0$ — semiplano esquerdo | Sistema de **fase mínima** | $y'(\infty)>0$; resposta mais rápida |
        | $b = a$ — coincide com o polo | **Cancelamento polo-zero** | $H'(s) = k$; resposta instantânea |
        | $b < 0$ — semiplano direito | Sistema de **fase não-mínima** | $y'(\infty)<0$; saída inverte de sinal |
        
        > **Fase não-mínima** ($b<0$): $y'(0^+)$ e $y'(\infty)$ têm sinais opostos — dificulta o controle.
        """)
        
        t_arr4 = np.linspace(0, 12, 600)
        k_v4=4.0; a_v4=0.8
        cenarios4=[(3.0, "b=+3 (fase mínima)",      COR["natural"]),
                    (-3.0,"b=−3 (fase não-mínima)",  COR["saida"]),
                    (0.8, "b=a=0.8 (cancela polo)",  COR["degrau"])]
        
        fig4a, axes4a = plt.subplots(1, 3, figsize=(9.5, 3.2))
        for ax, (bv, lbl, col) in zip(axes4a, cenarios4):
            y = step_response_zero(k_v4, a_v4, bv, t_arr4)
            ax.plot(t_arr4, y, color=col, lw=2.0)
            ax.axhline(y[-1], color="gray", lw=0.8, ls="--")
            ax.axhline(0, color="k", lw=0.5)
            estilo(ax, xlabel="t (s)"); ax.set_title(lbl, fontsize=8)
            ins = ax.inset_axes([0.54, 0.06, 0.44, 0.42])
            ins.axhline(0, color="k", lw=0.6); ins.axvline(0, color="k", lw=0.6)
            ins.plot(-a_v4, 0, "x", color=COR["saida"], ms=9, mew=2.0)
            ins.plot(-bv,   0, "o", color=COR["natural"], ms=7, mfc="white", mew=1.8)
            ins.set_xlim(-5, 3); ins.set_ylim(-1.2, 1.2)
            ins.set_xticks([]); ins.set_yticks([]); ins.set_title("Plano s", fontsize=7)
        plt.tight_layout()
        show_fig(fig4a, 0.88)
        
        b_vals4=[-5.0,-3.0,-1.0,1.0,3.0,5.0]
        colors4=plt.cm.RdYlGn(np.linspace(0.08, 0.92, len(b_vals4)))
        fig4b, axes4b = plt.subplots(1, 2, figsize=(9.5, 3.2))
        ax_pz4=axes4b[0]; ax_r4=axes4b[1]
        ax_pz4.axhline(0, color="k", lw=0.8); ax_pz4.axvline(0, color="k", lw=0.8)
        ax_pz4.fill_betweenx([-1.5,1.5],-8,0,alpha=0.06,color="seagreen")
        ax_pz4.fill_betweenx([-1.5,1.5], 0,4,alpha=0.06,color="crimson")
        ax_pz4.plot(-a_v4,0,"x",color=COR["saida"],ms=14,mew=3,label="polo")
        for bv, col in zip(b_vals4, colors4):
            y = step_response_zero(k_v4, a_v4, bv, t_arr4)
            ax_r4.plot(t_arr4, y, color=col, lw=1.8, label=f"b={bv:+.0f}")
            ax_pz4.plot(-bv, 0, "o", color=col, ms=8, mfc="white", mew=2.0, label=f"b={bv:+.0f}")
        ax_pz4.set_xlim(-8,4); ax_pz4.set_ylim(-1.5,1.5)
        ax_pz4.set_xlabel(r"$\sigma$",fontsize=8); ax_pz4.set_ylabel(r"$j\omega$",fontsize=8)
        ax_pz4.set_title(r"Plano $s$ — zeros e polo fixo",fontsize=8.5)
        ax_pz4.spines[["right","top"]].set_visible(False); ax_pz4.legend(ncol=2,fontsize=6.5)
        ax_r4.axhline(0,color="k",lw=0.5); estilo(ax_r4,xlabel="t (s)")
        ax_r4.set_title(r"Resposta ao degrau — variação de $b$",fontsize=8.5)
        ax_r4.legend(ncol=2,fontsize=7)
        plt.tight_layout()
        show_fig(fig4b, 0.88)
        
        # ── Explorador interativo seção 4 ─────────────────────────────────────────────
        st.markdown("### 4.4 🎛️ Explorador — Sistema com Zero")
        st.caption("○ = zero · × = polo · Varie $b$ de positivo para negativo e observe a inversão de $y(\\infty)$")
        
        c4a, c4b = st.columns([1, 2])
        with c4a:
            k4 = st.slider("Ganho $k$", 0.5, 6.0, 2.0, 0.1, key="k4")
            a4 = st.slider("Polo $a$",  0.2, 3.0, 0.8, 0.1, key="a4")
            b4 = st.slider("Zero $b$", -5.0, 5.0, 2.0, 0.1, key="b4")
            yinf_e4 = k4*b4/a4
            st.info(f"polo $s={-a4:.2f}$ · zero $s={-b4:.2f}$\n\n$y(0^+)={k4:.2f}$ · $y(\\infty)={yinf_e4:.3f}$ · $\\tau={1/a4:.2f}$ s")
        
        with c4b:
            t_e4 = np.linspace(0, 15, 600)
            y_e4 = step_response_zero(k4, a4, b4, t_e4)
        
            fig_e4 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s", "Resposta ao degrau"))
            fig_e4.add_vrect(x0=-8, x1=0, fillcolor="seagreen", opacity=0.06,
                             layer="below", line_width=0, row=1, col=1)
            fig_e4.add_vrect(x0=0, x1=4, fillcolor="crimson", opacity=0.06,
                             layer="below", line_width=0, row=1, col=1)
            fig_e4.add_trace(go.Scatter(x=[-a4], y=[0], mode="markers",
                marker=dict(symbol="x", size=14, color="#d62728",
                            line=dict(width=3, color="#d62728")),
                name="Polo", showlegend=True), row=1, col=1)
            fig_e4.add_trace(go.Scatter(x=[-b4], y=[0], mode="markers",
                marker=dict(symbol="circle-open", size=14, color="#2ca02c",
                            line=dict(width=2.5, color="#2ca02c")),
                name="Zero", showlegend=True), row=1, col=1)
            fig_e4.add_trace(go.Scatter(x=t_e4, y=y_e4, mode="lines",
                line=dict(color="#2ca02c", width=2.5),
                name="y(t)", showlegend=False), row=1, col=2)
            fig_e4.add_hline(y=yinf_e4, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e4.add_hline(y=0, line_width=0.8, line_color="black", row=1, col=2)
        
            fig_e4.update_xaxes(title_text="σ", range=[-8, 4],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e4.update_yaxes(title_text="jω", range=[-1.8, 1.8],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e4.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e4.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e4.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=10),
                                 template="plotly_white",
                                 legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_e4, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5 — POLO NO SEMIPLANO DIREITO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Polo no Semiplano Direito — Sistema Instável")
        
        st.markdown(r"""
        ### 5.1 Resposta ao degrau
        
        Para polo em $s = +a$ ($a > 0$):
        
        $$H(s) = \frac{k}{s - a} \quad\Rightarrow\quad y(t) = \frac{k}{a}\bigl(e^{+at} - 1\bigr), \quad y(0^+) = 0$$
        
        A saída **cresce sem limite** — sistema **BIBO instável**.
        
        ### 5.2 Efeito de um zero adicional
        
        Para $H'(s) = k(s+b)/(s-a)$, o coeficiente do modo instável é $k(a+b)/a$:
        
        | Zero | Coeficiente de $e^{+at}$ | Efeito |
        |---|---|---|
        | Sem zero | $k/a$ | Divergência padrão |
        | $b > 0$ (semiplano esq.) | $k(a+b)/a > k/a$ | Diverge **mais rápido** |
        | $b < 0$, $|b|< a$ | $0 < k(a+b)/a < k/a$ | Diverge **mais devagar** |
        | $b = -a$ (cancela polo) | $0$ | Cancelamento instável$^*$ |
        
        > $^*$Cancelamento polo-zero instável é perigoso: qualquer perturbação excita o modo instável.
        > Um polo real positivo **não pode ser estabilizado** em malha aberta por ajuste de ganho.
        """)
        
        t_arr5 = np.linspace(0, 6, 500)
        k_v5=1.0; a_v5=0.5
        cenarios5=[([k_v5],          [1,-a_v5], r"$H(s)=k/(s-a)$, sem zero"),
                    ([k_v5, 3*k_v5], [1,-a_v5], r"$H'(s)=k(s+3)/(s-a)$, zero esq."),
                    ([k_v5,-3*k_v5], [1,-a_v5], r"$H'(s)=k(s-3)/(s-a)$, zero dir.")]
        
        fig5, axes5 = plt.subplots(1, 3, figsize=(9.5, 3.0))
        for ax, (num, den, ttl) in zip(axes5, cenarios5):
            _, y = sc_step(lti(num, den), T=t_arr5)
            ax.plot(t_arr5, y, color=COR["instavel"], lw=2.0)
            ax.axhline(0, color="k", lw=0.6, ls="--")
            estilo(ax, xlabel="t (s)"); ax.set_title(ttl, fontsize=8)
        plt.suptitle(rf"Sistemas instáveis: polo em $s=+{a_v5}$", fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig5, 0.85)
        
        # ── Explorador interativo seção 5 ─────────────────────────────────────────────
        st.markdown("### 5.3 🎛️ Explorador — Velocidade de Divergência")
        st.caption("Quanto maior $a$, mais rápida a divergência (constante de tempo $1/a$)")
        
        c5a, c5b = st.columns([1, 2])
        with c5a:
            a5 = st.slider("Polo instável $a$", 0.1, 2.0, 0.5, 0.05, key="a5")
            st.info(f"polo $s=+{a5:.2f}$ · $y(t) \\propto e^{{+{a5:.2f}t}}$")
        
        with c5b:
            t_e5 = np.linspace(0, 8, 500)
            _, y_e5 = sc_step(lti([1.0], [1, -a5]), T=t_e5)
        
            fig_e5 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s (polo instável)", "Resposta ao degrau"))
            fig_e5.add_vrect(x0=-2, x1=0, fillcolor="seagreen", opacity=0.05,
                             layer="below", line_width=0, row=1, col=1)
            fig_e5.add_vrect(x0=0, x1=3, fillcolor="crimson", opacity=0.08,
                             layer="below", line_width=0, row=1, col=1)
            fig_e5.add_trace(go.Scatter(x=[a5], y=[0], mode="markers",
                marker=dict(symbol="x", size=14, color="#d62728",
                            line=dict(width=3, color="#d62728")),
                name=f"polo s=+{a5:.2f}", showlegend=False), row=1, col=1)
            fig_e5.add_trace(go.Scatter(x=t_e5, y=y_e5, mode="lines",
                line=dict(color="#d62728", width=2.5),
                name="y(t)", showlegend=False), row=1, col=2)
            fig_e5.add_hline(y=0, line_width=0.8, line_color="black", row=1, col=2)
        
            fig_e5.update_xaxes(title_text="σ", range=[-2, 3],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e5.update_yaxes(title_text="jω", range=[-1.5, 1.5],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e5.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e5.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e5.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=10),
                                 template="plotly_white")
            st.plotly_chart(fig_e5, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6 — POLO NA ORIGEM
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Polo na Origem — Sistema Marginalmente Estável")
        
        st.markdown(r"""
        ### 6.1 Integrador puro (malha aberta)
        
        Com polo em $s = 0$:
        
        $$G(s) = \frac{k}{s} \quad\Rightarrow\quad y(t) = k\,t \quad\text{(rampa)}$$
        
        A saída cresce sem limite mas **não exponencialmente** — sistema **marginalmente estável**.
        O integrador é ubíquo: motores DC, atuadores hidráulicos e servomecanismos possuem um polo na origem.
        
        ### 6.2 Estabilização por realimentação negativa unitária
        
        $$H(s) = \frac{G(s)}{1+G(s)} = \frac{k/s}{1+k/s} = \frac{k}{s+k}$$
        
        O polo migra de $s=0$ para $s=-k$:
        
        | | Malha aberta | Malha fechada |
        |---|---|---|
        | Polo | $s = 0$ | $s = -k$ |
        | Estabilidade | Marginal | **Estável** |
        | Resposta ao degrau | Rampa $k\,t$ | Exponencial; $y(\infty)=1$ |
        | Constante de tempo | — | $\tau = 1/k$ |
        | $T_{s_{2\%}}$ | — | $\approx 4/k$ |
        
        Aumentar $k$ afasta o polo da origem e acelera a resposta de malha fechada.
        """)
        
        t_arr6 = np.linspace(0, 10, 600)
        k_vals6=[1,2,3,4,5,6]; colors6=plt.cm.viridis(np.linspace(0.15, 0.88, len(k_vals6)))
        
        fig6a, axes6a = plt.subplots(1, 2, figsize=(9.5, 3.2))
        ax = axes6a[0]
        for kv, col in zip(k_vals6, colors6):
            ax.plot(t_arr6, kv*t_arr6, color=col, label=f"k={kv}")
        ax.set_ylim(0, 28); estilo(ax, xlabel="t (s)")
        ax.set_title(r"Malha aberta $G(s)=k/s$ — saída rampa", fontsize=8.5)
        ax.legend(ncol=2, fontsize=7)
        
        ax2 = axes6a[1]
        for kv, col in zip(k_vals6, colors6):
            ax2.plot(t_arr6, step_response(kv, kv, t_arr6), color=col, label=f"k={kv}")
        ax2.axhline(1.0, color="gray", lw=0.8, ls="--", label=r"$y(\infty)=1$")
        estilo(ax2, xlabel="t (s)")
        ax2.set_title(r"Malha fechada $H(s)=k/(s+k)$", fontsize=8.5)
        ax2.legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig6a, 0.88)
        
        fig6b, axes6b = plt.subplots(1, 2, figsize=(7.5, 2.8))
        ax = axes6b[0]
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        ax.plot(0, 0, "x", color="darkorange", ms=14, mew=3, label=r"polo $s=0$")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1, 1)
        ax.set_xlabel(r"$\sigma$", fontsize=8); ax.set_ylabel(r"$j\omega$", fontsize=8)
        ax.set_title(r"Malha aberta $G(s)=k/s$", fontsize=8.5)
        ax.spines[["right","top"]].set_visible(False); ax.legend(fontsize=7.5)
        
        ax2 = axes6b[1]
        ax2.axhline(0, color="k", lw=0.8); ax2.axvline(0, color="k", lw=0.8)
        ax2.fill_betweenx([-1,1], -8, 0, alpha=0.07, color="seagreen")
        for kv, col in zip(k_vals6, colors6):
            ax2.plot(-kv, 0, "x", color=col, ms=12, mew=2.5, label=f"k={kv}")
        ax2.set_xlim(-8, 1); ax2.set_ylim(-1, 1)
        ax2.set_xlabel(r"$\sigma$", fontsize=8); ax2.set_ylabel(r"$j\omega$", fontsize=8)
        ax2.set_title(r"Malha fechada — polo em $s=-k$", fontsize=8.5)
        ax2.spines[["right","top"]].set_visible(False); ax2.legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig6b, 0.72)
        
        # ── Explorador interativo seção 6 ─────────────────────────────────────────────
        st.markdown("### 6.3 🎛️ Explorador — Polo MA vs. Polo MF")
        st.caption("Laranja × = polo MA (fixo na origem) · Azul × = polo MF (desloca para $s=-k$)")
        
        c6a, c6b = st.columns([1, 2])
        with c6a:
            k6 = st.slider("Ganho $k$", 0.5, 8.0, 2.0, 0.1, key="k6")
            st.info(f"Polo MA: $s=0$  →  Polo MF: $s={-k6:.2f}$\n\n$\\tau={1/k6:.2f}$ s · $T_s={4/k6:.2f}$ s · $y(\\infty)=1$")
        
        with c6b:
            t_e6 = np.linspace(0, 10, 600)
            y_e6 = 1.0 * (1 - np.exp(-k6 * t_e6))
        
            fig_e6 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s: MA (laranja) vs MF (azul)",
                                                   "Resposta ao degrau — Malha Fechada"))
            fig_e6.add_vrect(x0=-9, x1=0, fillcolor="seagreen", opacity=0.06,
                             layer="below", line_width=0, row=1, col=1)
            # polo MA — laranja, fixo na origem
            fig_e6.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                marker=dict(symbol="x", size=14, color="darkorange",
                            line=dict(width=3, color="darkorange")),
                name="Polo MA (s=0)", showlegend=True), row=1, col=1)
            # polo MF — azul, desloca com k
            fig_e6.add_trace(go.Scatter(x=[-k6], y=[0], mode="markers",
                marker=dict(symbol="x", size=14, color="#1f77b4",
                            line=dict(width=3, color="#1f77b4")),
                name=f"Polo MF (s={-k6:.2f})", showlegend=True), row=1, col=1)
            # linha pontilhada entre os dois polos
            fig_e6.add_trace(go.Scatter(x=[0, -k6], y=[0, 0], mode="lines",
                line=dict(color="gray", width=1.2, dash="dot"),
                showlegend=False, hoverinfo="skip"), row=1, col=1)
            # resposta MF
            fig_e6.add_trace(go.Scatter(x=t_e6, y=y_e6, mode="lines",
                line=dict(color="#1f77b4", width=2.5),
                name="y(t) MF", showlegend=False), row=1, col=2)
            fig_e6.add_hline(y=1.0, line_dash="dash", line_color="gray",
                             annotation_text="y(∞)=1", annotation_position="top right",
                             row=1, col=2)
        
            fig_e6.update_xaxes(title_text="σ", range=[-9, 1.5],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e6.update_yaxes(title_text="jω", range=[-1.2, 1.2],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e6.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e6.update_yaxes(title_text="y(t)", range=[-0.05, 1.15], row=1, col=2)
            fig_e6.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=10),
                                 template="plotly_white",
                                 legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_e6, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7 — REFERÊNCIAS
        # ═══════════════════════════════════════════════════════════════════════════════
        with st.expander("7. Referências", expanded=False):
            st.markdown("""
        - **LATHI, B. P.; GREEN, R.** *Sinais e Sistemas Lineares*. 3ª ed. Oxford University Press, 2018.
        - **DORF, R. C.; BISHOP, R. H.** *Sistemas de Controle Modernos*. 13ª ed. LTC, 2017.
        - **OGATA, K.** *Engenharia de Controle Moderno*. 5ª ed. Pearson, 2014.
        - **NISE, N. S.** *Engenharia de Sistemas de Controle*. 7ª ed. Wiley / LTC, 2018.
        - **DE SOUZA, A. C. Z.; PINHEIRO, C. A. M.** *Introdução à Modelagem, Análise e Simulação de Sistemas Dinâmicos*. 1ª ed. Interciência, 2008.
        - **CASTRUCCI, P. B. de L.; BITTAR, A.; SALES, R. M.** *Controle Automático*. 2ª ed. LTC, 2018.
        """)
        
        st.divider()
        
        st.markdown(
            "<div style='text-align:center;color:gray;font-size:12px'>"
            "Dinâmica no Domínio do Tempo — Sistemas de Ordem 1 &nbsp;·&nbsp; SINTONIA — Sistemas de Controle<br>"
            "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
            " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
            "</div>",
            unsafe_allow_html=True,
        )
