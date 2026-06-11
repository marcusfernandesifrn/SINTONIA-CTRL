"""
📍 Análise de Sistemas com Realimentação — Estabilidade, Perturbação e LGR
Disciplina: Modelagem e Sistemas Lineares
Curso: Engenharia de Energia
Instituição: IFRN — Campus Natal-Central (CNAT)
Autor: Marcus V A Fernandes · marcus.fernandes@ifrn.edu.br · v1.0
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import lti, lsim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
        
def run():
        
        warnings.filterwarnings("ignore")
        
        # ── Configuração da Página ────────────────────────────────────────────────────
        st.set_page_config(
            page_title="Realimentação — LGR e Estabilidade",
            page_icon="🔄",
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
            ma     = "steelblue",
            mf     = "crimson",
            ref    = "gray",
            err    = "darkorange",
            pert   = "purple",
            degrau = "royalblue",
            rampa  = "seagreen",
            parab  = "darkorange",
        )
        
        CORES_LGR = ["#1f77b4","#d62728","#2ca02c","#9467bd","#e07000","#17becf"]
        
        # ── Helpers matplotlib ────────────────────────────────────────────────────────
        def estilo(ax, xlabel="t (s)", ylabel="Amplitude"):
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[["right","top"]].set_visible(False)
        
        def polo_x(ax, x, y, cor="#d62728", ms=11):
            ax.plot(x, y, "x", color=cor, ms=ms, mew=2.5, zorder=5)
        
        def plano_s_ax(ax, xlim=(-8,2), ylim=(-5,5)):
            ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
            ax.fill_betweenx([ylim[0],ylim[1]], xlim[0], 0, alpha=0.06, color="seagreen")
            ax.fill_betweenx([ylim[0],ylim[1]], 0, xlim[1], alpha=0.06, color="crimson")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\sigma$", fontsize=8)
            ax.set_ylabel(r"$j\omega$", fontsize=8)
            ax.spines[["right","top"]].set_visible(False)
        
        def bloco(ax, x, y, w, h, txt, fc="white", fs=8.5):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x-w/2,y-h/2), w, h,
                boxstyle="round,pad=0.04", facecolor=fc, edgecolor="k", lw=1.4, zorder=3))
            ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4)
        
        def seta(ax, x1, y1, x2, y2, lbl="", dy=0.12):
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4, shrinkA=0, shrinkB=0), zorder=2)
            if lbl:
                ax.text((x1+x2)/2, (y1+y2)/2+dy, lbl, ha="center", fontsize=8)
        
        def linha(ax, x1, y1, x2, y2):
            ax.plot([x1,x2],[y1,y2], color="k", lw=1.4, zorder=2)
        
        def circulo(ax, x, y, r=0.22):
            ax.add_patch(plt.Circle((x,y), r, fc="white", ec="k", lw=1.4, zorder=3))
        
        def ponto(ax, x, y):
            ax.plot(x, y, "o", color="k", ms=4, zorder=5)
        
        # ── Helpers Plotly ────────────────────────────────────────────────────────────
        def plotly_plano_s(fig, row, col, xlim=(-8,2), ylim=(-5,5)):
            fig.add_vrect(x0=xlim[0], x1=0, fillcolor="seagreen",
                          opacity=0.05, layer="below", line_width=0, row=row, col=col)
            fig.add_vrect(x0=0, x1=xlim[1], fillcolor="crimson",
                          opacity=0.05, layer="below", line_width=0, row=row, col=col)
            fig.update_xaxes(title_text="σ", range=list(xlim),
                             zeroline=True, zerolinecolor="black", row=row, col=col)
            fig.update_yaxes(title_text="jω", range=list(ylim),
                             zeroline=True, zerolinecolor="black", row=row, col=col)
        
        # ── Funções de simulação ──────────────────────────────────────────────────────
        def gera_entrada(tipo, t_arr, kr=1.0, atraso=0.0):
            dt = t_arr[1] - t_arr[0]
            n_atr = int(round(atraso / dt))
            if   tipo == "degrau":   u = np.ones_like(t_arr) * kr
            elif tipo == "rampa":    u = t_arr * kr
            elif tipo == "parabola": u = 0.5 * t_arr**2 * kr
            else:                    u = np.ones_like(t_arr) * kr
            if n_atr > 0:
                u = np.concatenate([np.zeros(n_atr), u[:-n_atr]])
            return u
        
        # ── Núcleo do LGR ────────────────────────────────────────────────────────────
        def calc_lgr(num_c, den_c, k_max=500.0):
            """Calcula ramos do LGR por varredura de k com rastreamento de continuidade."""
            k_arr = np.concatenate([
                np.linspace(0,    0.05,  500),
                np.linspace(0.05, 1,     600),
                np.linspace(1,    20,    800),
                np.linspace(20,   200,   800),
                np.linspace(200,  k_max, 1300),
            ])
            pd_c  = np.array(den_c, dtype=float)
            pn_c  = np.array(num_c, dtype=float)
            pn_pad = np.pad(pn_c, (len(pd_c)-len(pn_c), 0))
            n_p   = len(pd_c) - 1
            ramos = [[] for _ in range(n_p)]
            prev  = None
            for k in k_arr:
                roots = np.roots(pd_c + k * pn_pad)
                if prev is None:
                    ordered = sorted(roots, key=lambda z: (z.real, z.imag))
                else:
                    ordered = list(prev); rem = list(roots)
                    for ri in range(n_p):
                        b = int(np.argmin([abs(ordered[ri]-r) for r in rem]))
                        ordered[ri] = rem.pop(b)
                for ri in range(n_p):
                    ramos[ri].append(ordered[ri])
                prev = ordered
            return ramos, k_arr
        
        def k_crit_lgr(ramos, k_arr):
            """Estima k_crit como o menor k em que algum ramo cruza o eixo imaginário."""
            k_c = float("inf")
            for ramo in ramos:
                for ki in range(1, len(ramo)):
                    if ramo[ki-1].real * ramo[ki].real < 0:
                        k_c = min(k_c, k_arr[ki])
            return k_c
        
        def plot_lgr_ax(ax, poles, zeros, ramos, xlim, ylim):
            """Plota o LGR num eixo matplotlib com ramos, polos e zeros."""
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
            ax.fill_betweenx([ylim[0],ylim[1]], xlim[0], 0, alpha=0.05, color="seagreen")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.tick_params(labelsize=6)
            ax.spines[["right","top"]].set_visible(False)
            for ri, ramo in enumerate(ramos):
                cor = CORES_LGR[ri % len(CORES_LGR)]
                ax.plot([r.real for r in ramo], [r.imag for r in ramo],
                        color=cor, lw=1.1, alpha=0.85, zorder=3)
                ax.plot(ramo[-1].real, ramo[-1].imag, ".", color=cor, ms=3, zorder=3)
            for p in poles:
                ax.plot(p.real, p.imag, "x", color="#d62728", ms=9, mew=2.2, zorder=5)
            for z in zeros:
                ax.plot(z.real, z.imag, "o", color="#333", ms=7, mfc="white",
                        mew=1.8, zorder=5)
        
        # ── CSS responsivo + show_fig ─────────────────────────────────────────────────
        st.markdown("""
        <style>
        .fig-wrap { display:flex; justify-content:center; width:100%; }
        .fig-wrap > div { width:100%; }
        @media (min-width:769px) {
            .fig-wrap > div { width:var(--fw,65%); max-width:var(--fw,65%); }
        }
        .fig-wrap img, .fig-wrap [data-testid="stImage"] img {
            width:100% !important; height:auto !important;
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
            pct = f"{int(width_frac*100)}%"
            st.markdown(
                f'<div class="fig-wrap"><div style="--fw:{pct}">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;display:block;"/>'
                f'</div></div>', unsafe_allow_html=True)
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CABEÇALHO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.title("🔄 Análise de Sistemas com Realimentação")
        st.subheader("Estabilidade, Perturbação e Lugar Geométrico das Raízes")
        st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
        st.markdown("---")
        
        # ── Índice ────────────────────────────────────────────────────────────────────
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[5. Plantas de Ordem Superior — Estabilidade e Ganho Crítico](#5-plantas-de-ordem-superior-estabilidade-e-ganho-cr-tico)**
        - 5.1 Planta de 3ª ordem tipo 1: $H_{MF}(s) = k/[s^3+(a_1+a_2)s^2+a_1 a_2 s+k]$
        - 5.2 Critério de Routh-Hurwitz: $k_{crit} = a_1 a_2(a_1+a_2)$
        - 5.3 Trade-off fundamental: rapidez vs. amortecimento vs. margem de estabilidade
        - 🎛️ Explorador interativo: entrada, ganho $k$, atraso
        
        **[6. Erro com Perturbação](#6-erro-com-perturba-o)**
        - 6.1 Perturbação $D(s)$ na entrada da planta: superposição $H_R(s)$ e $H_D(s)$
        - 6.2 Erro de perturbação em regime permanente: $e_{rp,D} = -1/(a+k)$
        - 6.3 Estratégias para rejeição: ganho alto, integrador em $C(s)$, realimentação de estado
        - 🎛️ Explorador interativo: ganho $k$, instante de perturbação
        
        **[7. Lugar Geométrico das Raízes (LGR)](#7-lugar-geom-trico-das-ra-zes-lgr)**
        - 7.1 Definição: traçado dos polos de MF para $k \in [0, +\infty)$
        - 7.2 Regras de construção: partida/chegada, assíntotas, eixo real, cruzamento imaginário
        - 7.3 Interpretação para projeto de controladores
        - 7.4 Quadro 4×4: 16 sistemas (graus 1–4 × configurações de zeros)
        - 🎛️ LGR interativo: defina $N(s)$ e $D(s)$ e visualize o LGR completo
        
        **[8. Referências](#8-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5 — ORDEM SUPERIOR / ESTABILIDADE
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Plantas de Ordem Superior — Estabilidade e Ganho Crítico")
        
        st.markdown(r"""
        ### 5.1 Planta de 3ª ordem — Tipo 1
        
        Planta $G_p(s) = 1/[s(s+a_1)(s+a_2)]$, controlador proporcional $C(s)=k$:
        
        $$H_{MF}(s) = \frac{k}{s^3+(a_1+a_2)\,s^2+a_1 a_2\,s+k}$$
        
        Para estabilidade, o **critério de Routh-Hurwitz** para $s^3 + A\,s^2 + B\,s + C$ exige $A,B,C>0$ e $AB > C$:
        
        $$(a_1+a_2)\cdot a_1 a_2 > k \quad\Rightarrow\quad \boxed{k_{crit} = a_1 a_2(a_1+a_2)}$$
        
        ### 5.2 Trade-off fundamental
        
        | Regime | Condição | Comportamento |
        |---|---|---|
        | Estável amortecido | $k \ll k_{crit}$ | Resposta lenta, boa margem de estabilidade |
        | Subamortecido | $k < k_{crit}$ | Resposta oscilatória, ainda estável |
        | **Marginalmente estável** | $k = k_{crit}$ | Polos no eixo imaginário — oscilação sustentada |
        | Instável | $k > k_{crit}$ | Polos no SPD — amplitude crescente sem limite |
        
        > Este trade-off motiva o projeto de **controladores** (PID, compensadores): o ganho proporcional puro
        > é incapaz de satisfazer simultaneamente rapidez, amortecimento adequado **e** margem de estabilidade.
        """)
        
        # Figura estática seção 5
        a1_v5 = 2.0; a2_v5 = 3.0
        k_crit5 = a1_v5 * a2_v5 * (a1_v5 + a2_v5)
        ks5   = [1, 4, 8, k_crit5, 40]
        lbls5 = [f"k={kv}" if kv != k_crit5 else f"k=k_crit={k_crit5:.0f}" for kv in ks5]
        cols5 = [COR["ma"], "seagreen", "royalblue", "darkorange", COR["mf"]]
        t_arr5 = np.linspace(0, 40, 4000)
        u_deg5 = np.ones_like(t_arr5)
        
        fig5a, axes5a = plt.subplots(1, 2, figsize=(9.5, 3.6))
        ax = axes5a[0]
        plano_s_ax(ax, xlim=(-8,2), ylim=(-8,8))
        for kv, col, lbl in zip(ks5, cols5, lbls5):
            den5 = [1, a1_v5+a2_v5, a1_v5*a2_v5, kv]
            roots5 = np.roots(den5)
            for r in roots5:
                polo_x(ax, r.real, r.imag, cor=col, ms=9)
        ax.set_title(rf"Plano $s$ — $G(s)=k/[s(s+{a1_v5})(s+{a2_v5})]$", fontsize=8.5)
        patches5 = [mpatches.Patch(color=c, label=l) for c, l in zip(cols5, lbls5)]
        ax.legend(handles=patches5, fontsize=7, loc="upper right")
        
        ax2 = axes5a[1]
        for kv, col, lbl in zip(ks5, cols5, lbls5):
            den5 = [1, a1_v5+a2_v5, a1_v5*a2_v5, kv]
            try:
                _, y5, _ = lsim(lti([kv], den5), u_deg5, t_arr5)
                y5 = np.clip(y5, -5, 5)
            except Exception:
                y5 = np.zeros_like(t_arr5)
            ax2.plot(t_arr5, y5, color=col, lw=1.7, label=lbl)
        ax2.axhline(1.0, color=COR["ref"], lw=0.8, ls="--")
        ax2.axhline(0.0, color="k", lw=0.5)
        estilo(ax2); ax2.set_xlim(0, 40); ax2.set_ylim(-2, 3)
        ax2.set_title(rf"Resposta ao degrau — $k_{{crit}}={k_crit5:.0f}$", fontsize=8.5)
        ax2.legend(fontsize=7)
        plt.suptitle(rf"Planta 3ª ordem tipo 1: $a_1={a1_v5}$, $a_2={a2_v5}$  →  $k_{{crit}}={k_crit5:.0f}$",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig5a, 0.88)
        
        # ── Explorador seção 5 ────────────────────────────────────────────────────────
        st.markdown("### 5.3 🎛️ Explorador — Ordem Superior: Entrada, Ganho e Atraso")
        st.caption(f"$G_p(s)=1/[s(s+{a1_v5})(s+{a2_v5})]$ · Azul = estável · Vermelho = instável · $k_{{crit}}={k_crit5:.0f}$")
        
        ENTRADAS5 = ["degrau","rampa","parabola"]
        
        c5a, c5b = st.columns([1, 2])
        with c5a:
            ent5  = st.selectbox("Tipo de entrada", ENTRADAS5, key="ent5")
            k5_sl = st.slider("Ganho $k$", 0.5, float(k_crit5+15), 4.0, 0.5, key="k5sl")
            atr5  = st.slider("Atraso puro (s)", 0.0, 2.0, 0.0, 0.25, key="atr5")
        
            den5_e = [1, a1_v5+a2_v5, a1_v5*a2_v5, k5_sl]
            roots5_e = np.roots(den5_e)
            est5 = ("🔴 INSTÁVEL" if any(r.real > 1e-6 for r in roots5_e)
                    else ("🟡 CRÍTICO" if any(abs(r.real) < 1e-4 for r in roots5_e)
                          else "🟢 ESTÁVEL"))
            st.info(f"**{est5}**\n\n"
                    f"$k_{{crit}}={k_crit5:.1f}$ · $k={k5_sl:.1f}$\n\n"
                    + "\n".join([f"polo: $s={r.real:.3f}{r.imag:+.3f}j$" for r in roots5_e]))
        
        with c5b:
            t_e5 = np.linspace(0, 40, 2000)
            u_e5 = gera_entrada(ent5, t_e5, kr=1.0, atraso=atr5)
            try:
                _, y_e5, _ = lsim(lti([k5_sl], den5_e), u_e5, t_e5)
                y_e5 = np.clip(y_e5, -8, 8)
            except Exception:
                y_e5 = np.zeros_like(t_e5)
        
            cor5_e = "#d62728" if "INSTÁVEL" in est5 else ("#f97316" if "CRÍTICO" in est5 else "#1f77b4")
            fig_e5 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s","Saída y(t)"),
                                   horizontal_spacing=0.10)
            plotly_plano_s(fig_e5, 1, 1, xlim=(-8,3), ylim=(-8,8))
            fig_e5.add_trace(go.Scatter(
                x=[r.real for r in roots5_e], y=[r.imag for r in roots5_e],
                mode="markers",
                marker=dict(symbol="x", size=14, color=cor5_e, line=dict(width=3, color=cor5_e)),
                name="polos MF", showlegend=False), row=1, col=1)
            fig_e5.add_trace(go.Scatter(
                x=t_e5, y=u_e5, mode="lines",
                line=dict(color="gray", width=1.0, dash="dash"),
                name="r(t)", showlegend=False), row=1, col=2)
            fig_e5.add_trace(go.Scatter(
                x=t_e5, y=y_e5, mode="lines",
                line=dict(color=cor5_e, width=2.2),
                name="y(t)", showlegend=False), row=1, col=2)
            fig_e5.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e5.add_hline(y=0.0, line_width=0.6, line_color="black", row=1, col=2)
            fig_e5.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e5.update_yaxes(title_text="y(t)", range=[-3,4], row=1, col=2)
            fig_e5.update_layout(height=320, margin=dict(t=30,b=10,l=15,r=10),
                                 template="plotly_white")
            st.plotly_chart(fig_e5, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6 — ERRO COM PERTURBAÇÃO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Erro com Perturbação")
        
        st.markdown(r"""
        ### 6.1 Perturbação na entrada da planta
        
        Em sistemas reais, uma perturbação $D(s)$ atua sobre a entrada da planta.
        A saída resulta da superposição:
        
        $$Y(s) = \underbrace{\frac{C(s)\,G_p(s)}{1+C(s)\,G_p(s)}}_{H_R(s)}\,R(s) + \underbrace{\frac{G_p(s)}{1+C(s)\,G_p(s)}}_{H_D(s)}\,D(s)$$
        
        Para $C(s)=k$ e $G_p(s)=1/(s+a)$, com perturbação degrau $D(s)=1/s$:
        
        $$e_{rp,D} = \frac{-G_p(0)}{1+k\,G_p(0)} = \frac{-1/a}{1+k/a} = \frac{-1}{a+k}$$
        
        O sinal negativo indica que a perturbação empurra a saída **acima** da referência.
        
        > **Interpretação:** $|e_{rp,D}| = 1/(a+k)$ diminui com $k$, mas aumentar $k$ indefinidamente
        > pode desestabilizar sistemas de ordem superior (§5).
        
        ### 6.2 Estratégias para rejeição de perturbações
        
        | Estratégia | Mecanismo | Limitação |
        |---|---|---|
        | Aumentar $k$ | $\|e_{rp,D}\| = 1/(a+k) \to 0$ | Reduz margem de estabilidade |
        | Integrador em $C(s)$ | Erro nulo à perturbação degrau (tipo 1) | Pode reduzir margem de fase |
        | Realimentação de estado | Rejeição ativa com polos arbitrários | Requer observador ou sensores adicionais |
        
        > A realimentação negativa é **inerentemente robusta**: qualquer desvio de $Y(s)$ em relação
        > a $R(s)$ gera um erro que automaticamente corrige a ação de controle.
        """)
        
        # Diagrama com perturbação
        fig6d, ax6d = plt.subplots(figsize=(9.5, 2.6))
        ax6d.set_aspect("equal")
        ax6d.set_xlim(0, 12.5); ax6d.set_ylim(-1.6, 2.0)
        ax6d.axis("off")
        
        R_d=0.25; W_d=1.5; H_d=0.46
        xR=0.5; xS1=1.7; xC=3.4; xS2=5.2; xGp=7.1; xDot=9.0; xY=10.1; YF=-1.0
        
        ax6d.text(xR, 0, r"$R(s)$", ha="center", va="center", fontsize=9)
        seta(ax6d, xR+0.28, 0, xS1-R_d, 0)
        circulo(ax6d, xS1, 0, R_d)
        ax6d.text(xS1-R_d+0.02,+0.32,"+",ha="center",va="center",fontsize=11,fontweight="bold",color="seagreen")
        ax6d.text(xS1+0.30,-R_d-0.17,"−",ha="center",va="center",fontsize=13,fontweight="bold",color="crimson")
        seta(ax6d, xS1+R_d, 0, xC-W_d/2, 0, "E(s)", dy=0.12)
        bloco(ax6d, xC, 0, W_d, H_d, r"$C(s)=k$")
        seta(ax6d, xC+W_d/2, 0, xS2-R_d, 0)
        circulo(ax6d, xS2, 0, R_d)
        ax6d.text(xS2-R_d+0.02,+0.32,"+",ha="center",va="center",fontsize=11,fontweight="bold",color="seagreen")
        ax6d.text(xS2+0.33,R_d+0.12,"+",ha="center",va="center",fontsize=11,fontweight="bold",color="seagreen")
        ax6d.text(xS2, 1.55, r"$D(s)$", ha="center", va="bottom", fontsize=9)
        seta(ax6d, xS2, 1.42, xS2, R_d)
        seta(ax6d, xS2+R_d, 0, xGp-W_d/2, 0)
        bloco(ax6d, xGp, 0, W_d, H_d, r"$G_p(s)$"+"\nPlanta")
        linha(ax6d, xGp+W_d/2, 0, xDot, 0)
        ponto(ax6d, xDot, 0)
        seta(ax6d, xDot, 0, xY, 0, "Y(s)", dy=0.12)
        linha(ax6d, xDot, 0, xDot, YF)
        linha(ax6d, xDot, YF, xS1, YF)
        seta(ax6d, xS1, YF, xS1, -R_d)
        ax6d.text((xS1+xDot)/2, YF-0.22,
                  "Realimentação unitária negativa", ha="center", fontsize=8, color="gray")
        ax6d.set_title(r"Diagrama — Perturbação $D(s)$ na entrada da planta", fontsize=9, pad=4)
        plt.tight_layout()
        show_fig(fig6d, 0.78)
        
        # Figura estática seção 6
        t_arr6 = np.linspace(0, 20, 2000)
        a_val6 = 1.0; t_pert6 = 8.0
        ks6 = [1, 4, 10, 30]
        cols6 = plt.cm.viridis(np.linspace(0.15, 0.9, len(ks6)))
        
        fig6a, axes6a = plt.subplots(1, 2, figsize=(9.5, 3.2))
        for ax_idx, (titulo, r_val, d_val) in enumerate([
            ("Seguimento de referência (D=0)", 1.0, 0.0),
            ("Rejeição de perturbação (R=0, D em t=8s)", 0.0, 1.0),
        ]):
            ax = axes6a[ax_idx]
            for kv, col in zip(ks6, cols6):
                H_R6 = lti([kv],   [1, a_val6+kv])
                H_D6 = lti([1.0],  [1, a_val6+kv])
                u_r6 = np.ones_like(t_arr6) * r_val
                u_d6 = np.where(t_arr6 >= t_pert6, d_val, 0.0)
                if r_val > 0:
                    _, y6, _ = lsim(H_R6, u_r6, t_arr6)
                else:
                    _, y6, _ = lsim(H_D6, u_d6, t_arr6)
                ax.plot(t_arr6, y6, color=col, lw=1.7, label=f"k={kv}")
            if ax_idx == 1:
                ax.axvline(t_pert6, color="purple", lw=1.0, ls=":", label=f"perturbação t={t_pert6}s")
            ax.axhline(0, color="k", lw=0.5)
            estilo(ax); ax.set_xlim(0, 20)
            ax.set_title(titulo, fontsize=8.5); ax.legend(fontsize=7)
        plt.suptitle(rf"Efeito do ganho $k$ no seguimento e na rejeição de perturbação ($a={a_val6}$)",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig6a, 0.88)
        
        # ── Explorador seção 6 ────────────────────────────────────────────────────────
        st.markdown("### 6.3 🎛️ Explorador — Seguimento e Rejeição de Perturbação")
        st.caption("Painel esq.: seguimento de referência · Painel dir.: rejeição da perturbação degrau")
        
        c6a, c6b = st.columns([1, 2])
        with c6a:
            k6_sl   = st.slider("Ganho $k$", 0.5, 30.0, 4.0, 0.5, key="k6sl")
            tpert6_sl = st.slider("Instante da perturbação (s)", 1.0, 18.0, 8.0, 0.5, key="tpert6")
            erp6_seg = a_val6/(a_val6+k6_sl)
            erp6_pert = -1/(a_val6+k6_sl)
            st.info(f"$k={k6_sl:.1f}$  ·  $a={a_val6}$\n\n"
                    f"$e_{{rp}}(\\text{{seguimento}}) = {erp6_seg:.4f}$\n\n"
                    f"$e_{{rp,D}}(\\text{{perturbação}}) = {erp6_pert:.4f}$")
        
        with c6b:
            t_e6   = np.linspace(0, 20, 2000)
            H_R6_e = lti([k6_sl],  [1, a_val6+k6_sl])
            H_D6_e = lti([1.0],    [1, a_val6+k6_sl])
            u_r6_e = np.ones_like(t_e6)
            u_d6_e = np.where(t_e6 >= tpert6_sl, 1.0, 0.0)
            _, y_seg6, _  = lsim(H_R6_e, u_r6_e, t_e6)
            _, y_pert6, _ = lsim(H_D6_e, u_d6_e, t_e6)
        
            fig_e6 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Seguimento (D=0)","Perturbação (R=0)"))
            fig_e6.add_trace(go.Scatter(x=t_e6, y=u_r6_e, mode="lines",
                line=dict(color="gray",width=1.1,dash="dash"), name="r(t)",
                showlegend=False), row=1, col=1)
            fig_e6.add_trace(go.Scatter(x=t_e6, y=y_seg6, mode="lines",
                line=dict(color="#1f77b4",width=2.2), name="y(t) seg.",
                showlegend=False), row=1, col=1)
            fig_e6.add_trace(go.Scatter(x=t_e6, y=y_pert6, mode="lines",
                line=dict(color="#d62728",width=2.2), name="y(t) pert.",
                showlegend=False), row=1, col=2)
            fig_e6.add_vline(x=tpert6_sl, line_dash="dot", line_color="purple", row=1, col=2)
            fig_e6.add_hline(y=0, line_width=0.6, line_color="black", row=1, col=1)
            fig_e6.add_hline(y=0, line_width=0.6, line_color="black", row=1, col=2)
            fig_e6.update_xaxes(title_text="t (s)", row=1, col=1)
            fig_e6.update_yaxes(title_text="y(t)",  row=1, col=1)
            fig_e6.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e6.update_yaxes(title_text="y(t)",  row=1, col=2)
            fig_e6.update_layout(height=300, margin=dict(t=30,b=10,l=15,r=10),
                                 template="plotly_white")
            st.plotly_chart(fig_e6, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7 — LGR
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("7. Lugar Geométrico das Raízes (LGR)")
        
        st.markdown(r"""
        ### 7.1 Definição e motivação
        
        O **LGR** é o traçado de todos os polos de malha fechada no plano $s$ conforme $k$ varia de $0$ a $+\infty$.
        
        Para $H_{MF}(s) = k\,G(s)/[1+k\,G(s)]$ com $G(s)=N(s)/D(s)$, os polos satisfazem a **equação característica**:
        
        $$1 + k\,G(s) = 0 \;\Longrightarrow\; D(s) + k\,N(s) = 0$$
        
        O LGR é o conjunto de todos os $s \in \mathbb{C}$ que satisfazem essa equação para algum $k \geq 0$.
        
        ### 7.2 Regras de construção
        
        | Regra | Descrição |
        |---|---|
        | **Partida** ($k=0$) | Ramos partem dos **polos de MA** $\{p_i\}$ |
        | **Chegada** ($k\to\infty$) | Ramos chegam aos **zeros de MA** $\{z_j\}$; os $n-m$ restantes vão ao $\infty$ |
        | **Número de ramos** | Igual ao número de polos de MA $n$ |
        | **Simetria** | LGR é simétrico em relação ao eixo real |
        | **Eixo real** | Pertence ao LGR se há número **ímpar** de polos+zeros à direita |
        | **Assíntotas** ($n>m$) | Ângulo $\dfrac{(2q+1)\pi}{n-m}$, centradas em $\sigma_a = \dfrac{\sum p_i - \sum z_j}{n-m}$ |
        | **Cruzamento do eixo Im.** | Critério de Routh ou $s=j\omega$ na eq. característica → $k_{crit}$ |
        
        ### 7.3 Interpretação para projeto
        
        - $k=0$: polos MF = polos MA → sistema em **malha aberta**
        - $k \to \infty$: polos MF → zeros MA (ou infinito pelas assíntotas)
        - **Projeto**: escolher $k$ tal que os polos de MF tenham $\xi$ e $\omega_n$ desejados no SPE
        
        > O LGR mostra **simultaneamente** toda a trajetória dos polos — é o mapa completo do
        > comportamento do sistema em função do ganho proporcional.
        """)
        
        # ── Quadro 4×4 ────────────────────────────────────────────────────────────────
        st.markdown("### 7.4 Quadro 4×4 — 16 Sistemas")
        st.caption("Linhas: grau $n=1$ a $4$ · Colunas: configurações de zeros · × = polo MA · ○ = zero MA")
        
        SISTEMAS_LGR = [
            # n=1
            {"p":[-2],         "z":[],            "titulo":"n=1 | sem zeros",              "xl":(-5,0.5), "yl":(-1,1)},
            {"p":[-2],         "z":[-5],          "titulo":"n=1 | z=-5 (dir. ao polo)",    "xl":(-7,0.5), "yl":(-1,1)},
            {"p":[-2],         "z":[-0.5],        "titulo":"n=1 | z=-0.5 (esq. ao polo)",  "xl":(-3,0.5), "yl":(-1,1)},
            {"p":[-2],         "z":[1],           "titulo":"n=1 | z=+1 (fase não-mínima)", "xl":(-3,2),   "yl":(-1,1)},
            # n=2
            {"p":[-1,-3],      "z":[],            "titulo":"n=2 | sem zeros",              "xl":(-7,0.5), "yl":(-4,4)},
            {"p":[-1,-3],      "z":[-2],          "titulo":"n=2 | z=-2 (entre polos)",     "xl":(-6,0.5), "yl":(-4,4)},
            {"p":[-1,-3],      "z":[-0.5],        "titulo":"n=2 | z=-0.5",                 "xl":(-4,0.5), "yl":(-4,4)},
            {"p":[-1,-3],      "z":[-1+1j,-1-1j], "titulo":"n=2 | z complexos -1±j",      "xl":(-5,0.5), "yl":(-4,4)},
            # n=3
            {"p":[0,-2,-4],    "z":[],            "titulo":"n=3 | integrador + 2 reais",   "xl":(-8,0.5), "yl":(-6,6)},
            {"p":[0,-2,-4],    "z":[-1],          "titulo":"n=3 | z=-1",                   "xl":(-7,0.5), "yl":(-6,6)},
            {"p":[0,-2,-4],    "z":[-1,-3],       "titulo":"n=3 | z=-1,-3",                "xl":(-7,0.5), "yl":(-6,6)},
            {"p":[0,-2,-4],    "z":[-1+2j,-1-2j], "titulo":"n=3 | z complexos -1±2j",     "xl":(-7,0.5), "yl":(-6,6)},
            # n=4
            {"p":[0,-1,-3,-6], "z":[],            "titulo":"n=4 | integrador + 3 polos",   "xl":(-10,0.5),"yl":(-8,8)},
            {"p":[0,-1,-3,-6], "z":[-2],          "titulo":"n=4 | z=-2",                   "xl":(-9,0.5), "yl":(-8,8)},
            {"p":[0,-1,-3,-6], "z":[-2,-4],       "titulo":"n=4 | z=-2,-4",                "xl":(-8,0.5), "yl":(-8,8)},
            {"p":[0,-1,-3,-6], "z":[-2+3j,-2-3j], "titulo":"n=4 | z complexos -2±3j",     "xl":(-9,0.5), "yl":(-8,8)},
        ]
        LABELS_ROW7 = ["n = 1","n = 2","n = 3","n = 4"]
        LABELS_COL7 = ["Sem zeros","1 zero real","2 zeros reais","Zeros complexos"]
        
        @st.cache_data(show_spinner="Calculando quadro de LGR…")
        def calcular_quadro_lgr():
            resultados = []
            for sys7 in SISTEMAS_LGR:
                poles7 = [complex(p) for p in sys7["p"]]
                zeros7 = [complex(z) for z in sys7["z"]]
                num7 = np.poly(zeros7) if zeros7 else np.array([1.0])
                den7 = np.poly(poles7)
                ramos7, _ = calc_lgr(num7, den7, k_max=500)
                resultados.append((poles7, zeros7, ramos7, sys7["xl"], sys7["yl"], sys7["titulo"]))
            return resultados
        
        resultados_lgr = calcular_quadro_lgr()
        
        fig7q, axs7q = plt.subplots(4, 4, figsize=(13, 12))
        for col7, lbl_c in enumerate(LABELS_COL7):
            axs7q[0,col7].set_title(f"Col {col7+1}: {lbl_c}", fontsize=8, fontweight="bold",
                                     color="#333", pad=4)
        for idx7, (poles7,zeros7,ramos7,xl7,yl7,tit7) in enumerate(resultados_lgr):
            row7 = idx7//4; col7 = idx7%4
            ax7  = axs7q[row7,col7]
            plot_lgr_ax(ax7, poles7, zeros7, ramos7, xl7, yl7)
            ax7.set_title(tit7, fontsize=7.5, pad=3)
            if col7 == 0:
                ax7.set_ylabel(LABELS_ROW7[row7], fontsize=9, fontweight="bold",
                               color="#444", labelpad=4)
        plt.suptitle("Lugar Geométrico das Raízes — Quadro 4×4\n"
                     "Linhas: grau n=1 a 4 | Colunas: configurações de zeros",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        show_fig(fig7q, 0.98)
        
        # ── LGR Interativo ────────────────────────────────────────────────────────────
        st.markdown("### 7.5 🎛️ LGR Interativo — Defina $G(s) = N(s)/D(s)$")
        st.caption("Insira os coeficientes em **ordem decrescente** de $s$ (ex.: `1, 3` = $s+3$; `1, 6, 8, 0` = $s^3+6s^2+8s$)")
        
        st.markdown("""
        **Exemplos prontos:**
        
        | $G(s)$ | Numerador | Denominador |
        |:---|:---|:---|
        | $1/[s(s+2)(s+4)]$ | `1` | `1, 6, 8, 0` |
        | $(s+1)/[s^2(s+3)(s+5)]$ | `1, 1` | `1, 8, 15, 0, 0` |
        | $1/(s^2+2s+5)$ | `1` | `1, 2, 5` |
        | $(s+2)/(s^3+6s^2+11s+6)$ | `1, 2` | `1, 6, 11, 6` |
        | $1/[s(s+2)(s+4)(s+6)]$ | `1` | `1, 12, 44, 48, 0` |
        """)
        
        c7a, c7b = st.columns([1, 2])
        with c7a:
            num_str7 = st.text_input("Numerador $N(s)$", value="1", key="num7")
            den_str7 = st.text_input("Denominador $D(s)$", value="1, 6, 8, 0", key="den7")
            kmax_exp7 = st.slider("$k_{max}$ (potência de 10)", 0.5, 5.0, 3.0, 0.5, key="kmax7")
        
            # parse e validação
            lgr_ok = False
            try:
                num7_c = [float(x.strip()) for x in num_str7.split(",") if x.strip()]
                den7_c = [float(x.strip()) for x in den_str7.split(",") if x.strip()]
                assert len(den7_c) > len(num7_c), \
                    f"Grau de Den ({len(den7_c)-1}) deve ser > grau de Num ({len(num7_c)-1})."
                k_max7 = float(10 ** kmax_exp7)
                lgr_ok = True
            except AssertionError as e:
                st.error(str(e))
            except ValueError:
                st.error("Formato inválido — use números separados por vírgula.")
        
        with c7b:
            if lgr_ok:
                try:
                    ramos7_i, k_arr7_i = calc_lgr(num7_c, den7_c, k_max=k_max7)
                    poles7_i = np.roots(den7_c)
                    zeros7_i = np.roots(num7_c) if len(num7_c) > 1 else np.array([])
                    k_c7 = k_crit_lgr(ramos7_i, k_arr7_i)
                    k_cs7 = f"{k_c7:.3f}" if k_c7 < float("inf") else "∞ (sempre estável)"
        
                    # info no painel esquerdo
                    with c7a:
                        st.info(f"**$k_{{crit}} \\approx$ {k_cs7}**\n\n"
                                f"$n={len(den7_c)-1}$ polos · $m={len(num7_c)-1}$ zeros\n\n"
                                f"Ramos ao $\\infty$: $n-m={len(den7_c)-len(num7_c)}$")
        
                    # limites automáticos
                    all_re7 = [r.real for ramo in ramos7_i for r in ramo]
                    all_im7 = [r.imag for ramo in ramos7_i for r in ramo]
                    x0_7 = min(all_re7); x1_7 = max(all_re7)
                    y0_7 = min(all_im7); y1_7 = max(all_im7)
                    xr_7 = max(x1_7-x0_7, 0.5); yr_7 = max(y1_7-y0_7, 0.5)
                    xlim7 = [x0_7-0.15*xr_7, x1_7+0.15*xr_7]
                    ylim7 = [y0_7-0.15*yr_7, y1_7+0.15*yr_7]
        
                    fig_lgr7 = go.Figure()
                    # região estável
                    fig_lgr7.add_shape(type="rect",
                        x0=xlim7[0], x1=0, y0=ylim7[0], y1=ylim7[1],
                        fillcolor="rgba(44,160,44,0.07)", line_width=0, layer="below")
                    # eixo imaginário
                    fig_lgr7.add_shape(type="line",
                        x0=0, x1=0, y0=ylim7[0], y1=ylim7[1],
                        line=dict(color="gray", width=1.2, dash="dash"))
                    # eixo real
                    fig_lgr7.add_shape(type="line",
                        x0=xlim7[0], x1=xlim7[1], y0=0, y1=0,
                        line=dict(color="#ccc", width=0.8))
                    # ramos
                    for ri7, ramo7 in enumerate(ramos7_i):
                        cor7 = CORES_LGR[ri7 % len(CORES_LGR)]
                        fig_lgr7.add_trace(go.Scatter(
                            x=[r.real for r in ramo7], y=[r.imag for r in ramo7],
                            mode="lines", line=dict(color=cor7, width=2.2),
                            name=f"Ramo {ri7+1}",
                            hovertemplate="σ=%{x:.3f}  jω=%{y:.3f}<extra>Ramo "+str(ri7+1)+"</extra>"))
                    # polos MA
                    fig_lgr7.add_trace(go.Scatter(
                        x=[p.real for p in poles7_i], y=[p.imag for p in poles7_i],
                        mode="markers",
                        marker=dict(symbol="x", size=14, color="#d62728", line=dict(width=3)),
                        name="Polos MA × (k=0)",
                        hovertemplate="polo: %{x:.3f}+%{y:.3f}j<extra></extra>"))
                    # zeros MA
                    if len(zeros7_i) > 0:
                        fig_lgr7.add_trace(go.Scatter(
                            x=[z.real for z in zeros7_i], y=[z.imag for z in zeros7_i],
                            mode="markers",
                            marker=dict(symbol="circle-open", size=13, color="#333",
                                        line=dict(width=2.2)),
                            name="Zeros MA ○ (k→∞)",
                            hovertemplate="zero: %{x:.3f}+%{y:.3f}j<extra></extra>"))
        
                    fig_lgr7.update_layout(
                        title=dict(
                            text=(f"LGR — N(s)=[{num_str7}] / D(s)=[{den_str7}]"
                                  f"  |  k<sub>crit</sub> ≈ {k_cs7}"),
                            font=dict(size=12)),
                        xaxis=dict(title="σ  (Re s)", range=xlim7,
                                   zeroline=True, zerolinecolor="gray"),
                        yaxis=dict(title="jω  (Im s)", range=ylim7,
                                   zeroline=True, zerolinecolor="gray",
                                   scaleanchor="x", scaleratio=1),
                        height=540, margin=dict(l=65,r=150,t=75,b=55),
                        template="plotly_white",
                        legend=dict(x=1.01, y=0.98, xanchor="left", yanchor="top"))
                    st.plotly_chart(fig_lgr7, use_container_width=True)
        
                except Exception as ex:
                    st.error(f"Erro no cálculo do LGR: {ex}")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 8 — REFERÊNCIAS
        # ═══════════════════════════════════════════════════════════════════════════════
        with st.expander("8. Referências", expanded=False):
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
            "📍 Análise de Sistemas com Realimentação — Estabilidade, Perturbação e LGR"
            "🎛️ SINTONIA — Sistemas de Controle<br>"
            "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
            " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
            "</div>",
            unsafe_allow_html=True,
        )
