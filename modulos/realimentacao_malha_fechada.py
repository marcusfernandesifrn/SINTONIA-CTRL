"""
🔄 Análise de Sistemas com Realimentação — Sistemas de Ordem 1 e 2
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
            page_title="Realimentação — Malha Fechada",
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
            ma      = "steelblue",
            mf      = "crimson",
            ref     = "gray",
            err     = "darkorange",
            pert    = "purple",
            degrau  = "royalblue",
            rampa   = "seagreen",
            parab   = "darkorange",
            bloco_f = "#FFFFFF",
            bloco_e = "#000000",
        )
        
        # ── Helpers matplotlib ────────────────────────────────────────────────────────
        def estilo(ax, xlabel="t (s)", ylabel="Amplitude"):
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        def polo_x(ax, x, y, cor="#d62728", ms=11):
            ax.plot(x, y, "x", color=cor, ms=ms, mew=2.5, zorder=5)
        
        def bloco(ax, x, y, w, h, txt, fc="white", fs=8.5):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x-w/2, y-h/2), w, h,
                boxstyle="round,pad=0.04", facecolor=fc, edgecolor="k", lw=1.4, zorder=3))
            ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4)
        
        def seta(ax, x1, y1, x2, y2, lbl="", dy=0.12):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4, shrinkA=0, shrinkB=0), zorder=2)
            if lbl:
                ax.text((x1+x2)/2, (y1+y2)/2+dy, lbl, ha="center", fontsize=8)
        
        def linha(ax, x1, y1, x2, y2):
            ax.plot([x1, x2], [y1, y2], color="k", lw=1.4, zorder=2)
        
        def circulo(ax, x, y, r=0.22):
            ax.add_patch(plt.Circle((x, y), r, fc="white", ec="k", lw=1.4, zorder=3))
        
        def ponto(ax, x, y):
            ax.plot(x, y, "o", color="k", ms=4, zorder=5)
        
        def plano_s_ax(ax, xlim=(-8, 2), ylim=(-5, 5)):
            ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
            ax.fill_betweenx([ylim[0], ylim[1]], xlim[0], 0, alpha=0.06, color="seagreen")
            ax.fill_betweenx([ylim[0], ylim[1]], 0, xlim[1], alpha=0.06, color="crimson")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\sigma$", fontsize=8); ax.set_ylabel(r"$j\omega$", fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        # ── Helpers Plotly ────────────────────────────────────────────────────────────
        def plotly_plano_s(fig, row, col, xlim=(-8, 2), ylim=(-5, 5)):
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
        
        def simular(sys_lti, t_arr, tipo="degrau", kr=1.0, atraso=0.0):
            u = gera_entrada(tipo, t_arr, kr, atraso)
            _, y, _ = lsim(sys_lti, u, t_arr)
            return np.clip(y, -30, 30), u
        
        def cor_entrada(tipo):
            return {"degrau": COR["degrau"], "rampa": COR["rampa"],
                    "parabola": COR["parab"]}.get(tipo, "black")
        
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
        st.subheader("Sistemas de Ordem 1 e 2 em Malha Fechada")
        st.caption("Modelagem e Sistemas Lineares · Engenharia de Energia · IFRN-CNAT · Marcus V A Fernandes")
        st.markdown("---")
        
        # ── Índice ────────────────────────────────────────────────────────────────────
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[1. Estrutura de Malha Fechada — Conceitos Fundamentais](#1-estrutura-de-malha-fechada-conceitos-fundamentais)**
        - 1.1 Diagrama de blocos e função de transferência de MF
        - 1.2 Vantagens da realimentação: erro, sensibilidade, estabilidade, perturbações
        - 1.3 Tipo do sistema em malha aberta ($\nu$ polos na origem)
        
        **[2. Erro em Regime Permanente e Tipo do Sistema](#2-erro-em-regime-permanente-e-tipo-do-sistema)**
        - 2.1 Definição do erro: $E(s) = R(s)/(1+G(s))$; teorema do valor final
        - 2.2 Constantes de erro estático: $K_p$, $K_v$, $K_a$
        - 2.3 Resumo por tipo: erro ao degrau, rampa e parábola
        - 🎛️ Explorador interativo: tipo do sistema, entrada, ganho $k$
        
        **[3. Planta de 1ª Ordem em Malha Fechada](#3-planta-de-1-ordem-em-malha-fechada)**
        - 3.1 Tipo 0 ($C(s)=k$): polo MF $s=-(a+k)$, $\tau_{MF}=1/(a+k)$, $e_{rp}=a/(a+k)$
        - 3.2 Tipo 1 ($C(s)=k/s$): MF de 2ª ordem, $\omega_n=\sqrt{k}$, $\xi=a/(2\sqrt{k})$
        - 🎛️ Explorador interativo: entrada, ganho $k$, atraso puro
        
        **[4. Planta de 2ª Ordem em Malha Fechada](#4-planta-de-2-ordem-em-malha-fechada)**
        - 4.1 Tipo 0 ($C(s)=k$): $\omega_n^{MF}=\omega_n\sqrt{1+k}$, $\xi^{MF}=\xi/\sqrt{1+k}$
        - 4.2 Tipo 1 ($C(s)=k/s$): MF de 3ª ordem, $e_{rp}(\text{rampa})=1/k$
        - 4.3 Tipo 2 (dois integradores): erro nulo à parábola
        - 🎛️ Explorador interativo: entrada, ganho $k$, atraso puro
        
        **[5. Referências](#5-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1 — ESTRUTURA DE MALHA FECHADA
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Estrutura de Malha Fechada — Conceitos Fundamentais")
        
        st.markdown(r"""
        ### 1.1 Diagrama de blocos
        
        A configuração padrão de **realimentação negativa unitária** é:
        
        $$E(s) = R(s) - Y(s), \qquad Y(s) = G(s)\,E(s), \qquad \boxed{H_{MF}(s) = \frac{G(s)}{1+G(s)}}$$
        
        onde $G(s) = C(s)\,G_p(s)$ é o produto do **controlador** $C(s)$ pela **planta** $G_p(s)$.
        Nas seções 3–4 considera-se $C(s)=k$ (proporcional) ou $C(s)=k/s$ (proporcional-integral).
        """)
        
        # Diagrama de malha fechada
        fig_d1, ax_d1 = plt.subplots(figsize=(9, 2.4))
        ax_d1.set_aspect("equal")
        ax_d1.set_xlim(0, 11); ax_d1.set_ylim(-1.5, 1.5)
        ax_d1.axis("off")
        
        R = 0.25; W = 1.5; H = 0.46
        xR=0.5; xS1=1.7; xC=3.5; xGp=5.8; xDot=7.6; xY=8.8; YF=-1.0
        
        ax_d1.text(xR, 0, r"$R(s)$", ha="center", va="center", fontsize=9)
        seta(ax_d1, xR+0.28, 0, xS1-R, 0)
        circulo(ax_d1, xS1, 0, R)
        ax_d1.text(xS1-R+0.02, +0.32, "+", ha="center", va="center",
                   fontsize=11, fontweight="bold", color="seagreen")
        ax_d1.text(xS1+0.30, -R-0.17, "−", ha="center", va="center",
                   fontsize=13, fontweight="bold", color="crimson")
        seta(ax_d1, xS1+R, 0, xC-W/2, 0, "E(s)", dy=0.12)
        bloco(ax_d1, xC, 0, W, H, r"$C(s)$" + "\nControlador")
        seta(ax_d1, xC+W/2, 0, xGp-W/2, 0)
        bloco(ax_d1, xGp, 0, W, H, r"$G_p(s)$" + "\nPlanta")
        linha(ax_d1, xGp+W/2, 0, xDot, 0)
        ponto(ax_d1, xDot, 0)
        seta(ax_d1, xDot, 0, xY, 0, "Y(s)", dy=0.12)
        linha(ax_d1, xDot, 0,  xDot, YF)
        linha(ax_d1, xDot, YF, xS1,  YF)
        seta(ax_d1, xS1,  YF, xS1,  -R)
        ax_d1.text((xS1+xY)/2, 1.15,
            r"$H_{MF}(s)=\dfrac{C(s)\,G_p(s)}{1+C(s)\,G_p(s)}$",
            ha="center", fontsize=9.5, color="navy")
        ax_d1.text((xS1+xDot)/2, YF-0.22,
            "Realimentação unitária negativa", ha="center", fontsize=8, color="gray")
        ax_d1.set_title("Diagrama — Controlador + Planta em Malha Fechada", fontsize=9, pad=4)
        plt.tight_layout()
        show_fig(fig_d1, 0.78)
        
        st.markdown(r"""
        ### 1.2 Vantagens da realimentação
        
        | Aspecto | Malha Aberta | Malha Fechada |
        |---|---|---|
        | Erro em regime permanente | Não controlado | Pode ser reduzido ou zerado |
        | Sensibilidade a variações da planta | Alta | Reduzida pelo fator $1+G(s)$ |
        | Estabilidade | Depende só de $G_p(s)$ | Pode estabilizar plantas com polo real instável$^*$ |
        | Rejeição a perturbações | Nenhuma | Significativa |
        
        > $^*$A realimentação proporcional **não estabiliza qualquer planta instável**. Um polo instável real simples
        > pode ser estabilizado com ganho suficiente; sistemas com múltiplos polos instáveis ou com atraso puro
        > podem requerer estratégias mais elaboradas.
        
        ### 1.3 Tipo do sistema em malha aberta
        
        O **tipo** do sistema é o número $\nu$ de polos na origem de $G(s) = C(s)\,G_p(s)$:
        
        $$G(s) = \frac{k\,N(s)}{s^\nu\,D(s)}, \quad \nu = \text{tipo do sistema}$$
        
        | Tipo $\nu$ | Polos em $s=0$ | $e_{rp}$ degrau | $e_{rp}$ rampa | $e_{rp}$ parábola |
        |---|---|---|---|---|
        | 0 | 0 | Finito: $k_r/(1+K_p)$ | $\infty$ | $\infty$ |
        | 1 | 1 | $0$ | Finito: $k_r/K_v$ | $\infty$ |
        | 2 | 2 | $0$ | $0$ | Finito: $k_r/K_a$ |
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2 — ERRO EM REGIME PERMANENTE
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Erro em Regime Permanente e Tipo do Sistema")
        
        st.markdown(r"""
        ### 2.1 Definição do erro
        
        O **erro de rastreamento** e o **Teorema do Valor Final** fornecem:
        
        $$E(s) = \frac{R(s)}{1+G(s)}, \qquad e_{rp} = \lim_{t\to\infty} e(t) = \lim_{s\to 0}\,\frac{s\,R(s)}{1+G(s)}$$
        
        ### 2.2 Constantes de erro estático
        
        | Constante | Definição | Entrada | Erro em regime permanente |
        |---|---|---|---|
        | Posição $K_p$ | $\lim_{s\to 0} G(s)$ | Degrau $R(s)=k_r/s$ | $e_{rp} = k_r/(1+K_p)$ |
        | Velocidade $K_v$ | $\lim_{s\to 0} s\,G(s)$ | Rampa $R(s)=k_r/s^2$ | $e_{rp} = k_r/K_v$ |
        | Aceleração $K_a$ | $\lim_{s\to 0} s^2 G(s)$ | Parábola $R(s)=k_r/s^3$ | $e_{rp} = k_r/K_a$ |
        
        > Para erro nulo à rampa, $G(s)$ deve ter ao menos **um polo em $s=0$** (tipo $\geq 1$).
        > Para erro nulo à parábola, tipo $\geq 2$.
        
        ### 2.3 Resumo por tipo
        
        | Tipo | $K_p$ | $K_v$ | $K_a$ | $e_{rp}$ degrau | $e_{rp}$ rampa | $e_{rp}$ parábola |
        |---|---|---|---|---|---|---|
        | 0 | Finito | $0$ | $0$ | $k_r/(1+K_p)$ | $\infty$ | $\infty$ |
        | 1 | $\infty$ | Finito | $0$ | $0$ | $k_r/K_v$ | $\infty$ |
        | 2 | $\infty$ | $\infty$ | Finito | $0$ | $0$ | $k_r/K_a$ |
        
        > Para $G(s) = k/(s+a)$ (tipo 0): $K_p = k/a$, $e_{rp}(\text{degrau}) = k_r\,a/(a+k)$.
        > Para $G(s) = k/[s(s+a)]$ (tipo 1): $K_v = k/a$, $e_{rp}(\text{rampa}) = k_r\,a/k$.
        """)
        
        # Figura estática: comparação tipo 0 vs tipo 1
        t_arr2 = np.linspace(0, 12, 1200)
        k2 = 2.0; a2 = 1.0
        u_deg2 = np.ones_like(t_arr2)
        u_ram2 = t_arr2.copy()
        H0_2 = lti([k2], [1, a2 + k2])
        H1_2 = lti([k2], [1, a2, k2])
        _, y0_d2, _ = lsim(H0_2, u_deg2, t_arr2)
        _, y1_d2, _ = lsim(H1_2, u_deg2, t_arr2)
        _, y0_r2, _ = lsim(H0_2, u_ram2, t_arr2)
        _, y1_r2, _ = lsim(H1_2, u_ram2, t_arr2)
        
        fig2a, axes2a = plt.subplots(1, 2, figsize=(9.5, 3.2))
        ax = axes2a[0]
        ax.plot(t_arr2, u_deg2, color=COR["ref"], lw=1.2, ls="--", label="referência (degrau)")
        ax.plot(t_arr2, y0_d2, color=COR["ma"],  lw=2.0, label=rf"Tipo 0 — $e_{{rp}}={a2/(a2+k2):.2f}$")
        ax.plot(t_arr2, y1_d2, color=COR["mf"],  lw=2.0, label=r"Tipo 1 — $e_{rp}=0$")
        estilo(ax); ax.set_xlim(0, 12); ax.set_title("Entrada degrau ($k_r=1$)", fontsize=8.5)
        ax.legend(fontsize=7)
        
        ax2 = axes2a[1]
        ax2.plot(t_arr2, u_ram2,        color=COR["ref"], lw=1.2, ls="--", label="referência (rampa)")
        ax2.plot(t_arr2, y0_r2,         color=COR["ma"],  lw=2.0, label=r"Tipo 0 — $e_{rp}=\infty$")
        ax2.plot(t_arr2, np.clip(y1_r2, -1, 14), color=COR["mf"], lw=2.0,
                 label=rf"Tipo 1 — $e_{{rp}}={1/k2:.2f}$")
        estilo(ax2); ax2.set_xlim(0, 12); ax2.set_ylim(-0.5, 13)
        ax2.set_title("Entrada rampa ($k_r=1$)", fontsize=8.5)
        ax2.legend(fontsize=7)
        plt.suptitle(rf"Comparação Tipo 0 vs Tipo 1 — $k={k2}$, $a={a2}$",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig2a, 0.82)
        
        # ── Explorador seção 2 ────────────────────────────────────────────────────────
        st.markdown("### 🎛️ Explorador — Tipo do Sistema, Entrada e Ganho $k$")
        st.caption("Selecione o **tipo do sistema** e o **tipo de entrada**, depois ajuste $k$ para ver $y(t)$, $e(t)$ e $e_{rp}$.")
        
        TIPOS_LBL  = ["Tipo 0  G=k/(s+a)",
                      "Tipo 1  G=k/[s(s+a)]",
                      "Tipo 2  G=k/[s²(s+a)]"]
        ENTRADAS2  = ["degrau", "rampa", "parabola"]
        a_val2 = 1.0
        
        c2a, c2b = st.columns([1, 2])
        with c2a:
            tipo2    = st.selectbox("Tipo do sistema", TIPOS_LBL, key="tipo2")
            entrada2 = st.selectbox("Tipo de entrada", ENTRADAS2, key="entrada2")
            k2_sl    = st.slider("Ganho $k$", 0.5, 10.0, 2.0, 0.5, key="k2sl")
            ti2 = TIPOS_LBL.index(tipo2)
            ei2 = ENTRADAS2.index(entrada2)
        
            # calcular e_rp
            Kp2 = k2_sl / a_val2 if ti2 == 0 else float("inf")
            Kv2 = (k2_sl if ti2 == 1 else (0 if ti2 == 0 else float("inf")))
            Ka2 = k2_sl if ti2 == 2 else 0
            if   ei2 == 0:
                erp2 = 1/(1+Kp2) if Kp2 != float("inf") else 0
            elif ei2 == 1:
                erp2 = (1/Kv2 if Kv2 not in (0, float("inf")) else
                        (float("inf") if Kv2 == 0 else 0))
            else:
                erp2 = (1/Ka2 if Ka2 not in (0, float("inf")) else
                        (float("inf") if Ka2 == 0 else 0))
            erp2_str = f"{erp2:.4f}" if erp2 != float("inf") else "∞"
        
            # função de transferência MF
            if   ti2 == 0: sys_mf2 = lti([k2_sl], [1, a_val2 + k2_sl])
            elif ti2 == 1: sys_mf2 = lti([k2_sl], [1, a_val2, k2_sl])
            else:           sys_mf2 = lti([k2_sl], [1, a_val2, 0, k2_sl])
        
            st.info(f"**$e_{{rp}}$ = {erp2_str}**\n\n"
                    f"$K_p={Kp2:.3f}$  $K_v={Kv2:.3f}$  $K_a={Ka2:.3f}$")
        
        with c2b:
            t_e2 = np.linspace(0, 15, 1200)
            u_e2 = gera_entrada(entrada2, t_e2, kr=1.0, atraso=0.0)
            try:
                _, y_e2, _ = lsim(sys_mf2, u_e2, t_e2)
                y_e2 = np.clip(y_e2, -20, 20)
            except Exception:
                y_e2 = np.zeros_like(t_e2)
            e_e2 = u_e2 - y_e2
        
            fig_e2 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Saída y(t) vs Referência", "Erro e(t)"))
            fig_e2.add_trace(go.Scatter(x=t_e2, y=u_e2, mode="lines",
                line=dict(color="gray", width=1.2, dash="dash"),
                name="r(t)", showlegend=True), row=1, col=1)
            fig_e2.add_trace(go.Scatter(x=t_e2, y=y_e2, mode="lines",
                line=dict(color="#1f77b4", width=2.2),
                name="y(t)", showlegend=True), row=1, col=1)
            fig_e2.add_trace(go.Scatter(x=t_e2, y=e_e2, mode="lines",
                line=dict(color="#d62728", width=2.0),
                name="e(t)", showlegend=True), row=1, col=2)
            fig_e2.add_hline(y=0, line_width=0.8, line_color="black", row=1, col=2)
            fig_e2.update_xaxes(title_text="t (s)", row=1, col=1)
            fig_e2.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig_e2.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e2.update_yaxes(title_text="e(t)", row=1, col=2)
            fig_e2.update_layout(height=300, margin=dict(t=30,b=10,l=20,r=10),
                                 template="plotly_white",
                                 legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_e2, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3 — PLANTA DE 1ª ORDEM EM MALHA FECHADA
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Planta de 1ª Ordem em Malha Fechada")
        
        st.markdown(r"""
        ### 3.1 Tipo 0 — controlador proporcional $C(s)=k$
        
        Planta $G_p(s)=1/(s+a)$, malha aberta $G(s)=k/(s+a)$:
        
        $$H_{MF}(s) = \frac{k}{s+a+k}$$
        
        | Parâmetro | Malha Aberta | Malha Fechada |
        |---|---|---|
        | Polo | $s = -a$ | $s = -(a+k)$ |
        | Ganho DC | $k/a$ | $k/(a+k)$ |
        | Constante de tempo | $\tau_{MA} = 1/a$ | $\tau_{MF} = 1/(a+k)$ |
        | $e_{rp}$ ao degrau $k_r$ | indefinido | $a\,k_r/(a+k)$ |
        
        > **Efeito de $k$:** polo MF se afasta da origem → $\tau_{MF}$ decresce (mais rápido) e $e_{rp}$ decresce.
        > Mas o erro **nunca se anula** com controlador proporcional puro em sistema tipo 0.
        
        ### 3.2 Tipo 1 — controlador com integrador $C(s)=k/s$
        
        Malha aberta $G(s) = k/[s(s+a)]$ — tipo 1. Malha fechada (2ª ordem):
        
        $$H_{MF}(s) = \frac{k}{s^2 + a\,s + k}, \quad \omega_n^{MF} = \sqrt{k}, \quad \xi^{MF} = \frac{a}{2\sqrt{k}}$$
        
        | Entrada | $e_{rp}$ |
        |---|---|
        | Degrau | $0$ (integrador garante rastreamento perfeito) |
        | Rampa | $a/k$ (finito) |
        | Parábola | $\infty$ |
        
        > **Compromisso:** aumentar $k$ reduz $e_{rp}$ à rampa mas reduz $\xi^{MF}=a/(2\sqrt{k})$,
        > podendo tornar a resposta oscilatória.
        """)
        
        # Figura estática seção 3
        t_arr3 = np.linspace(0, 10, 800)
        a_val3 = 0.8
        ks3 = [2, 4, 8, 16, 32]
        cols_k3 = plt.cm.viridis(np.linspace(0.15, 0.9, len(ks3)))
        u_deg3 = np.ones_like(t_arr3)
        
        fig3a, axes3a = plt.subplots(1, 3, figsize=(10.5, 3.2))
        
        ax = axes3a[0]
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.7)
        ax.fill_betweenx([-0.5, 0.5], -36, 0, alpha=0.07, color="seagreen")
        ax.set_xlim(-36, 2); ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(r"$\sigma$", fontsize=8); ax.set_ylabel(r"$j\omega$", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
        polo_x(ax, -a_val3, 0, cor="k")
        ax.annotate("MA", xy=(-a_val3, 0), xytext=(-a_val3+0.5, 0.18),
                    fontsize=7, color="k",
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
        for kv, col in zip(ks3, cols_k3):
            polo_x(ax, -(a_val3+kv), 0, cor=col, ms=9)
            ax.text(-(a_val3+kv), 0.17, f"k={kv}", fontsize=6.5, ha="center", color=col)
        ax.set_title(r"Polo MF em $s=-(a+k)$", fontsize=8.5)
        
        ax2 = axes3a[1]
        k_ma3 = 4.0
        _, y_ma3, _ = lsim(lti([k_ma3], [1, a_val3]), u_deg3, t_arr3)
        ax2.plot(t_arr3, y_ma3, color="k", ls="--", lw=1.6, label="MA (k=4)")
        for kv, col in zip(ks3, cols_k3):
            _, y_mf3, _ = lsim(lti([kv], [1, a_val3+kv]), u_deg3, t_arr3)
            erp3 = a_val3/(a_val3+kv)
            ax2.plot(t_arr3, y_mf3, color=col, lw=1.7,
                     label=rf"k={kv}, $e_{{rp}}={erp3:.3f}$")
        ax2.axhline(1.0, color=COR["ref"], lw=0.8, ls=":")
        estilo(ax2); ax2.set_xlim(0, 10)
        ax2.set_title(r"MF — variação de $k$ (degrau $k_r=1$)", fontsize=8.5)
        ax2.legend(fontsize=6.5, ncol=2)
        
        ax3 = axes3a[2]
        ks_c3 = np.linspace(0.1, 50, 300)
        ax3.plot(ks_c3, a_val3/(a_val3+ks_c3), color=COR["mf"], lw=2.0)
        ax3.axhline(0, color="k", lw=0.5, ls="--")
        for kv, col in zip(ks3, cols_k3):
            ax3.plot(kv, a_val3/(a_val3+kv), "o", color=col, ms=8)
        estilo(ax3, xlabel="k", ylabel=r"$e_{rp}$")
        ax3.set_title(r"$e_{rp}$ (degrau) vs $k$", fontsize=8.5)
        ax3.set_xlim(0, 50); ax3.set_ylim(-0.02, 0.5)
        plt.suptitle(rf"Planta 1ª ordem: $G_p(s)=1/(s+{a_val3})$ — efeito do ganho",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig3a, 0.92)
        
        # ── Explorador seção 3 ────────────────────────────────────────────────────────
        st.markdown("### 3.3 🎛️ Explorador — Planta 1ª Ordem: Entrada, Ganho e Atraso")
        st.caption("Plano $s$ (esq.) · Saída $y(t)$ (centro) · Erro $e(t)$ (dir.) · × preto = polo MA · × azul = polo MF")
        
        ENTRADAS3 = ["degrau", "rampa", "parabola"]
        a_p3 = 0.8
        
        c3a, c3b = st.columns([1, 2])
        with c3a:
            ent3   = st.selectbox("Tipo de entrada", ENTRADAS3, key="ent3")
            k3_sl  = st.slider("Ganho $k$", 1.0, 20.0, 4.0, 0.5, key="k3sl")
            atr3   = st.slider("Atraso puro (s)", 0.0, 3.0, 0.0, 0.25, key="atr3")
            polo_mf3 = -(a_p3 + k3_sl)
            erp3_v = a_p3/(a_p3+k3_sl) if ent3=="degrau" else float("inf")
            erp3_s = f"{erp3_v:.4f}" if erp3_v != float("inf") else "∞"
            st.info(f"polo MA: $s={-a_p3:.2f}$  →  polo MF: $s={polo_mf3:.2f}$\n\n"
                    f"$\\tau_{{MF}}={1/(a_p3+k3_sl):.3f}$ s · $e_{{rp}}={erp3_s}$")
        
        with c3b:
            t_e3 = np.linspace(0, 15, 1500)
            u_e3 = gera_entrada(ent3, t_e3, kr=1.0, atraso=atr3)
            try:
                _, y_e3, _ = lsim(lti([k3_sl],[1, a_p3+k3_sl]), u_e3, t_e3)
                y_e3 = np.clip(y_e3, -30, 30)
            except Exception:
                y_e3 = np.zeros_like(t_e3)
            e_e3 = u_e3 - y_e3
        
            fig_e3 = make_subplots(rows=1, cols=3,
                                   subplot_titles=("Plano s", "Saída y(t)", "Erro e(t)"),
                                   horizontal_spacing=0.09)
            # plano s
            plotly_plano_s(fig_e3, 1, 1, xlim=(-22, 1), ylim=(-2, 2))
            fig_e3.add_trace(go.Scatter(x=[-a_p3], y=[0], mode="markers",
                marker=dict(symbol="x",size=14,color="black",line=dict(width=2.5)),
                name="polo MA", showlegend=True), row=1, col=1)
            fig_e3.add_trace(go.Scatter(x=[polo_mf3], y=[0], mode="markers",
                marker=dict(symbol="x",size=13,color="#1f77b4",line=dict(width=2.5)),
                name=f"polo MF ({polo_mf3:.2f})", showlegend=True), row=1, col=1)
            # linha de migração
            fig_e3.add_trace(go.Scatter(x=[-a_p3, polo_mf3], y=[0, 0], mode="lines",
                line=dict(color="gray",width=1,dash="dot"),
                showlegend=False, hoverinfo="skip"), row=1, col=1)
            # saída
            fig_e3.add_trace(go.Scatter(x=t_e3, y=u_e3, mode="lines",
                line=dict(color="gray",width=1.1,dash="dash"), name="r(t)",
                showlegend=False), row=1, col=2)
            fig_e3.add_trace(go.Scatter(x=t_e3, y=y_e3, mode="lines",
                line=dict(color="#1f77b4",width=2.2), name="y(t)",
                showlegend=False), row=1, col=2)
            # erro
            fig_e3.add_trace(go.Scatter(x=t_e3, y=e_e3, mode="lines",
                line=dict(color="#d62728",width=2.0), name="e(t)",
                showlegend=False), row=1, col=3)
            fig_e3.add_hline(y=0, line_width=0.7, line_color="black", row=1, col=3)
            fig_e3.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e3.update_yaxes(title_text="y(t)",  row=1, col=2)
            fig_e3.update_xaxes(title_text="t (s)", row=1, col=3)
            fig_e3.update_yaxes(title_text="e(t)",  row=1, col=3)
            fig_e3.update_layout(height=310, margin=dict(t=30,b=10,l=15,r=10),
                                 template="plotly_white",
                                 legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_e3, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4 — PLANTA DE 2ª ORDEM EM MALHA FECHADA
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Planta de 2ª Ordem em Malha Fechada")
        
        st.markdown(r"""
        ### 4.1 Tipo 0 — controlador proporcional $C(s)=k$
        
        Planta: $G_p(s) = \omega_n^2/(s^2+2\xi\omega_n s+\omega_n^2)$, com $C(s)=k$:
        
        $$H_{MF}(s) = \frac{k\omega_n^2}{s^2+2\xi\omega_n s+(1+k)\omega_n^2}$$
        
        | Parâmetro | Malha Aberta | Malha Fechada |
        |---|---|---|
        | $\omega_n$ | $\omega_n$ | $\omega_n^{MF} = \omega_n\sqrt{1+k}$ |
        | $\xi$ | $\xi$ | $\xi^{MF} = \xi/\sqrt{1+k}$ |
        | Ganho DC | $k$ | $k/(1+k)$ |
        | $e_{rp}$ ao degrau | indefinido | $1/(1+k)$ |
        
        > **Compromisso:** aumentar $k$ → $\omega_n^{MF}$ sobe (mais rápido) **e** $\xi^{MF}$ cai (mais oscilação).
        
        ### 4.2 Tipo 1 — controlador com integrador $C(s)=k/s$
        
        Malha aberta: $G(s) = k\omega_n^2/[s(s^2+2\xi\omega_n s+\omega_n^2)]$ — tipo 1, MF de 3ª ordem:
        
        $$H_{MF}(s) = \frac{k\omega_n^2}{s^3 + 2\xi\omega_n\,s^2 + \omega_n^2\,s + k\omega_n^2}$$
        
        $K_v = \lim_{s\to 0}s\,G(s) = k$, portanto $e_{rp}(\text{rampa}) = 1/k$.
        
        | Entrada | $e_{rp}$ |
        |---|---|
        | Degrau | $0$ |
        | Rampa | $1/k$ (finito) |
        | Parábola | $\infty$ |
        
        ### 4.3 Tipo 2 — dois integradores
        
        $G(s) = k\omega_n^2/[s^2(s+a)]$: $K_a = k\omega_n^2/a$, logo $e_{rp}(\text{parábola}) = a/(k\omega_n^2)$.
        """)
        
        # Figura estática seção 4
        t_arr4 = np.linspace(0, 10, 800)
        xi4 = 0.7; wn4 = 2.0
        ks4 = [2, 4, 8, 16, 32, 64]
        cols4 = plt.cm.plasma(np.linspace(0.1, 0.88, len(ks4)))
        u_deg4 = np.ones_like(t_arr4)
        
        fig4a, axes4a = plt.subplots(1, 3, figsize=(10.5, 3.4))
        
        ax = axes4a[0]
        plano_s_ax(ax, xlim=(-12, 2), ylim=(-15, 15))
        wd_ma4 = wn4*np.sqrt(1-xi4**2)
        polo_x(ax, -xi4*wn4,  wd_ma4, cor="k")
        polo_x(ax, -xi4*wn4, -wd_ma4, cor="k")
        ax.text(-xi4*wn4+0.2, wd_ma4+0.6, "MA", fontsize=7)
        for kv, col in zip(ks4, cols4):
            wn_mf4 = wn4*np.sqrt(1+kv)
            xi_mf4 = xi4/np.sqrt(1+kv)
            wd_mf4 = wn_mf4*np.sqrt(max(0, 1-xi_mf4**2))
            polo_x(ax, -xi_mf4*wn_mf4,  wd_mf4, cor=col, ms=9)
            polo_x(ax, -xi_mf4*wn_mf4, -wd_mf4, cor=col, ms=9)
        ax.set_title(r"Polos MF: $\omega_n^{MF}=\omega_n\sqrt{1+k}$", fontsize=8.5)
        
        ax2 = axes4a[1]
        _, y_ma4, _ = lsim(lti([wn4**2],[1,2*xi4*wn4,wn4**2]), u_deg4, t_arr4)
        ax2.plot(t_arr4, y_ma4, "k--", lw=1.6, label="MA")
        for kv, col in zip(ks4, cols4):
            _, y_mf4, _ = lsim(lti([kv*wn4**2],[1,2*xi4*wn4,(1+kv)*wn4**2]), u_deg4, t_arr4)
            xi_mf4 = xi4/np.sqrt(1+kv)
            ax2.plot(t_arr4, y_mf4, color=col, lw=1.7,
                     label=rf"k={kv} ($\xi^{{MF}}={xi_mf4:.2f}$)")
        ax2.axhline(1.0, color=COR["ref"], lw=0.8, ls=":")
        estilo(ax2); ax2.set_xlim(0, 10)
        ax2.set_title(rf"MF — $\xi={xi4}$, $\omega_n={wn4}$", fontsize=8.5)
        ax2.legend(fontsize=6.5, ncol=2)
        
        ax3 = axes4a[2]
        ks_c4 = np.linspace(0.1, 70, 300)
        xi_mf_c4 = xi4/np.sqrt(1+ks_c4)
        ax3r = ax3.twinx()
        l1,=ax3.plot(ks_c4, 1/(1+ks_c4), color=COR["mf"], lw=2.0, label=r"$e_{rp}=1/(1+k)$")
        l2,=ax3r.plot(ks_c4, xi_mf_c4, color=COR["ma"], lw=2.0, ls="--", label=r"$\xi^{MF}$")
        ax3r.axhline(1.0, color="gray", lw=0.6, ls=":")
        ax3.set_xlabel("k", fontsize=8); ax3.set_ylabel(r"$e_{rp}$", fontsize=8)
        ax3r.set_ylabel(r"$\xi^{MF}$", fontsize=8)
        ax3.set_title("Compromisso: erro vs amortecimento", fontsize=8.5)
        ax3.spines[["right","top"]].set_visible(False)
        ax3.legend(handles=[l1,l2], fontsize=7, loc="center right")
        plt.suptitle(rf"Planta 2ª ordem: $\xi={xi4}$, $\omega_n={wn4}$ — efeito do ganho",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig4a, 0.92)
        
        # Figura: e_rp vs k
        st.markdown("#### Curvas $e_{rp}$ e $\\xi^{MF}$ vs $k$")
        st.caption(f"Sistema de 2ª ordem com $\\xi={xi4}$, $\\omega_n={wn4}$. "
                   "Aumentar $k$ reduz o erro mas também reduz o amortecimento de MF.")
        
        fig4b, ax4b = plt.subplots(figsize=(5.0, 2.8))
        ks_plt = np.linspace(0.1, 30, 300)
        ax4b_r = ax4b.twinx()
        ax4b.plot(ks_plt, 1/(1+ks_plt), color=COR["mf"], lw=2.0, label=r"$e_{rp}$")
        ax4b_r.plot(ks_plt, xi4/np.sqrt(1+ks_plt), color=COR["ma"],
                     lw=2.0, ls="--", label=r"$\xi^{MF}$")
        ax4b_r.axhline(1.0, color="gray", lw=0.7, ls=":", label=r"$\xi^{MF}=1$ (crítico)")
        ax4b.set_xlabel("k", fontsize=8); ax4b.set_ylabel(r"$e_{rp}$", fontsize=8, color=COR["mf"])
        ax4b_r.set_ylabel(r"$\xi^{MF}$", fontsize=8, color=COR["ma"])
        ax4b.set_xlim(0, 30); ax4b.set_ylim(0, 0.55)
        ax4b_r.set_ylim(0, 1.2)
        ax4b.spines[["right","top"]].set_visible(False)
        ax4b.legend(loc="upper right", fontsize=7)
        ax4b_r.legend(loc="center right", fontsize=7)
        plt.tight_layout()
        show_fig(fig4b, 0.48)
        
        # ── Explorador seção 4 ────────────────────────────────────────────────────────
        st.markdown("### 4.4 🎛️ Explorador — Planta 2ª Ordem: Entrada, Ganho e Atraso")
        st.caption("Plano $s$ (esq.) · Saída $y(t)$ (centro) · Erro $e(t)$ (dir.) · × preto = polos MA · × azul = polos MF")
        
        ENTRADAS4 = ["degrau", "rampa", "parabola"]
        xi_p4 = 0.7; wn_p4 = 2.0
        
        c4a, c4b = st.columns([1, 2])
        with c4a:
            ent4  = st.selectbox("Tipo de entrada", ENTRADAS4, key="ent4")
            k4_sl = st.slider("Ganho $k$", 0.5, 20.0, 4.0, 0.5, key="k4sl")
            atr4  = st.slider("Atraso puro (s)", 0.0, 3.0, 0.0, 0.25, key="atr4")
        
            wn_mf4_e = wn_p4 * np.sqrt(1 + k4_sl)
            xi_mf4_e = xi_p4 / np.sqrt(1 + k4_sl)
            wd_mf4_e = wn_mf4_e * np.sqrt(max(0, 1-xi_mf4_e**2))
            sig_mf4_e = xi_mf4_e * wn_mf4_e
            erp4_v = 1/(1+k4_sl) if ent4 == "degrau" else float("inf")
            erp4_s = f"{erp4_v:.4f}" if erp4_v != float("inf") else "∞"
        
            st.info(f"$\\omega_n^{{MF}}={wn_mf4_e:.3f}$ · $\\xi^{{MF}}={xi_mf4_e:.3f}$\n\n"
                    f"polo MF: $s={-sig_mf4_e:.2f}\\pm{wd_mf4_e:.2f}j$\n\n"
                    f"$e_{{rp}}={erp4_s}$")
        
        with c4b:
            t_e4 = np.linspace(0, 15, 1500)
            u_e4 = gera_entrada(ent4, t_e4, kr=1.0, atraso=atr4)
            try:
                _, y_e4, _ = lsim(
                    lti([k4_sl*wn_p4**2],[1, 2*xi_p4*wn_p4, (1+k4_sl)*wn_p4**2]),
                    u_e4, t_e4)
                y_e4 = np.clip(y_e4, -20, 20)
            except Exception:
                y_e4 = np.zeros_like(t_e4)
            e_e4 = u_e4 - y_e4
        
            # polos MA
            wd_ma4_e  = wn_p4*np.sqrt(1-xi_p4**2)
            sig_ma4_e = xi_p4*wn_p4
        
            fig_e4 = make_subplots(rows=1, cols=3,
                                   subplot_titles=("Plano s","Saída y(t)","Erro e(t)"),
                                   horizontal_spacing=0.09)
            plotly_plano_s(fig_e4, 1, 1, xlim=(-14, 2), ylim=(-12, 12))
            # polos MA
            fig_e4.add_trace(go.Scatter(
                x=[-sig_ma4_e,-sig_ma4_e], y=[wd_ma4_e,-wd_ma4_e], mode="markers",
                marker=dict(symbol="x",size=14,color="black",line=dict(width=2.5)),
                name="polos MA", showlegend=True), row=1, col=1)
            # polos MF
            fig_e4.add_trace(go.Scatter(
                x=[-sig_mf4_e,-sig_mf4_e], y=[wd_mf4_e,-wd_mf4_e], mode="markers",
                marker=dict(symbol="x",size=13,color="#1f77b4",line=dict(width=2.5)),
                name=f"polos MF ({-sig_mf4_e:.2f}±{wd_mf4_e:.2f}j)", showlegend=True), row=1, col=1)
            # arco de circunferência wn_MF
            theta_arc = np.linspace(np.pi/2, np.pi*3/2, 100)
            fig_e4.add_trace(go.Scatter(
                x=wn_mf4_e*np.cos(theta_arc), y=wn_mf4_e*np.sin(theta_arc),
                mode="lines", line=dict(color="#1f77b4",width=0.8,dash="dot"),
                showlegend=False, hoverinfo="skip"), row=1, col=1)
            # saída
            fig_e4.add_trace(go.Scatter(x=t_e4, y=u_e4, mode="lines",
                line=dict(color="gray",width=1.1,dash="dash"),
                name="r(t)", showlegend=False), row=1, col=2)
            fig_e4.add_trace(go.Scatter(x=t_e4, y=y_e4, mode="lines",
                line=dict(color="#1f77b4",width=2.2),
                name="y(t)", showlegend=False), row=1, col=2)
            # erro
            fig_e4.add_trace(go.Scatter(x=t_e4, y=e_e4, mode="lines",
                line=dict(color="#d62728",width=2.0),
                name="e(t)", showlegend=False), row=1, col=3)
            fig_e4.add_hline(y=0, line_width=0.7, line_color="black", row=1, col=3)
            fig_e4.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e4.update_yaxes(title_text="y(t)",  row=1, col=2)
            fig_e4.update_xaxes(title_text="t (s)", row=1, col=3)
            fig_e4.update_yaxes(title_text="e(t)",  row=1, col=3)
            fig_e4.update_layout(height=320, margin=dict(t=30,b=10,l=15,r=10),
                                 template="plotly_white",
                                 legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_e4, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5 — REFERÊNCIAS
        # ═══════════════════════════════════════════════════════════════════════════════
        with st.expander("5. Referências", expanded=False):
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
            "Análise de Sistemas com Realimentação &nbsp;·&nbsp; Modelagem e Sistemas Lineares"
            " &nbsp;·&nbsp; Engenharia de Energia &nbsp;·&nbsp; CNAT — IFRN<br>"
            "Autor: Marcus V A Fernandes &nbsp;·&nbsp; marcus.fernandes@ifrn.edu.br"
            " &nbsp;·&nbsp; v1.0"
            "</div>",
            unsafe_allow_html=True,
        )
