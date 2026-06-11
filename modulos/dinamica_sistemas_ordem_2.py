"""
📊 Dinâmica no Domínio do Tempo — Sistemas de Ordem 2
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
            page_title="Dinâmica — Sistemas de Ordem 2",
            page_icon="📊",
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
            sub   = "royalblue",
            crit  = "seagreen",
            sobre = "crimson",
            imed  = "darkorange",
            inst  = "purple",
            imag  = "teal",
            ref   = "gray",
        )
        
        # ── Helpers matplotlib ────────────────────────────────────────────────────────
        def estilo(ax, xlabel="t (s)", ylabel="y(t)"):
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        def polo_x(ax, x, y, cor="#d62728"):
            ax.plot(x, y, "x", color=cor, ms=12, mew=2.5, zorder=5)
        
        def zero_o(ax, x, y, cor="#1f77b4"):
            ax.plot(x, y, "o", color=cor, ms=8, mfc="white", mew=2, zorder=5)
        
        def plano_s(ax, xlim=(-6, 1.5), ylim=(-5, 5)):
            ax.axhline(0, color="k", lw=0.8)
            ax.axvline(0, color="k", lw=0.8)
            ax.fill_betweenx([ylim[0], ylim[1]], xlim[0], 0, alpha=0.06, color="seagreen")
            ax.fill_betweenx([ylim[0], ylim[1]], 0, xlim[1], alpha=0.06, color="crimson")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\sigma$", fontsize=8)
            ax.set_ylabel(r"$j\omega$", fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        # ── Helpers Plotly ────────────────────────────────────────────────────────────
        def plotly_plano_s(fig, row, col, xlim=(-7, 2), ylim=(-5, 5)):
            fig.add_vrect(x0=xlim[0], x1=0, fillcolor="seagreen",
                          opacity=0.05, layer="below", line_width=0, row=row, col=col)
            fig.add_vrect(x0=0, x1=xlim[1], fillcolor="crimson",
                          opacity=0.05, layer="below", line_width=0, row=row, col=col)
            fig.update_xaxes(title_text="σ", range=list(xlim),
                             zeroline=True, zerolinecolor="black", row=row, col=col)
            fig.update_yaxes(title_text="jω", range=list(ylim),
                             zeroline=True, zerolinecolor="black", row=row, col=col)
        
        def get_poles(xi_v, wn_v):
            d = (xi_v * wn_v)**2 - wn_v**2
            if abs(d) < 1e-9:
                return [(-xi_v * wn_v, 0)]
            elif d < 0:
                wd = wn_v * np.sqrt(1 - xi_v**2)
                return [(-xi_v * wn_v, wd), (-xi_v * wn_v, -wd)]
            else:
                return [(-xi_v * wn_v + np.sqrt(d), 0),
                        (-xi_v * wn_v - np.sqrt(d), 0)]
        
        def regime(xi_v):
            if xi_v < 0:   return "INSTÁVEL"
            elif xi_v == 0: return "oscilatório puro"
            elif xi_v < 1:  return "subamortecido"
            elif abs(xi_v - 1) < 1e-9: return "criticamente amortecido"
            else:           return "sobreamortecido"
        
        def cor_xi(xi_v):
            if xi_v < 0:   return "#9467bd"
            elif xi_v == 0: return "#17becf"
            elif xi_v < 1:  return "#1f77b4"
            elif abs(xi_v - 1) < 1e-9: return "#2ca02c"
            else:           return "#d62728"
        
        def step2(k_v, xi_v, wn_v, t_arr):
            _, y = sc_step(lti([k_v * wn_v**2], [1, 2*xi_v*wn_v, wn_v**2]), T=t_arr)
            return y
        
        def specs_sub(xi_v, wn_v):
            if 0 < xi_v < 1:
                wd  = wn_v * np.sqrt(1 - xi_v**2)
                UP  = 100 * np.exp(-np.pi * xi_v / np.sqrt(1 - xi_v**2))
                Tp  = np.pi / wd
                Ts2 = 4.0 / (xi_v * wn_v)
                phi = np.arccos(xi_v)
                Tr  = (np.pi - phi) / wd
                return wd, UP, Tp, Ts2, Tr
            return 0, 0, 0, 0, 0
        
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
        st.title("📊 Dinâmica no Domínio do Tempo")
        st.subheader("Sistemas de Ordem 2")
        st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
        st.markdown("---")
        
        # ── Índice ────────────────────────────────────────────────────────────────────
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[1. Função de Transferência de 2ª Ordem — Forma Canônica](#1-fun-o-de-transfer-ncia-de-2-ordem-forma-can-nica)**
        - 1.1 Forma geral: $k$, $\omega_n$, $\xi$
        - 1.2 Polos: $s_{1,2} = -\xi\omega_n \pm \omega_n\sqrt{\xi^2-1}$
        - 1.3 Classificação por $\xi$: subamortecido, crítico, sobreamortecido, oscilatório puro, instável
        
        **[2. Tipos de Polos e Resposta ao Degrau](#2-tipos-de-polos-e-resposta-ao-degrau-unit-rio)**
        - 2.1 Subamortecido ($0<\xi<1$): senoide amortecida, envelope $e^{-\sigma t}$
        - 2.2 Criticamente amortecido ($\xi=1$): resposta mais rápida sem ultrapassagem
        - 2.3 Sobreamortecido ($\xi>1$): dois polos reais, polo dominante
        - 2.4 Oscilatório puro ($\xi=0$): oscilação sustentada
        - 2.5 Instável ($\xi<0$): amplitude crescente
        - 🎛️ Explorador interativo: slider $\xi$
        
        **[3. Especificações de Desempenho](#3-especifica-es-de-desempenho-regime-subamortecido)**
        - $UP(\%)$, $T_p$, $T_r$, $T_{s_{2\%}}$, $T_{s_{5\%}}$
        - Localização dos polos: circunferência de raio $\omega_n$, ângulo $\arccos(\xi)$
        - 🎛️ Explorador interativo: sliders $\xi$ e $\omega_n$
        
        **[4. Efeito dos Parâmetros $\xi$, $\omega_n$ e $k$](#4-efeito-dos-par-metros-xi-omega-n-e-k)**
        - 4.1 Variação de $\xi$: ângulo dos polos, UP
        - 4.2 Variação de $\omega_n$: raio dos polos, escala de tempo
        - 4.3 Variação de $k$: só $y(\infty)$ muda
        - 🎛️ Explorador interativo: sliders $\xi$, $\omega_n$, $k$
        
        **[5. Efeito dos Parâmetros $\sigma$ e $\omega_d$](#5-efeito-dos-par-metros-sigma-e-omega-d-polos-complexos)**
        - 5.1 $\sigma$ fixo, $\omega_d$ varia: frequência de oscilação muda, envelope fixo
        - 5.2 $\omega_d$ fixo, $\sigma$ varia: envelope muda, frequência fixa
        - 🎛️ Explorador interativo: sliders $\sigma$ e $\omega_d$
        
        **[6. Exemplos Físicos de Sistemas de 2ª Ordem](#6-exemplos-f-sicos-de-sistemas-de-2-ordem)**
        - Massa-mola-amortecedor: $\omega_n=\sqrt{K/M}$, $\xi=B/(2\sqrt{KM})$
        - Circuito RLC série: $\omega_n=1/\sqrt{LC}$, $\xi=(R/2)\sqrt{C/L}$
        
        **[7. Polos e Zeros Adicionais](#7-polos-e-zeros-adicionais)**
        - 7.1 Polo adicional em $s=-a$: desacelera a resposta
        - 7.2 Zero adicional em $s=-b$: fase mínima vs. fase não-mínima
        - 🎛️ Explorador interativo: sliders $a$ e $b$
        
        **[8. Sistema Oscilatório — Polos sobre o Eixo Imaginário](#8-sistema-oscilat-rio-polos-sobre-o-eixo-imagin-rio)**
        - $H(s)=k/(s^2+k)$: polos em $\pm j\sqrt{k}$, $y(t)=1-\cos(\sqrt{k}\,t)$
        - Efeito de $k$: frequência $\omega_n=\sqrt{k}$, amplitude invariante
        - 🎛️ Explorador interativo: slider $k$
        
        **[9. Referências](#9-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Função de Transferência de 2ª Ordem — Forma Canônica")
        
        st.markdown(r"""
        ### 1.1 Forma geral
        
        Um sistema de ordem 2 e grau relativo 2 (sem zeros finitos) tem a forma canônica:
        
        $$H(s) = \frac{k\,\omega_n^2}{s^2 + 2\xi\omega_n s + \omega_n^2}$$
        
        | Parâmetro | Símbolo | Significado |
        |---|---|---|
        | Ganho DC | $k$ | $y(\infty)/k_r$ para degrau $k_r$ |
        | Frequência natural | $\omega_n$ [rad/s] | Raio dos polos no plano $s$; frequência livre ($\xi=0$) |
        | Amortecimento | $\xi$ (*zeta*, adimensional) | Controla o tipo de resposta transitória |
        
        ### 1.2 Polos da função de transferência
        
        O denominador $s^2 + 2\xi\omega_n s + \omega_n^2 = 0$ tem raízes:
        
        $$s_{1,2} = -\xi\omega_n \pm \omega_n\sqrt{\xi^2 - 1}$$
        
        O discriminante $\Delta = \xi^2 - 1$ determina o regime. Para $\xi < 1$ os polos são complexos conjugados:
        
        $$s_{1,2} = -\underbrace{\xi\omega_n}_{\sigma} \pm j\underbrace{\omega_n\sqrt{1-\xi^2}}_{\omega_d}$$
        
        onde $\sigma = \xi\omega_n > 0$ é a **taxa de decaimento** e $\omega_d$ é a **frequência natural amortecida**.
        
        ### 1.3 Classificação pelo coeficiente de amortecimento $\xi$
        
        | Regime | Condição | Tipo de polos | Comportamento |
        |---|---|---|---|
        | **Subamortecido** | $0 < \xi < 1$ | Complexos conjugados no SPE | Oscila com decaimento exponencial |
        | **Criticamente amortecido** | $\xi = 1$ | Polo duplo real em $s = -\omega_n$ | Decai sem oscilação — o mais rápido sem ultrapassagem |
        | **Sobreamortecido** | $\xi > 1$ | Dois polos reais no SPE | Decai sem oscilação — mais lento que o crítico |
        | **Oscilatório puro** | $\xi = 0$ | Imaginários puros $\pm j\omega_n$ | Oscilação sustentada — marginalmente estável |
        | **Instável** | $\xi < 0$ | Parte real positiva (SPD) | Oscilação com amplitude crescente |
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Tipos de Polos e Resposta ao Degrau Unitário")
        
        st.markdown(r"""
        ### 2.1 Subamortecido ($0 < \xi < 1$)
        
        Polos complexos $s_{1,2} = -\sigma \pm j\omega_d$:
        
        $$y(t) = k\!\left[1 - \frac{e^{-\sigma t}}{\sqrt{1-\xi^2}}\,\sin(\omega_d t + \varphi)\right], \quad \varphi = \arccos(\xi), \quad t \geq 0$$
        
        Envelope de decaimento $e^{-\sigma t}$; constante de tempo $\tau = 1/\sigma = 1/(\xi\omega_n)$.
        
        ### 2.2 Criticamente amortecido ($\xi = 1$)
        
        Polo duplo real em $s = -\omega_n$:
        
        $$y(t) = k\!\left[1 - (1 + \omega_n t)\,e^{-\omega_n t}\right], \quad t \geq 0$$
        
        É a **resposta mais rápida sem ultrapassagem** — limiar entre oscilante e não oscilante.
        
        ### 2.3 Sobreamortecido ($\xi > 1$)
        
        Dois polos reais negativos distintos $s_1, s_2 < 0$:
        
        $$y(t) = k\!\left[1 + \frac{s_1\,e^{s_2 t} - s_2\,e^{s_1 t}}{s_2 - s_1}\right], \quad t \geq 0$$
        
        Sem oscilação e mais lento que o crítico. O polo mais próximo da origem domina.
        
        ### 2.4 Oscilatório puro ($\xi = 0$)
        
        Polos imaginários puros $s_{1,2} = \pm j\omega_n$:
        
        $$y(t) = k\bigl[1 - \cos(\omega_n t)\bigr], \quad t \geq 0$$
        
        Oscilação sustentada — sistema **marginalmente estável** (BIBO instável).
        
        ### 2.5 Instável ($\xi < 0$)
        
        Polos no semiplano direito. A envoltória cresce como $e^{|\sigma|t}$ — sistema **instável**.
        """)
        
        t_arr = np.linspace(0, 12, 800)
        wn_ref = 2.0; k_ref = 1.0
        casos_ref = [
            (0.0,  COR["imag"],  r"$\xi=0$ (oscilatório puro)"),
            (0.3,  COR["sub"],   r"$\xi=0.3$ (subamortecido)"),
            (0.7,  COR["sub"],   r"$\xi=0.7$ (subamortecido)"),
            (1.0,  COR["crit"],  r"$\xi=1$ (criticamente amortecido)"),
            (1.5,  COR["sobre"], r"$\xi=1.5$ (sobreamortecido)"),
            (2.5,  COR["sobre"], r"$\xi=2.5$ (sobreamortecido)"),
        ]
        fig2a, axes2a = plt.subplots(1, 2, figsize=(9.5, 3.6))
        ax_pz = axes2a[0]
        plano_s(ax_pz, xlim=(-6, 1.5), ylim=(-4, 4))
        ax_pz.set_title(r"Plano $s$ — localização dos polos", fontsize=8.5)
        ax_r  = axes2a[1]
        for xi_v, cor, lbl in casos_ref:
            _, y = sc_step(lti([k_ref*wn_ref**2],[1,2*xi_v*wn_ref,wn_ref**2]), T=t_arr)
            ax_r.plot(t_arr, y, color=cor, lw=1.7, label=lbl)
            disc = (xi_v*wn_ref)**2 - wn_ref**2
            if xi_v == 0:
                polo_x(ax_pz, 0,  wn_ref, cor); polo_x(ax_pz, 0, -wn_ref, cor)
            elif abs(disc) < 1e-9:
                polo_x(ax_pz, -xi_v*wn_ref, 0, cor)
            elif disc < 0:
                wd = wn_ref*np.sqrt(1-xi_v**2)
                polo_x(ax_pz, -xi_v*wn_ref,  wd, cor)
                polo_x(ax_pz, -xi_v*wn_ref, -wd, cor)
            else:
                p1 = -xi_v*wn_ref+np.sqrt(disc); p2 = -xi_v*wn_ref-np.sqrt(disc)
                polo_x(ax_pz, p1, 0, cor); polo_x(ax_pz, p2, 0, cor)
        ax_r.axhline(k_ref, color=COR["ref"], lw=0.8, ls="--", label=r"$y(\infty)=k$")
        estilo(ax_r); ax_r.legend(fontsize=7, ncol=2)
        ax_r.set_title(rf"Resposta ao degrau — $\omega_n={wn_ref}$, $k={k_ref}$", fontsize=8.5)
        ax_r.set_xlim(0, 12)
        plt.tight_layout()
        show_fig(fig2a, 0.88)
        
        st.markdown("### 🎛️ Explorador — Regimes de Amortecimento")
        st.caption("Varie $\\xi$ de negativo (instável) a positivo e observe a transição entre os regimes.")
        
        c2a, c2b = st.columns([1, 2])
        with c2a:
            xi2 = st.slider("Amortecimento $\\xi$", -0.5, 3.0, 0.7, 0.05, key="xi2")
            wn2 = st.slider("Freq. natural $\\omega_n$", 0.5, 5.0, 2.0, 0.1, key="wn2")
            poles_2 = get_poles(xi2, wn2)
            col2 = cor_xi(xi2)
            poles_str = ", ".join([f"({p[0]:.3f}{p[1]:+.3f}j)" for p in poles_2])
            st.info(f"**Regime:** {regime(xi2)}\n\n"
                    f"polos: {poles_str}\n\n"
                    f"σ={xi2*wn2:.3f}  ωd={wn2*np.sqrt(max(0,1-xi2**2)):.3f}")
        
        with c2b:
            t_e2 = np.linspace(0, 14, 700)
            _, y_e2 = sc_step(lti([k_ref*wn2**2],[1,2*xi2*wn2,wn2**2]), T=t_e2)
            fig_e2 = make_subplots(rows=1, cols=2, subplot_titles=("Plano s", "Resposta ao degrau"))
            plotly_plano_s(fig_e2, 1, 1, xlim=(-7, 2), ylim=(-5, 5))
            px_l = [p[0] for p in poles_2]; py_l = [p[1] for p in poles_2]
            fig_e2.add_trace(go.Scatter(x=px_l, y=py_l, mode="markers",
                marker=dict(symbol="x", size=14, color=col2,
                            line=dict(width=3, color=col2)),
                showlegend=False), row=1, col=1)
            fig_e2.add_trace(go.Scatter(x=t_e2, y=y_e2, mode="lines",
                line=dict(color=col2, width=2.5), showlegend=False), row=1, col=2)
            fig_e2.add_hline(y=k_ref, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e2.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e2.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e2.update_layout(height=320, margin=dict(t=30,b=10,l=20,r=10), template="plotly_white")
            st.plotly_chart(fig_e2, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Especificações de Desempenho — Regime Subamortecido")
        
        st.markdown(r"""
        ### 3.1 Resposta ao degrau unitário ($0 < \xi < 1$)
        
        $$\boxed{y(t) = k\!\left[1 - \frac{e^{-\xi\omega_n t}}{\sqrt{1-\xi^2}}\,\sin(\omega_d t + \varphi)\right]}, \quad \omega_d=\omega_n\sqrt{1-\xi^2},\quad\varphi=\arccos(\xi)$$
        
        ### 3.2 Especificações de desempenho
        
        | Especificação | Símbolo | Expressão analítica | Observação |
        |---|---|---|---|
        | Ultrapassagem percentual | $UP\,(\%)$ | $100\,e^{-\pi\xi/\sqrt{1-\xi^2}}$ | Depende **apenas** de $\xi$ |
        | Instante de pico | $T_p$ | $\pi/\omega_d$ | |
        | Tempo de subida (0→100%) | $T_r$ | $(\pi - \varphi)/\omega_d$ | |
        | Tempo de acomodação 2% | $T_{s_{2\%}}$ | $\approx 4/(\xi\omega_n)$ | |
        | Tempo de acomodação 5% | $T_{s_{5\%}}$ | $\approx 3/(\xi\omega_n)$ | |
        
        > Expressões válidas **apenas** para o sistema canônico de 2ª ordem sem zeros ou polos adicionais.
        
        ### 3.3 Localização dos polos e especificações
        
        Os polos ficam sobre uma **circunferência de raio $\omega_n$**, com ângulo $\theta = \arccos(\xi)$:
        
        $$|s_{1,2}| = \omega_n, \qquad \angle s_{1,2} = \pi - \arccos(\xi)$$
        
        | Aumentar… | Efeito nos polos | Efeito na resposta |
        |---|---|---|
        | $\omega_n$ | Afasta da origem (ângulo fixo) | Mais rápido: $T_p$, $T_r$, $T_s$ reduzem |
        | $\xi$ | Aproxima do eixo real (raio fixo) | Menor $UP$, maior $T_r$ |
        | $\sigma = \xi\omega_n$ | Parte real mais negativa | Envelope decai mais rápido; menor $T_s$ |
        """)
        
        xi_d = 0.4; wn_d = 2.0; k_d = 1.0
        t_d = np.linspace(0, 10, 1000)
        _, y_d = sc_step(lti([k_d*wn_d**2],[1,2*xi_d*wn_d,wn_d**2]), T=t_d)
        wd_d, UP_d, Tp_d, Ts2_d, Tr_d = specs_sub(xi_d, wn_d)
        sigma_d = xi_d * wn_d
        
        fig3a, axes3a = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax = axes3a[0]
        plano_s(ax, xlim=(-3.5, 1), ylim=(-3, 3))
        theta_circ = np.linspace(np.pi/2, np.pi*3/2, 200)
        ax.plot(wn_d*np.cos(theta_circ), wn_d*np.sin(theta_circ), "k--", lw=0.8, alpha=0.4, label=r"$|s|=\omega_n$")
        ax.plot([-sigma_d, 0], [wd_d, 0], "k:", lw=0.8)
        polo_x(ax, -sigma_d,  wd_d); polo_x(ax, -sigma_d, -wd_d)
        ax.annotate("", xy=(-sigma_d, wd_d), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="navy", shrinkA=0, shrinkB=4))
        ax.text(-sigma_d/2+0.1, wd_d/2, rf"$\omega_n={wn_d}$", fontsize=7.5, color="navy")
        ax.text(-sigma_d-0.05, 0.15, rf"$\sigma={sigma_d:.1f}$", fontsize=7.5, ha="right")
        ax.text(-sigma_d+0.05, wd_d+0.18, rf"$\omega_d={wd_d:.2f}$", fontsize=7.5)
        ax.set_title(rf"Plano $s$ — $\xi={xi_d}$, $\omega_n={wn_d}$", fontsize=8.5)
        ax.legend(fontsize=7.5)
        
        ax2 = axes3a[1]
        ax2.plot(t_d, y_d, color=COR["sub"], lw=2.0)
        ax2.axhline(k_d, color=COR["ref"], lw=0.8, ls="--")
        ax2.axhline(1.02*k_d, color="brown", lw=0.7, ls=":")
        ax2.axhline(0.98*k_d, color="brown", lw=0.7, ls=":")
        ax2.axhline((1+UP_d/100)*k_d, color=COR["sub"], lw=0.7, ls=":")
        for tv, lbl, cor in [(Tp_d,r"$T_p$",COR["sub"]), (Ts2_d,r"$T_{s_{2\%}}$","brown"), (Tr_d,r"$T_r$",COR["imed"])]:
            if tv < t_d[-1]:
                ax2.axvline(tv, color=cor, lw=0.9, ls=":")
                ax2.text(tv+0.05, 0.08, lbl, fontsize=8, color=cor)
        ax2.annotate("", xy=(Tp_d, float(np.interp(Tp_d,t_d,y_d))), xytext=(Tp_d, k_d),
            arrowprops=dict(arrowstyle="<->", color=COR["sub"], lw=1.2))
        ax2.text(Tp_d+0.1, (1+UP_d/200)*k_d, f"UP={UP_d:.1f}%", fontsize=8, color=COR["sub"])
        estilo(ax2); ax2.set_title("Especificações de desempenho", fontsize=8.5)
        ax2.set_xlim(0, 10)
        plt.tight_layout()
        show_fig(fig3a, 0.88)
        
        st.markdown("### 🎛️ Explorador — Plano $s$ + Especificações")
        st.caption("Slider **azul** = $\\xi$ (ângulo dos polos muda, raio fixo) · Slider **vermelho** = $\\omega_n$ (raio muda, ângulo fixo)")
        
        c3a, c3b = st.columns([1, 2])
        with c3a:
            xi3 = st.slider("Amortecimento $\\xi$", 0.05, 0.95, 0.4, 0.05, key="xi3")
            wn3 = st.slider("Freq. natural $\\omega_n$", 0.5, 5.0, 2.0, 0.25, key="wn3")
            wd3, UP3, Tp3, Ts2_3, Tr3 = specs_sub(xi3, wn3)
            st.info(f"$UP={UP3:.1f}\\%$ · $T_p={Tp3:.2f}$ s\n\n"
                    f"$T_r={Tr3:.2f}$ s · $T_{{s_{{2\\%}}}}={Ts2_3:.2f}$ s\n\n"
                    f"$\\omega_d={wd3:.3f}$ · $\\sigma={xi3*wn3:.3f}$")
        
        with c3b:
            t_e3 = np.linspace(0, 14, 700)
            _, y_e3 = sc_step(lti([k_d*wn3**2],[1,2*xi3*wn3,wn3**2]), T=t_e3)
            sigma3 = xi3 * wn3
            fig_e3 = make_subplots(rows=1, cols=2, subplot_titles=("Plano s","Resposta ao degrau"))
            plotly_plano_s(fig_e3, 1, 1, xlim=(-12,2), ylim=(-10,10))
            theta_c3 = np.linspace(0, 2*np.pi, 120)
            fig_e3.add_trace(go.Scatter(x=wn3*np.cos(theta_c3), y=wn3*np.sin(theta_c3),
                mode="lines", line=dict(color="black",width=0.8,dash="dash"),
                showlegend=False), row=1, col=1)
            fig_e3.add_trace(go.Scatter(x=[-sigma3,-sigma3], y=[wd3,-wd3], mode="markers",
                marker=dict(symbol="x",size=14,color="#1f77b4",line=dict(width=3,color="#1f77b4")),
                showlegend=False), row=1, col=1)
            fig_e3.add_trace(go.Scatter(x=t_e3, y=y_e3, mode="lines",
                line=dict(color="#1f77b4",width=2.5), showlegend=False), row=1, col=2)
            fig_e3.add_hline(y=k_d, line_dash="dash", line_color="gray", row=1, col=2)
            if Tp3 > 0:
                fig_e3.add_vline(x=Tp3, line_dash="dot", line_color=COR["sub"], row=1, col=2)
            if Ts2_3 > 0:
                fig_e3.add_vline(x=Ts2_3, line_dash="dot", line_color="brown", row=1, col=2)
            fig_e3.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e3.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e3.update_layout(height=320, margin=dict(t=30,b=10,l=20,r=10), template="plotly_white")
            st.plotly_chart(fig_e3, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Efeito dos Parâmetros $\\xi$, $\\omega_n$ e $k$")
        
        st.markdown(r"""
        ### 4.1 Variação de $\xi$ (amortecimento)
        
        - Altera o **regime** e a **ultrapassagem**: $UP(\%) = 100\,e^{-\pi\xi/\sqrt{1-\xi^2}}$
        - Maior $\xi$ (até 1): menor $UP$, resposta mais suave, maior $T_r$
        - Os polos permanecem sobre a mesma circunferência de raio $\omega_n$ — apenas o ângulo $\theta = \arccos(\xi)$ muda
        
        ### 4.2 Variação de $\omega_n$ (frequência natural)
        
        - Escala **todos os tempos** proporcionalmente: $T_p \propto 1/\omega_n$, $T_s \propto 1/\omega_n$
        - Polos se afastam da origem mantendo o ângulo $\theta = \arccos(\xi)$ fixo
        - $y(\infty) = k$ e $UP$ são invariantes
        
        ### 4.3 Variação de $k$ (ganho DC)
        
        - Escala apenas o **valor final**: $y(\infty) = k\,k_r$
        - Não altera a posição dos polos, velocidade ou $UP$
        """)
        
        t_arr4 = np.linspace(0, 12, 700)
        fig4, axes4 = plt.subplots(1, 3, figsize=(10.5, 3.4))
        wn_f4=2.0; k_f4=1.0
        xis4=[0.1,0.3,0.5,0.7,1.0,1.5]
        for xv, col in zip(xis4, plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(xis4)))):
            _, y = sc_step(lti([k_f4*wn_f4**2],[1,2*xv*wn_f4,wn_f4**2]), T=t_arr4)
            axes4[0].plot(t_arr4, y, color=col, lw=1.7, label=rf"$\xi={xv}$")
        axes4[0].axhline(k_f4, color=COR["ref"], lw=0.8, ls="--")
        estilo(axes4[0]); axes4[0].set_xlim(0,12)
        axes4[0].set_title(rf"Variação de $\xi$ ($\omega_n={wn_f4}$, $k={k_f4}$)", fontsize=8.5)
        axes4[0].legend(ncol=2, fontsize=7)
        
        xi_f4=0.7; k_f4b=1.0
        wns4=[0.5,1.0,2.0,3.0,5.0]
        for wv, col in zip(wns4, plt.cm.viridis(np.linspace(0.15, 0.9, len(wns4)))):
            _, y = sc_step(lti([k_f4b*wv**2],[1,2*xi_f4*wv,wv**2]), T=t_arr4)
            axes4[1].plot(t_arr4, y, color=col, lw=1.7, label=rf"$\omega_n={wv}$")
        axes4[1].axhline(k_f4b, color=COR["ref"], lw=0.8, ls="--")
        estilo(axes4[1]); axes4[1].set_xlim(0,12)
        axes4[1].set_title(rf"Variação de $\omega_n$ ($\xi={xi_f4}$, $k={k_f4b}$)", fontsize=8.5)
        axes4[1].legend(ncol=2, fontsize=7)
        
        ks4=[0.3,0.6,1.0,2.0,4.0]
        for kv, col in zip(ks4, plt.cm.plasma(np.linspace(0.15, 0.9, len(ks4)))):
            _, y = sc_step(lti([kv*wn_f4**2],[1,2*xi_f4*wn_f4,wn_f4**2]), T=t_arr4)
            axes4[2].plot(t_arr4, y, color=col, lw=1.7, label=rf"$k={kv}$")
        estilo(axes4[2]); axes4[2].set_xlim(0,12)
        axes4[2].set_title(rf"Variação de $k$ ($\xi={xi_f4}$, $\omega_n={wn_f4}$)", fontsize=8.5)
        axes4[2].legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig4, 0.92)
        
        st.markdown("### 🎛️ Explorador — Sliders $\\xi$, $\\omega_n$ e $k$")
        st.caption("Slider **azul** = $\\xi$ · **vermelho** = $\\omega_n$ · **verde** = $k$")
        
        c4a, c4b = st.columns([1, 2])
        with c4a:
            xi4  = st.slider("Amortecimento $\\xi$", 0.1, 2.0, 0.7, 0.05,  key="xi4")
            wn4  = st.slider("Freq. natural $\\omega_n$", 0.5, 5.0, 2.0, 0.25, key="wn4")
            k4   = st.slider("Ganho $k$", 0.25, 4.0, 1.0, 0.25, key="k4")
            UP4  = 100*np.exp(-np.pi*xi4/np.sqrt(1-xi4**2)) if 0<xi4<1 else 0.0
            Ts4  = 4/(xi4*wn4) if xi4 > 0 else 999
            st.info(f"$y(\\infty)={k4:.2f}$ · $UP={UP4:.1f}\\%$\n\n"
                    f"$T_{{s_{{2\\%}}}}={Ts4:.2f}$ s · polo: $s={-xi4*wn4:.3f}\\pm{wn4*np.sqrt(max(0,1-xi4**2)):.3f}j$")
        
        with c4b:
            t_e4 = np.linspace(0, 14, 700)
            _, y_e4 = sc_step(lti([k4*wn4**2],[1,2*xi4*wn4,wn4**2]), T=t_e4)
            sigma4 = xi4*wn4; wd4 = wn4*np.sqrt(max(0,1-xi4**2))
            fig_e4 = make_subplots(rows=1, cols=2, subplot_titles=("Plano s","Resposta ao degrau"))
            plotly_plano_s(fig_e4, 1, 1, xlim=(-12,2), ylim=(-10,10))
            theta_c4 = np.linspace(0, 2*np.pi, 120)
            fig_e4.add_trace(go.Scatter(x=wn4*np.cos(theta_c4), y=wn4*np.sin(theta_c4),
                mode="lines", line=dict(color="black",width=0.8,dash="dash"), showlegend=False), row=1, col=1)
            poles_4 = get_poles(xi4, wn4)
            fig_e4.add_trace(go.Scatter(x=[p[0] for p in poles_4], y=[p[1] for p in poles_4],
                mode="markers", marker=dict(symbol="x",size=14,color="#1f77b4",line=dict(width=3)),
                showlegend=False), row=1, col=1)
            fig_e4.add_trace(go.Scatter(x=t_e4, y=y_e4, mode="lines",
                line=dict(color="#1f77b4",width=2.5), showlegend=False), row=1, col=2)
            fig_e4.add_hline(y=k4, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e4.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e4.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e4.update_layout(height=320, margin=dict(t=30,b=10,l=20,r=10), template="plotly_white")
            st.plotly_chart(fig_e4, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Efeito dos Parâmetros $\\sigma$ e $\\omega_d$ — Polos Complexos")
        
        st.markdown(r"""
        Parametrizando diretamente pela posição dos polos $s_{1,2} = -\sigma \pm j\omega_d$:
        
        $$H(s) = \frac{\omega_n^2}{s^2 + 2\sigma s + \omega_n^2}, \quad \omega_n = \sqrt{\sigma^2+\omega_d^2}, \quad \xi = \frac{\sigma}{\omega_n}$$
        
        Esta parametrização é útil porque $\sigma$ e $\omega_d$ aparecem **diretamente** nas especificações:
        
        $$T_{s_{2\%}} \approx \frac{4}{\sigma}, \qquad T_p = \frac{\pi}{\omega_d}, \qquad f_d = \frac{\omega_d}{2\pi}$$
        
        ### 5.1 Efeito de $\omega_d$ ($\sigma$ fixo)
        
        $\sigma$ fixo → envelope de decaimento $e^{-\sigma t}$ **invariante** → $T_{s_{2\%}}$ não muda.
        Aumentar $\omega_d$ aumenta a frequência de oscilação e reduz $T_p$.
        
        ### 5.2 Efeito de $\sigma$ ($\omega_d$ fixo)
        
        $\omega_d$ fixo → frequência de oscilação e $T_p$ **invariantes**.
        Aumentar $\sigma$ acelera o decaimento → reduz $T_{s_{2\%}}$.
        """)
        
        t_arr5 = np.linspace(0, 12, 700)
        fig5, axes5 = plt.subplots(1, 2, figsize=(9.5, 3.4))
        sigma_f5=1.0
        for ov, col in zip([0.5,1.0,2.0,3.0,4.0,5.0], plt.cm.Blues(np.linspace(0.35,0.95,6))):
            wn2 = sigma_f5**2 + ov**2
            _, y = sc_step(lti([wn2],[1,2*sigma_f5,wn2]), T=t_arr5)
            axes5[0].plot(t_arr5, y, color=col, lw=1.7, label=rf"$\omega_d={ov}$")
        env5 = np.exp(-sigma_f5*t_arr5)
        axes5[0].plot(t_arr5, 1+env5, "k--", lw=0.8, alpha=0.5, label=r"envelope $e^{-\sigma t}$")
        axes5[0].plot(t_arr5, 1-env5, "k--", lw=0.8, alpha=0.5)
        estilo(axes5[0]); axes5[0].set_xlim(0,12)
        axes5[0].set_title(rf"Variação de $\omega_d$ ($\sigma={sigma_f5}$ fixo)", fontsize=8.5)
        axes5[0].legend(ncol=2, fontsize=7)
        
        omega_f5=0.5
        for sv, col in zip([0.5,1.0,2.0,3.0,4.0,5.0], plt.cm.Reds(np.linspace(0.35,0.95,6))):
            wn2 = sv**2 + omega_f5**2
            _, y = sc_step(lti([wn2],[1,2*sv,wn2]), T=t_arr5)
            axes5[1].plot(t_arr5, y, color=col, lw=1.7, label=rf"$\sigma={sv}$")
        axes5[1].axhline(1.0, color=COR["ref"], lw=0.8, ls="--")
        estilo(axes5[1]); axes5[1].set_xlim(0,12)
        axes5[1].set_title(rf"Variação de $\sigma$ ($\omega_d={omega_f5}$ fixo)", fontsize=8.5)
        axes5[1].legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig5, 0.88)
        
        st.markdown("### 🎛️ Explorador — Sliders $\\sigma$ e $\\omega_d$")
        st.caption("Slider **vermelho** = $\\sigma$ (polo move horizontalmente) · Slider **azul** = $\\omega_d$ (polo move verticalmente)")
        
        c5a, c5b = st.columns([1, 2])
        with c5a:
            sig5 = st.slider("Taxa de decaimento $\\sigma$", 0.2, 5.0, 1.0, 0.1, key="sig5")
            od5  = st.slider("Freq. amortecida $\\omega_d$", 0.2, 5.0, 2.0, 0.1, key="od5")
            wn5 = np.sqrt(sig5**2 + od5**2); xi5 = sig5/wn5
            st.info(f"$\\omega_n={wn5:.3f}$ · $\\xi={xi5:.3f}$\n\n"
                    f"$T_{{s_{{2\\%}}}}={4/sig5:.2f}$ s · $T_p={np.pi/od5:.2f}$ s\n\n"
                    f"$f_d={od5/(2*np.pi):.3f}$ Hz")
        
        with c5b:
            t_e5 = np.linspace(0, 14, 700)
            wn5_2 = sig5**2 + od5**2
            _, y_e5 = sc_step(lti([wn5_2],[1,2*sig5,wn5_2]), T=t_e5)
            fig_e5 = make_subplots(rows=1, cols=2, subplot_titles=("Plano s","Resposta ao degrau"))
            plotly_plano_s(fig_e5, 1, 1, xlim=(-8,1), ylim=(-7,7))
            fig_e5.add_trace(go.Scatter(x=[-sig5,-sig5], y=[od5,-od5], mode="markers",
                marker=dict(symbol="x",size=14,color="#d62728",line=dict(width=3)),
                showlegend=False,
                hovertemplate=f"polo=({-sig5:.2f}±{od5:.2f}j)"), row=1, col=1)
            fig_e5.add_trace(go.Scatter(x=t_e5, y=y_e5, mode="lines",
                line=dict(color="#d62728",width=2.5), showlegend=False), row=1, col=2)
            t_env5 = np.linspace(0, 14, 300)
            fig_e5.add_trace(go.Scatter(x=t_env5, y=1+np.exp(-sig5*t_env5), mode="lines",
                line=dict(color="black",width=0.9,dash="dash"), name="envelope", showlegend=False), row=1, col=2)
            fig_e5.add_trace(go.Scatter(x=t_env5, y=1-np.exp(-sig5*t_env5), mode="lines",
                line=dict(color="black",width=0.9,dash="dash"), showlegend=False), row=1, col=2)
            fig_e5.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e5.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e5.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e5.update_layout(height=320, margin=dict(t=30,b=10,l=20,r=10), template="plotly_white")
            st.plotly_chart(fig_e5, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Exemplos Físicos de Sistemas de 2ª Ordem")
        
        st.markdown(r"""
        Sistemas de 2ª ordem surgem quando dois elementos armazenadores de energia de **naturezas diferentes** são acoplados:
        
        | Domínio | Sistema ($u \to y$) | $\omega_n$ | $\xi$ |
        |---|---|---|---|
        | Mecânico translacional | Massa-mola-amortecedor: $F \to x$ | $\sqrt{K/M}$ | $B/(2\sqrt{KM})$ |
        | Mecânico rotacional | Inércia-mola-amortecedor: $\mathcal{T} \to \theta$ | $\sqrt{K_t/J}$ | $B/(2\sqrt{K_t J})$ |
        | Elétrico (RLC série) | $V_e \to V_C$ | $1/\sqrt{LC}$ | $(R/2)\sqrt{C/L}$ |
        | Elétrico (RLC paralelo) | $I \to V$ | $1/\sqrt{LC}$ | $(1/2R)\sqrt{L/C}$ |
        
        > **Analogia:** $M \leftrightarrow L$, $K \leftrightarrow 1/C$, $B \leftrightarrow R$ (mecânico translacional ↔ RLC série).
        """)
        
        t_arr6 = np.linspace(0, 10, 600)
        fig6, axes6 = plt.subplots(1, 2, figsize=(9.5, 3.4))
        M=1.0; K=4.0; wn_mola=np.sqrt(K/M)
        for Bv, col in zip([0.5,2.0,4.0,6.0,8.0], plt.cm.coolwarm(np.linspace(0.1,0.9,5))):
            xi_v=Bv/(2*np.sqrt(K*M))
            _, y=sc_step(lti([wn_mola**2],[1,2*xi_v*wn_mola,wn_mola**2]), T=t_arr6)
            axes6[0].plot(t_arr6, y, color=col, lw=1.7, label=rf"$B={Bv}$ ($\xi={xi_v:.2f}$)")
        axes6[0].axhline(1.0, color=COR["ref"], lw=0.8, ls="--")
        estilo(axes6[0], xlabel="t (s)", ylabel="x(t)/(F/K)")
        axes6[0].set_title(rf"Massa-mola-amortecedor — $M={M}$, $K={K}$", fontsize=8.5)
        axes6[0].legend(fontsize=7); axes6[0].set_xlim(0,10)
        
        L=1.0; C=0.25; wn_rlc=1/np.sqrt(L*C)
        for Rv, col in zip([0.5,1.0,2.0,3.0,4.0], plt.cm.PuOr(np.linspace(0.1,0.9,5))):
            xi_v=Rv/2*np.sqrt(C/L)
            _, y=sc_step(lti([wn_rlc**2],[1,2*xi_v*wn_rlc,wn_rlc**2]), T=t_arr6)
            axes6[1].plot(t_arr6, y, color=col, lw=1.7, label=rf"$R={Rv}\,\Omega$ ($\xi={xi_v:.2f}$)")
        axes6[1].axhline(1.0, color=COR["ref"], lw=0.8, ls="--")
        estilo(axes6[1], xlabel="t (s)", ylabel=r"$V_C(t)/V_e$")
        axes6[1].set_title(rf"Circuito RLC série — $L={L}$ H, $C={C}$ F", fontsize=8.5)
        axes6[1].legend(fontsize=7); axes6[1].set_xlim(0,10)
        plt.tight_layout()
        show_fig(fig6, 0.88)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("7. Polos e Zeros Adicionais")
        
        st.markdown(r"""
        ### 7.1 Polo adicional real em $s = -a$
        
        $$H_3(s) = \frac{k\,\omega_n^2}{(s^2+2\xi\omega_n s+\omega_n^2)(s+a)}$$
        
        O polo adicional acrescenta um modo $e^{-at}$ que **desacelera** a resposta.
        
        - **$a \approx \xi\omega_n$:** polo próximo dos dominantes → grande distorção
        - **$a \gg \xi\omega_n$:** modo desaparece rapidamente → resposta ≈ 2ª ordem puro (*polos dominantes*)
        
        ### 7.2 Zero adicional em $s = -b$
        
        $$H_z(s) = \frac{k\,\omega_n^2\,(s+b)}{s^2+2\xi\omega_n s+\omega_n^2}$$
        
        O zero não altera os polos mas **modifica os resíduos** dos modos.
        
        | Posição do zero | Efeito |
        |---|---|
        | $b \gg \xi\omega_n$ (afastado à esquerda) | Efeito desprezível |
        | $b$ próximo de $\xi\omega_n$ | Aumenta $UP$ e distorce o transitório |
        | $b < 0$ (semiplano direito — fase não-mínima) | *Undershoot* inicial: resposta começa na direção oposta |
        """)
        
        t_arr7 = np.linspace(0, 14, 800)
        xi_v7=0.4; wn_v7=2.0; k_v7=1.0
        _, y0_7 = sc_step(lti([k_v7*wn_v7**2],[1,2*xi_v7*wn_v7,wn_v7**2]), T=t_arr7)
        
        fig7, axes7 = plt.subplots(1, 2, figsize=(9.5, 3.4))
        axes7[0].plot(t_arr7, y0_7, "k--", lw=2.0, label="sem polo adicional")
        for av, col in zip([0.5,1.0,2.0,4.0,8.0], plt.cm.Reds(np.linspace(0.35,0.92,5))):
            _, y=sc_step(lti([k_v7*wn_v7**2], np.polymul([1,2*xi_v7*wn_v7,wn_v7**2],[1,av])), T=t_arr7)
            axes7[0].plot(t_arr7, y, color=col, lw=1.7, label=rf"$a={av}$")
        estilo(axes7[0]); axes7[0].set_xlim(0,14)
        axes7[0].set_title(rf"Polo adicional em $s=-a$ ($\xi={xi_v7}$, $\omega_n={wn_v7}$)", fontsize=8.5)
        axes7[0].legend(fontsize=7); axes7[0].axhline(k_v7, color=COR["ref"], lw=0.8, ls=":")
        
        axes7[1].plot(t_arr7, y0_7, "k--", lw=2.0, label="sem zero adicional")
        for bv, col in zip([-4.0,-1.0,1.0,2.0,4.0,10.0], plt.cm.RdYlGn(np.linspace(0.08,0.92,6))):
            _, y=sc_step(lti([k_v7*wn_v7**2,k_v7*wn_v7**2*bv],[1,2*xi_v7*wn_v7,wn_v7**2]), T=t_arr7)
            axes7[1].plot(t_arr7, y, color=col, lw=1.7, label=rf"$b={bv}$")
        axes7[1].axhline(1.0, color=COR["ref"], lw=0.8, ls=":")
        estilo(axes7[1]); axes7[1].set_xlim(0,14)
        axes7[1].set_title(rf"Zero adicional em $s=-b$ ($\xi={xi_v7}$, $\omega_n={wn_v7}$)", fontsize=8.5)
        axes7[1].legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig7, 0.88)
        
        st.markdown("### 7.3 🎛️ Explorador — Polo e Zero Adicionais")
        st.caption("× preto = polos dominantes (fixos) · × vermelho = polo adicional $-a$ · ○ azul = zero adicional $-b$")
        
        c7a, c7b = st.columns([1, 2])
        with c7a:
            usa_polo7 = st.toggle("🔴 Ativar polo adicional", value=True,  key="uso_polo7")
            usa_zero7 = st.toggle("🔵 Ativar zero adicional", value=False, key="uso_zero7")
            st.markdown("---")
            a7 = st.slider("Polo adicional $a$",  0.3, 10.0,  2.0, 0.1,  key="a7",
                           disabled=not usa_polo7)
            b7 = st.slider("Zero adicional $b$", -6.0, 10.0,  4.0, 0.25, key="b7",
                           disabled=not usa_zero7)
            st.markdown("---")
            # resumo de H(s) ativa
            num_str = f"$k\\omega_n^2" + ("(s+b)" if usa_zero7 else "") + "$"
            den_str = ("$(s^2+2\\xi\\omega_n s+\\omega_n^2)" +
                       ("(s+a)$" if usa_polo7 else "$"))
            st.info(f"**$H(s)$ ativa:** {num_str} / {den_str}\n\n"
                    f"polos dom.: $s={-xi_v7*wn_v7:.2f}\\pm{wn_v7*np.sqrt(1-xi_v7**2):.2f}j$\n\n" +
                    (f"polo adicional: $s={-a7:.2f}$\n\n" if usa_polo7 else "") +
                    (f"zero adicional: $s={-b7:.2f}$" if usa_zero7 else ""))
        
        with c7b:
            t_e7 = np.linspace(0, 14, 700)
        
            # sempre: referência 2ª ordem puro
            _, y_ref7 = sc_step(lti([k_v7*wn_v7**2],
                                     [1, 2*xi_v7*wn_v7, wn_v7**2]), T=t_e7)
        
            # sistema ativo (combina polo e/ou zero conforme toggles)
            den_base7 = [1, 2*xi_v7*wn_v7, wn_v7**2]
            den_ativo7 = np.polymul(den_base7, [1, a7]) if usa_polo7 else den_base7
            if usa_zero7:
                num_ativo7 = [k_v7*wn_v7**2, k_v7*wn_v7**2*b7]
            else:
                num_ativo7 = [k_v7*wn_v7**2]
            _, y_ativo7 = sc_step(lti(num_ativo7, den_ativo7), T=t_e7)
        
            # cor da curva ativa: vermelho se só polo, azul se só zero,
            # roxo se ambos, preto se nenhum
            if usa_polo7 and usa_zero7:
                cor_ativo7 = "#9467bd"; lbl_ativo7 = f"polo a={a7:.1f} + zero b={b7:.1f}"
            elif usa_polo7:
                cor_ativo7 = "#d62728"; lbl_ativo7 = f"polo adicional a={a7:.1f}"
            elif usa_zero7:
                cor_ativo7 = "#1f77b4"; lbl_ativo7 = f"zero adicional b={b7:.1f}"
            else:
                cor_ativo7 = "black";   lbl_ativo7 = "2ª ordem puro (igual à referência)"
        
            wd7    = wn_v7 * np.sqrt(1 - xi_v7**2)
            sigma7 = xi_v7 * wn_v7
        
            fig_e7 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s", "Resposta ao degrau"))
            plotly_plano_s(fig_e7, 1, 1, xlim=(-12, 5), ylim=(-5, 5))
        
            # polos dominantes — sempre visíveis
            fig_e7.add_trace(go.Scatter(
                x=[-sigma7, -sigma7], y=[wd7, -wd7], mode="markers",
                marker=dict(symbol="x", size=14, color="black", line=dict(width=2.5)),
                name="polos dom.", showlegend=True), row=1, col=1)
        
            # polo adicional no plano s — só se ativo
            if usa_polo7:
                fig_e7.add_trace(go.Scatter(
                    x=[-a7], y=[0], mode="markers",
                    marker=dict(symbol="x", size=13, color="#d62728", line=dict(width=2.5)),
                    name=f"polo a={a7:.1f}", showlegend=True), row=1, col=1)
        
            # zero adicional no plano s — só se ativo
            if usa_zero7:
                fig_e7.add_trace(go.Scatter(
                    x=[-b7], y=[0], mode="markers",
                    marker=dict(symbol="circle-open", size=12, color="#1f77b4",
                                line=dict(width=2.5)),
                    name=f"zero b={b7:.1f}", showlegend=True), row=1, col=1)
        
            # curvas de resposta
            fig_e7.add_trace(go.Scatter(
                x=t_e7, y=y_ref7, mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                name="2ª ordem puro (referência)"), row=1, col=2)
        
            fig_e7.add_trace(go.Scatter(
                x=t_e7, y=y_ativo7, mode="lines",
                line=dict(color=cor_ativo7, width=2.5),
                name=lbl_ativo7), row=1, col=2)
        
            fig_e7.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e7.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e7.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e7.update_layout(
                height=340, margin=dict(t=30, b=10, l=20, r=10),
                template="plotly_white", legend=dict(orientation="h", y=1.10))
            st.plotly_chart(fig_e7, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 8
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("8. Sistema Oscilatório — Polos sobre o Eixo Imaginário")
        
        st.markdown(r"""
        ### 8.1 Estrutura do sistema
        
        $$H(s) = \frac{k}{s^2 + k}$$
        
        Comparando com a forma canônica: $\xi = 0$ e $\omega_n = \sqrt{k}$, com polos em $s_{1,2} = \pm j\sqrt{k}$.
        
        > **Não confundir com o duplo integrador** $H(s) = 1/s^2$ (dois polos em $s=0$). Aqui os polos estão sobre o eixo imaginário, não na origem.
        
        Resposta ao degrau unitário:
        
        $$y(t) = 1 - \cos(\sqrt{k}\,t), \quad t \geq 0$$
        
        O sistema é **marginalmente estável** (BIBO instável): sem amortecimento, a energia não é dissipada.
        
        ### 8.2 Efeito do ganho $k$
        
        | Parâmetro | Expressão | Dependência de $k$ |
        |---|---|---|
        | Frequência natural | $\omega_n = \sqrt{k}$ | Aumenta com $\sqrt{k}$ |
        | Período de oscilação | $T_{osc} = 2\pi/\sqrt{k}$ | Diminui com $\sqrt{k}$ |
        | Amplitude de oscilação | $A = 1$ (oscila entre 0 e 2) | **Independente de $k$** |
        | Localização dos polos | $s = \pm j\sqrt{k}$ | Sobem no eixo imaginário |
        """)
        
        t_arr8 = np.linspace(0, 12, 700)
        k_vals8=[1,2,3,4,5,6]; colors8=plt.cm.viridis(np.linspace(0.15,0.90,len(k_vals8)))
        fig8, axes8 = plt.subplots(1, 2, figsize=(9.5, 3.4))
        ax8_p=axes8[0]
        ax8_p.axhline(0,color="k",lw=0.8); ax8_p.axvline(0,color="k",lw=0.8)
        ax8_p.fill_betweenx([-4,4],-1,0,alpha=0.06,color="seagreen")
        ax8_p.fill_betweenx([-4,4], 0,3,alpha=0.06,color="crimson")
        ax8_p.set_xlim(-1,3); ax8_p.set_ylim(-4,4)
        ax8_p.set_xlabel(r"$\sigma$",fontsize=8); ax8_p.set_ylabel(r"$j\omega$",fontsize=8)
        ax8_p.spines[["right","top"]].set_visible(False)
        for kv, col in zip(k_vals8, colors8):
            wn_k=np.sqrt(kv)
            polo_x(ax8_p, 0, wn_k, col); polo_x(ax8_p, 0, -wn_k, col)
            ax8_p.annotate(f"k={kv}", xy=(0.06,wn_k+0.05), fontsize=7, color=col)
        ax8_p.set_title(r"Plano $s$ — polos de $H(s)=k/(s^2+k)$", fontsize=8.5)
        for kv, col in zip(k_vals8, colors8):
            _, y=sc_step(lti([kv],[1,0,kv]), T=t_arr8)
            axes8[1].plot(t_arr8, y, color=col, lw=1.7, label=f"k={kv}")
        axes8[1].axhline(1.0, color=COR["ref"], lw=0.8, ls="--")
        estilo(axes8[1]); axes8[1].set_xlim(0,12)
        axes8[1].set_title(r"Resposta ao degrau — $H(s) = k/(s^2+k)$", fontsize=8.5)
        axes8[1].legend(ncol=2, fontsize=7)
        plt.tight_layout()
        show_fig(fig8, 0.88)
        
        st.markdown("### 8.3 🎛️ Explorador — Slider $k$")
        st.caption("Observe os polos (×) subindo no eixo imaginário e a frequência de oscilação aumentar com $\\sqrt{k}$")
        
        c8a, c8b = st.columns([1, 2])
        with c8a:
            k8 = st.slider("Ganho $k$", 0.5, 10.0, 2.0, 0.25, key="k8")
            wn8 = np.sqrt(k8)
            st.info(f"$\\omega_n={wn8:.3f}$ rad/s · $T_{{osc}}={2*np.pi/wn8:.3f}$ s\n\n"
                    f"polos: $s=\\pm j{wn8:.3f}$\n\namplitude: oscila entre 0 e 2 (sempre)")
        
        with c8b:
            t_e8 = np.linspace(0, 12, 600)
            _, y_e8 = sc_step(lti([k8],[1,0,k8]), T=t_e8)
            fig_e8 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s (polos imaginários)", "Resposta ao degrau"))
            fig_e8.add_vrect(x0=-1.5, x1=0, fillcolor="seagreen",
                             opacity=0.05, layer="below", line_width=0, row=1, col=1)
            fig_e8.add_trace(go.Scatter(x=[0, 0], y=[wn8, -wn8], mode="markers",
                marker=dict(symbol="x",size=14,color="#17becf",line=dict(width=3,color="#17becf")),
                showlegend=False,
                hovertemplate=f"polo=±{wn8:.3f}j<extra>k={k8}</extra>"), row=1, col=1)
            fig_e8.add_trace(go.Scatter(x=t_e8, y=y_e8, mode="lines",
                line=dict(color="#17becf",width=2.5), showlegend=False), row=1, col=2)
            fig_e8.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e8.update_xaxes(title_text="σ", range=[-1.5,1.5],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e8.update_yaxes(title_text="jω", range=[-4,4],
                                zeroline=True, zerolinecolor="black", row=1, col=1)
            fig_e8.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e8.update_yaxes(title_text="y(t)", row=1, col=2)
            fig_e8.update_layout(height=320, margin=dict(t=30,b=10,l=20,r=10), template="plotly_white")
            st.plotly_chart(fig_e8, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 9 — REFERÊNCIAS
        # ═══════════════════════════════════════════════════════════════════════════════
        with st.expander("9. Referências", expanded=False):
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
            "📊 Dinâmica no Domínio do Tempo — Sistemas de Ordem 2 &nbsp;·&nbsp; 🎛️ SINTONIA — Sistemas de Controle<br>"
            "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
            " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
            "</div>",
            unsafe_allow_html=True,
        )
