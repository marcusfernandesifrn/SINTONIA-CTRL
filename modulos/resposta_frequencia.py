"""
📉 Resposta em Frequência de Sistemas em Tempo Contínuo — Parte 1 de 2
Disciplina: Modelagem e Sistemas Lineares
Curso: Engenharia de Energia
Instituição: IFRN — Campus Natal-Central (CNAT)
Autor: Marcus V A Fernandes · marcus.fernandes@ifrn.edu.br · v1.0
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
from scipy.signal import butter, freqs as sp_freqs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
        
def run():
        
        warnings.filterwarnings("ignore")
        
        # ── Configuração da Página ────────────────────────────────────────────────────
        st.set_page_config(
            page_title="Resposta em Frequência — Parte 1",
            page_icon="📉",
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
            pct = f"{int(width_frac * 100)}%"
            st.markdown(
                f'<div class="fig-wrap"><div style="--fw:{pct}">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;display:block;"/>'
                f'</div></div>', unsafe_allow_html=True)
        
        # ── Helpers ───────────────────────────────────────────────────────────────────
        def bode_calc(num, den, w):
            _, H = signal.freqs(num, den, worN=w)
            mag  = 20 * np.log10(np.abs(H) + 1e-15)
            fase = np.degrees(np.unwrap(np.angle(H)))
            return mag, fase
        
        def bode_fig(sistemas, labels, colors=None, titulo="Diagrama de Bode",
                     w_range=(1e-2, 1e3), figsize=(8.5, 5.5)):
            w = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), 2000)
            if colors is None:
                colors = plt.cm.tab10(np.linspace(0, 0.9, len(sistemas)))
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
            for (num, den), lbl, col in zip(sistemas, labels, colors):
                mag, fase = bode_calc(num, den, w)
                axes[0].semilogx(w, mag,  color=col, lw=2.0, label=lbl)
                axes[1].semilogx(w, fase, color=col, lw=2.0, label=lbl)
            axes[0].set_ylabel("Magnitude (dB)", fontsize=8)
            axes[0].legend(fontsize=7); axes[0].set_title(titulo, fontsize=9)
            axes[1].set_ylabel("Fase (°)", fontsize=8)
            axes[1].set_xlabel(r"Frequência $\omega$ (rad/s)", fontsize=8)
            axes[1].legend(fontsize=7)
            for ax in axes:
                ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
                ax.spines[["right","top"]].set_visible(False)
            plt.tight_layout()
            return fig, axes
        
        def plotly_bode(w, mags, fases, labels, titulo, colors=None, show_asint=None):
            """Retorna figura Plotly de Bode (magnitude + fase)."""
            if colors is None:
                palette = ["#1f77b4","#d62728","#2ca02c","#9467bd","#e07000","#17becf"]
                colors = [palette[i % len(palette)] for i in range(len(labels))]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.55, 0.45],
                                subplot_titles=("Magnitude (dB)", "Fase (°)"))
            for i, (mag, fase, lbl, col) in enumerate(zip(mags, fases, labels, colors)):
                fig.add_trace(go.Scatter(x=w, y=mag, mode="lines",
                    line=dict(color=col, width=2.2), name=lbl), row=1, col=1)
                fig.add_trace(go.Scatter(x=w, y=fase, mode="lines",
                    line=dict(color=col, width=2.2), name=lbl, showlegend=False), row=2, col=1)
            if show_asint:
                for asint, col, lbl in show_asint:
                    fig.add_trace(go.Scatter(x=w, y=asint, mode="lines",
                        line=dict(color=col, width=1.4, dash="dash"),
                        name=lbl, showlegend=True), row=1, col=1)
            fig.add_hline(y=0, line_color="black", line_dash="dash", line_width=0.6, row=1, col=1)
            fig.update_xaxes(type="log", title_text="ω (rad/s)")
            fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
            fig.update_yaxes(title_text="Fase (°)", row=2, col=1)
            fig.update_layout(title=dict(text=titulo, font=dict(size=11)),
                              height=490, margin=dict(l=60,r=20,t=60,b=50),
                              template="plotly_white",
                              legend=dict(orientation="h", y=1.08))
            return fig
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CABEÇALHO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.title("📉 Resposta em Frequência de Sistemas")
        st.subheader("Diagramas de Bode")
        st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
        st.markdown("---")
        
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[1. Conceitos de Resposta em Frequência](#1-conceitos-de-resposta-em-frequ-ncia)**
        - 1.1 Definição: entrada senoidal → saída em regime permanente
        - 1.2 Decomposição: componente forçada (persiste) e natural (decai)
        - 1.3 Representação gráfica — Diagrama de Bode (escala logarítmica)
        
        **[2. Como Traçar o Diagrama de Bode](#2-como-tra-ar-o-diagrama-de-bode)**
        - 2.1 Fatores elementares: ganho, polo/zero na origem, polo/zero real, par complexo
        - 2.2 Procedimento de traçado assintótico — exemplo G=10(s+2)/[s(s+5)(s+10)]
        - 2.3 Fatores elementares detalhados (6 tabs): $s$, $1/s$, $s+a$, $1/(s+a)$, $(s+b)/(s+a)$, $1/[(s+a_1)(s+a_2)]$
        - Tabela de referência rápida
        
        **[3. Frequência de Corte e Banda Passante](#3-frequ-ncia-de-corte-e-banda-passante)**
        - 3.1 Frequência de corte $\omega_c$ (−3 dB = metade da potência)
        - 3.2 Banda passante $\omega_{BW}$ vs $\xi$ (sistemas de 2ª ordem)
        - 3.3 Frequência de cruzamento de ganho $\omega_{gc}$
        
        **[4. Margem de Ganho e Margem de Fase](#4-margem-de-ganho-e-margem-de-fase)**
        - 4.1 Margem de fase $\phi_m$ — atraso tolerável até instabilidade
        - 4.2 Margem de ganho $G_m$ — aumento de ganho tolerável
        - 4.3 Leitura no Diagrama de Bode · 🎛️ Explorador interativo
        
        **[5. Diagramas de Bode — Sistemas de 1ª Ordem](#5-diagramas-de-bode-sistemas-de-1-ordem)**
        - $G(s)=k/(s+a)$: assíntotas, frequência de corte, ganho DC
        - 🎛️ Explorador interativo: sliders $k$ e $a$
        
        **[6. Diagramas de Bode — Sistemas de 2ª Ordem](#6-diagramas-de-bode-sistemas-de-2-ordem)**
        - Pico de ressonância $\omega_r$, magnitude de pico $M_r$, banda passante
        - Exemplos 1 e 2
        - 🎛️ Explorador interativo: sliders $k$, $\xi$, $\omega_n$
        
        **[7. Diagrama de Nichols](#7-diagrama-de-nichols)**
        - Ponto crítico $(-180°, 0\text{ dB})$, leitura de margens
        - 🎛️ Explorador: variação de $\xi$
        
        **[8. Filtros em Frequência](#8-filtros-em-frequ-ncia)**
        - Tipos: Passa-baixa, Passa-alta, Passa-faixa, Rejeita-faixa
        - Butterworth: maximalmente plano, $n \times 20$ dB/década
        - 🎛️ Explorador interativo: tipo de filtro e ordem
        
        **[9. Explorador Geral de Bode](#9-explorador-geral-de-bode)**
        - Insira $N(s)/D(s)$ e visualize automaticamente $\omega_c$, $\omega_{BW}$, $\phi_m$, $G_m$
        
        **[10. Referências](#10-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Conceitos de Resposta em Frequência")
        
        st.markdown(r"""
        ### 1.1 Definição
        
        Quando uma **entrada senoidal** $u(t) = A\sin(\omega t)$ é aplicada a um sistema LTI **estável**,
        a saída em **regime permanente** é também uma senoide de **mesma frequência**:
        
        $$y_{rp}(t) = A\,|H(j\omega)|\,\sin\!\bigl(\omega t + \angle H(j\omega)\bigr)$$
        
        onde $H(j\omega) = H(s)\big|_{s=j\omega}$ é a **função de transferência no eixo imaginário**.
        
        | Indicador | Definição | Unidade |
        |---|---|---|
        | **Magnitude** $\|H(j\omega)\|$ | Razão amplitude saída/entrada | adimensional (ou dB) |
        | **Fase** $\angle H(j\omega)$ | Diferença de fase saída − entrada | graus ou radianos |
        
        > Esses indicadores descrevem **exclusivamente** a resposta senoidal em **regime permanente** — não o transitório.
        
        ### 1.2 Decomposição da resposta
        
        | Componente | Origem | Comportamento |
        |---|---|---|
        | **Estado nulo** (forçada) | Entrada senoidal | Persiste em regime permanente |
        | **Entrada nula** (natural) | Condições iniciais / polos do sistema | Decai a zero se $\text{Re}(p_i) < 0$ |
        
        Para sistema estável, a componente natural **decai exponencialmente** e resta apenas $y_{rp}(t)$.
        
        ### 1.3 Diagrama de Bode
        
        $$|H(j\omega)|_{dB} = 20\log_{10}|H(j\omega)|, \qquad \angle H(j\omega) \text{ em graus}$$
        
        O eixo horizontal usa **escala logarítmica** em décadas, linearizando as assíntotas.
        A escala dB comprime faixas dinâmicas de $10^{-6}$ a $10^6$ em representação compacta.
        """)
        
        # Fig 1.1 — resposta senoidal entrada/saída
        t1 = np.linspace(-0.3, 2.5, 600)
        fe1 = 0.4; Me1=1.0; phi_e1=0.0; Ms1=1.35; phi_s1=phi_e1+np.radians(40)
        fig1a, ax1a = plt.subplots(1, 2, figsize=(9.5, 3.2))
        for ax, M, phi, ttl in [(ax1a[0],Me1,phi_e1,"Entrada $f(t)$"),(ax1a[1],Ms1,phi_s1,"Saída $x(t)$")]:
            y = M * np.sin(2*np.pi*fe1*t1 + phi)
            ax.plot(t1, y, "k-", lw=2.2)
            ax.axhline(0, color="k", lw=0.9); ax.axvline(0, color="k", lw=0.9)
            ax.set_xlim(-0.3, 2.5); ax.set_ylim(-1.8, 2.0); ax.axis("off")
            ax.set_title(ttl, fontsize=10)
            t_pk = (np.pi/2 - phi) / (2*np.pi*fe1)
            ax.annotate("", xy=(t_pk, M), xytext=(t_pk, 0),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
        ax1a[0].text(0.75+0.04, Me1/2, "$M_e$", fontsize=10)
        ax1a[1].text(1.05+0.04, Ms1/2, r"$M_s = M_e \cdot |H|$", fontsize=9)
        fig1a.suptitle(r"Resposta senoidal em RP: $M_s = M_e|H(j\omega)|$, $\phi_s = \phi_e + \angle H(j\omega)$", fontsize=9)
        plt.tight_layout()
        show_fig(fig1a, 0.78)
        
        # Fig 1.2 — demonstração numérica G=1/(s+1), ω=2
        w_ex1 = 2.0; t_arr1 = np.linspace(0, 8, 2000)
        sys1_d = signal.lti([1], [1, 1])
        u1_d = np.sin(w_ex1 * t_arr1)
        t_o1, y_o1, _ = signal.lsim(sys1_d, u1_d, t_arr1)
        Hjw1 = 1 / (1j*w_ex1 + 1)
        mag1 = abs(Hjw1); fase1_deg = np.degrees(np.angle(Hjw1))
        y_rp1 = mag1 * np.sin(w_ex1*t_arr1 + np.radians(fase1_deg))
        
        fig1b, ax1b = plt.subplots(figsize=(8.5, 3.0))
        ax1b.plot(t_arr1, u1_d,  color="steelblue", lw=1.8, label=r"Entrada $\sin(2t)$")
        ax1b.plot(t_o1,  y_o1,  color="crimson",   lw=2.0, label="Saída simulada")
        ax1b.plot(t_arr1, y_rp1, color="darkorange",lw=1.4, ls="--",
                  label=f"RP analítico |H|={mag1:.3f}, ∠H={fase1_deg:.1f}°")
        ax1b.axvline(5.5, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax1b.text(5.6, 0.8, "regime\npermanente", fontsize=7.5, color="gray")
        ax1b.set_xlabel("t (s)", fontsize=8); ax1b.set_ylabel("Amplitude", fontsize=8)
        ax1b.legend(fontsize=7); ax1b.spines[["right","top"]].set_visible(False)
        ax1b.set_title(r"$G(s)=1/(s+1)$ — entrada senoidal $\omega=2$ rad/s", fontsize=9)
        plt.tight_layout()
        show_fig(fig1b, 0.72)
        
        st.info(f"$H(j2) = {Hjw1:.4f}$  ·  $|H| = {mag1:.4f}$  ·  $\\angle H = {fase1_deg:.2f}°$")
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Como Traçar o Diagrama de Bode")
        
        st.markdown(r"""
        ### 2.1 Fatores elementares
        
        Todo $G(s)$ decompõe-se em fatores. O Bode resultante é a **soma** das contribuições (dB e graus):
        
        | Fator | DC | Inclinação alta freq. | Fase baixa | Fase alta |
        |---|---|---|---|---|
        | $s$ | $20\log\omega$ | +20 dB/déc | +90° | +90° |
        | $1/s$ | $-20\log\omega$ | −20 dB/déc | −90° | −90° |
        | $s+a$ | $20\log a$ | +20 dB/déc (após $a$) | 0° | +90° |
        | $1/(s+a)$ | $-20\log a$ | −20 dB/déc (após $a$) | 0° | −90° |
        | $(s+b)/(s+a)$ | $20\log(b/a)$ | 0 dB/déc | 0° | 0° |
        | $1/[(s+a_1)(s+a_2)]$ | $-20\log(a_1 a_2)$ | −40 dB/déc (após $a_2$) | 0° | −180° |
        
        **Regra geral:** some os níveis DC de cada fator e acrescente ±20 dB/déc a cada quebra.
        
        ### 2.2 Procedimento de traçado — exemplo
        
        $$G(s) = \frac{10\,(s+2)}{s\,(s+5)\,(s+10)} = 0{,}4 \cdot \frac{(1+s/2)}{(s/1)\,(1+s/5)\,(1+s/10)}$$
        
        | Fator | Tipo | Quebra | Contribuição |
        |---|---|---|---|
        | $0{,}4$ | ganho DC | — | $-7{,}96$ dB |
        | $1/s$ | polo origem | $\omega=0$ | −20 dB/déc partindo de −7,96 dB em $\omega=1$ |
        | $(1+s/2)$ | zero real | $\omega_z=2$ | +20 dB/déc |
        | $1/(1+s/5)$ | polo real | $\omega_{p1}=5$ | −20 dB/déc |
        | $1/(1+s/10)$ | polo real | $\omega_{p2}=10$ | −20 dB/déc |
        """)
        
        w2 = np.logspace(-1, 3, 3000)
        num_ex2 = [10, 20]; den_ex2 = [1, 15, 50, 0]
        mag_ex2, fase_ex2 = bode_calc(num_ex2, den_ex2, w2)
        K_dc2 = 20*np.log10(10*2/(5*10))
        asint2 = (K_dc2 - 20*np.log10(w2)
                  + np.where(w2>=2,  +20*np.log10(w2/2),  0)
                  + np.where(w2>=5,  -20*np.log10(w2/5),  0)
                  + np.where(w2>=10, -20*np.log10(w2/10), 0))
        fase_a2 = (-90 + np.degrees(np.arctan(w2/2))
                      - np.degrees(np.arctan(w2/5))
                      - np.degrees(np.arctan(w2/10)))
        
        fig2a, axes2a = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
        axes2a[0].semilogx(w2, mag_ex2,  "b-",  lw=2.2, label="Exato")
        axes2a[0].semilogx(w2, asint2,   "r--", lw=1.6, label="Assíntota")
        for wk, lbl, dy in [(2,r"$\omega_z=2$",2),(5,r"$\omega_{p1}=5$",2),(10,r"$\omega_{p2}=10$",2)]:
            axes2a[0].axvline(wk, color="gray", ls=":", lw=0.9)
            axes2a[0].text(wk*1.1, -30+dy, lbl, fontsize=7.5, color="gray")
        axes2a[0].axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        axes2a[0].set_ylabel("Magnitude (dB)", fontsize=8); axes2a[0].legend(fontsize=8)
        axes2a[0].set_title(r"$G(s)=10(s+2)/[s(s+5)(s+10)]$ — construção assintótica", fontsize=9)
        axes2a[1].semilogx(w2, fase_ex2, "b-",  lw=2.2, label="Fase exata")
        axes2a[1].semilogx(w2, fase_a2,  "r--", lw=1.6, label="Fase assintótica")
        axes2a[1].set_ylabel("Fase (°)", fontsize=8); axes2a[1].set_xlabel("ω (rad/s)", fontsize=8)
        axes2a[1].legend(fontsize=8)
        for ax in axes2a: ax.spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig2a, 0.78)
        st.info(f"Ganho DC = {K_dc2:.2f} dB · Inclinação final: −20+20−20−20 = **−40 dB/déc**")
        
        # ── 2.3 Fatores elementares em tabs ──────────────────────────────────────────
        st.markdown("### 2.3 Fatores Elementares — análise de cada bloco")
        tabs_fat = st.tabs(["Fator 1: s", "Fator 2: 1/s", "Fator 3: s+a",
                            "Fator 4: 1/(s+a)", "Fator 5: (s+b)/(s+a)",
                            "Fator 6: 1/[(s+a₁)(s+a₂)]"])
        
        w_fat = np.logspace(-2, 3, 2000)
        
        # ── Tab F1: s
        with tabs_fat[0]:
            st.markdown(r"""
        **$G(s) = s$** — zero na origem
        
        $$|G(j\omega)|_{dB} = 20\log_{10}\omega \qquad \angle G(j\omega) = +90°$$
        
        | Propriedade | Valor |
        |---|---|
        | Magnitude | Reta **+20 dB/década** cruzando 0 dB em $\omega=1$ |
        | Fase | Constante **+90°** |
        """)
            mag_f1, fase_f1 = bode_calc([1, 0], [1], w_fat)
            asint_f1 = 20*np.log10(w_fat)
            fig_f1, ax_f1 = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)
            ax_f1[0].semilogx(w_fat, mag_f1,  "b-",  lw=2.5, label="Exato")
            ax_f1[0].semilogx(w_fat, asint_f1,"r--", lw=1.5, label="+20 dB/déc")
            ax_f1[0].axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
            ax_f1[0].axvline(1, color="gray", ls=":", lw=1.0)
            ax_f1[0].text(1.1, -3, "0 dB em ω=1", fontsize=7.5, color="gray")
            ax_f1[0].set_ylabel("Magnitude (dB)", fontsize=8); ax_f1[0].legend(fontsize=7)
            ax_f1[0].set_title(r"$G(s)=s$ — zero na origem", fontsize=9)
            ax_f1[1].semilogx(w_fat, fase_f1, "b-", lw=2.5)
            ax_f1[1].semilogx(w_fat, 90*np.ones_like(w_fat), "r--", lw=1.5, label="+90° constante")
            ax_f1[1].set_ylabel("Fase (°)", fontsize=8); ax_f1[1].set_xlabel("ω (rad/s)", fontsize=8)
            ax_f1[1].legend(fontsize=7); ax_f1[1].set_ylim(0, 180)
            for ax in ax_f1: ax.spines[["right","top"]].set_visible(False)
            plt.tight_layout()
            show_fig(fig_f1, 0.60)
        
        # ── Tab F2: 1/s
        with tabs_fat[1]:
            st.markdown(r"""
        **$G(s) = 1/s$** — polo na origem (integrador)
        
        $$|G(j\omega)|_{dB} = -20\log_{10}\omega \qquad \angle G(j\omega) = -90°$$
        
        | Propriedade | Valor |
        |---|---|
        | Magnitude | Reta **−20 dB/década** cruzando 0 dB em $\omega=1$ |
        | Fase | Constante **−90°** |
        | Efeito | Cada integrador adiciona −90° à fase — reduz margem de fase |
        """)
            mag_f2, fase_f2 = bode_calc([1], [1, 0], w_fat)
            asint_f2 = -20*np.log10(w_fat)
            fig_f2, ax_f2 = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)
            ax_f2[0].semilogx(w_fat, mag_f2,  "b-",  lw=2.5, label="Exato")
            ax_f2[0].semilogx(w_fat, asint_f2,"r--", lw=1.5, label="−20 dB/déc")
            ax_f2[0].axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
            ax_f2[0].axvline(1, color="gray", ls=":", lw=1.0)
            ax_f2[0].text(1.1, 3, "0 dB em ω=1", fontsize=7.5, color="gray")
            ax_f2[0].set_ylabel("Magnitude (dB)", fontsize=8); ax_f2[0].legend(fontsize=7)
            ax_f2[0].set_title(r"$G(s)=1/s$ — polo na origem (integrador)", fontsize=9)
            ax_f2[1].semilogx(w_fat, fase_f2, "b-", lw=2.5)
            ax_f2[1].semilogx(w_fat, -90*np.ones_like(w_fat), "r--", lw=1.5, label="−90° constante")
            ax_f2[1].set_ylabel("Fase (°)", fontsize=8); ax_f2[1].set_xlabel("ω (rad/s)", fontsize=8)
            ax_f2[1].legend(fontsize=7); ax_f2[1].set_ylim(-180, 0)
            for ax in ax_f2: ax.spines[["right","top"]].set_visible(False)
            plt.tight_layout()
            show_fig(fig_f2, 0.60)
        
        # ── Tab F3: s+a
        with tabs_fat[2]:
            st.markdown(r"""
        **$G(s) = s+a$** — zero real simples
        
        $$|G(j\omega)|_{dB} = 20\log_{10}\!\sqrt{\omega^2+a^2} \qquad \angle G(j\omega) = +\arctan\!\left(\tfrac{\omega}{a}\right)$$
        
        | Região | Magnitude assintótica | Fase |
        |---|---|---|
        | $\omega \ll a$ | $20\log_{10}(a)$ dB | $\approx 0°$ |
        | $\omega = a$ | patamar + **3 dB** | $+45°$ |
        | $\omega \gg a$ | $20\log_{10}(a)$ + 20 dB/déc | $\approx +90°$ |
        """)
            a_f3_sl = st.slider("$a$ (freq. de quebra)", 0.1, 20.0, 2.0, 0.1, key="af3")
            mag_f3, fase_f3 = bode_calc([1, a_f3_sl], [1], w_fat)
            mag_dc_f3 = 20*np.log10(a_f3_sl)
            asint_f3 = np.where(w_fat < a_f3_sl, mag_dc_f3,
                                mag_dc_f3 + 20*np.log10(w_fat/a_f3_sl))
            fig_f3 = plotly_bode(w_fat, [mag_f3], [fase_f3],
                                 [rf"$s+{a_f3_sl}$ (exato)"],
                                 rf"$G(s)=s+{a_f3_sl}$ — zero real simples",
                                 show_asint=[(asint_f3, "#d62728", f"assíntota (quebra ω={a_f3_sl})")])
            fig_f3.add_vline(x=a_f3_sl, line_dash="dot", line_color="gray", line_width=1.0)
            st.plotly_chart(fig_f3, use_container_width=True)
        
        # ── Tab F4: 1/(s+a)
        with tabs_fat[3]:
            st.markdown(r"""
        **$G(s) = 1/(s+a)$** — polo real simples *(fator mais comum)*
        
        $$|G(j\omega)|_{dB} = -20\log_{10}\!\sqrt{\omega^2+a^2} \qquad \angle G(j\omega) = -\arctan\!\left(\tfrac{\omega}{a}\right)$$
        
        | Região | Magnitude assintótica | Fase |
        |---|---|---|
        | $\omega \ll a$ | $-20\log_{10}(a)$ dB | $\approx 0°$ |
        | $\omega = a$ | patamar **− 3 dB** | $-45°$ |
        | $\omega \gg a$ | $-20\log_{10}(a)$ − 20 dB/déc | $\approx -90°$ |
        
        > Ganho DC = $1/a$; frequência de corte = $\omega_c = a$. Erro máximo da assíntota: −3 dB.
        """)
            a_f4_sl = st.slider("$a$ (polo)", 0.1, 20.0, 2.0, 0.1, key="af4")
            mag_f4, fase_f4 = bode_calc([1], [1, a_f4_sl], w_fat)
            mag_dc_f4 = -20*np.log10(a_f4_sl)
            asint_f4 = np.where(w_fat < a_f4_sl, mag_dc_f4,
                                mag_dc_f4 - 20*np.log10(w_fat/a_f4_sl))
            fig_f4 = plotly_bode(w_fat, [mag_f4], [fase_f4],
                                 [rf"$1/(s+{a_f4_sl})$ (exato)"],
                                 rf"$G(s)=1/(s+{a_f4_sl})$ — polo real simples",
                                 show_asint=[(asint_f4, "#d62728", f"assíntota (quebra ω={a_f4_sl})")])
            fig_f4.add_vline(x=a_f4_sl, line_dash="dot", line_color="gray", line_width=1.0)
            st.plotly_chart(fig_f4, use_container_width=True)
            st.info(f"Ganho DC = $1/{a_f4_sl} = {1/a_f4_sl:.4f}$ ({mag_dc_f4:.2f} dB) · $\\omega_c = {a_f4_sl}$ rad/s")
        
        # ── Tab F5: (s+b)/(s+a)
        with tabs_fat[4]:
            st.markdown(r"""
        **$G(s) = (s+b)/(s+a)$** — zero e polo reais (compensador lead/lag)
        
        $$|G(j\omega)|_{dB} = 20\log_{10}\!\sqrt{\omega^2+b^2} - 20\log_{10}\!\sqrt{\omega^2+a^2}$$
        
        | Caso | Posição | Comportamento |
        |---|---|---|
        | $b > a$ | Zero à direita do polo | Magnitude sobe entre $a$ e $b$ — **lead** |
        | $b < a$ | Zero à esquerda do polo | Magnitude cai entre $b$ e $a$ — **lag** |
        | $b = a$ | Cancelamento polo-zero | Ganho = 1 (0 dB) |
        
        > $G(0) = b/a$ · $G(\infty) = 1$ (0 dB)
        """)
            c5a, c5b = st.columns(2)
            with c5a:
                b_f5 = st.slider("$b$ (zero)", 0.1, 20.0, 5.0, 0.1, key="bf5")
            with c5b:
                a_f5 = st.slider("$a$ (polo)", 0.1, 20.0, 1.0, 0.1, key="af5")
            mag_f5, fase_f5 = bode_calc([1, b_f5], [1, a_f5], w_fat)
            tipo_f5 = "lead" if b_f5 > a_f5 else ("cancela" if abs(b_f5-a_f5)<0.05 else "lag")
            fig_f5 = plotly_bode(w_fat, [mag_f5], [fase_f5],
                                 [rf"$(s+{b_f5})/(s+{a_f5})$ [{tipo_f5}]"],
                                 rf"$G(s)=(s+{b_f5})/(s+{a_f5})$")
            fig_f5.add_vline(x=a_f5, line_dash="dot", line_color="#d62728", line_width=1.0)
            fig_f5.add_vline(x=b_f5, line_dash="dot", line_color="#2ca02c", line_width=1.0)
            st.plotly_chart(fig_f5, use_container_width=True)
            st.info(f"$G(0) = {b_f5}/{a_f5} = {b_f5/a_f5:.3f}$ ({20*np.log10(b_f5/a_f5):.2f} dB) · $G(\\infty)=1$ (0 dB)")
        
        # ── Tab F6: 1/[(s+a1)(s+a2)]
        with tabs_fat[5]:
            st.markdown(r"""
        **$G(s) = 1/[(s+a_1)(s+a_2)]$** — dois polos reais distintos
        
        | Região | Inclinação |
        |---|---|
        | $\omega \ll a_1$ | 0 dB/déc (patamar em $-20\log_{10}(a_1 a_2)$) |
        | $a_1 < \omega < a_2$ | −20 dB/déc |
        | $\omega \gg a_2$ | −40 dB/déc |
        
        **Fase:** 0° a −180° passando por −90° em $\omega \approx \sqrt{a_1 a_2}$ (média geométrica).
        """)
            c6a, c6b = st.columns(2)
            with c6a:
                a1_f6 = st.slider("$a_1$ (1° polo)", 0.1, 10.0, 2.0, 0.1, key="a1f6")
            with c6b:
                a2_f6 = st.slider("$a_2$ (2° polo)", 0.2, 20.0, 8.0, 0.1, key="a2f6")
            a1_f6 = min(a1_f6, a2_f6); a2_f6 = max(a1_f6+0.1, a2_f6)
            mag_f6, fase_f6 = bode_calc([1], [1, a1_f6+a2_f6, a1_f6*a2_f6], w_fat)
            mag_dc_f6 = -20*np.log10(a1_f6*a2_f6)
            asint_f6 = (mag_dc_f6
                        + np.where(w_fat>=a1_f6, -20*np.log10(w_fat/a1_f6), 0)
                        + np.where(w_fat>=a2_f6, -20*np.log10(w_fat/a2_f6), 0))
            w_med_f6 = np.sqrt(a1_f6*a2_f6)
            fig_f6 = plotly_bode(w_fat, [mag_f6], [fase_f6],
                                 [rf"$1/[(s+{a1_f6})(s+{a2_f6})]$"],
                                 rf"$G(s)=1/[(s+{a1_f6})(s+{a2_f6})]$",
                                 show_asint=[(asint_f6, "#d62728", "Assíntota de Bode")])
            for xv, col in [(a1_f6,"#d62728"),(a2_f6,"#9467bd"),(w_med_f6,"seagreen")]:
                fig_f6.add_vline(x=xv, line_dash="dot", line_color=col, line_width=1.0)
            st.plotly_chart(fig_f6, use_container_width=True)
            st.info(f"$G(0) = {1/(a1_f6*a2_f6):.4f}$ ({mag_dc_f6:.2f} dB) · "
                    f"$\\omega_{{med}} = \\sqrt{{{a1_f6:.1f}\\times{a2_f6:.1f}}} = {w_med_f6:.3f}$ rad/s")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Frequência de Corte e Banda Passante")
        
        st.markdown(r"""
        ### 3.1 Frequência de corte $\omega_c$
        
        $$|G(j\omega_c)|_{dB} = |G(0)|_{dB} - 3\;\text{dB} \qquad\Leftrightarrow\qquad |G(j\omega_c)| = \frac{|G(0)|}{\sqrt{2}}$$
        
        O fator $1/\sqrt{2}$ corresponde a **metade da potência** transmitida ($P \propto |H|^2$).
        Para a 1ª ordem $G(s) = k/(s+a)$: $\omega_c = a$ (polo = frequência de corte).
        
        ### 3.2 Banda passante $\omega_{BW}$
        
        Para a 2ª ordem ($k=1$):
        
        $$\omega_{BW} = \omega_n\sqrt{1-2\xi^2+\sqrt{4\xi^4-4\xi^2+2}}$$
        
        | $\xi$ | $\omega_{BW}/\omega_n$ |
        |---|---|
        | 0,1 | ≈ 1,43 |
        | 0,5 | ≈ 1,27 |
        | 0,707 | ≈ 1,00 (Butterworth) |
        | 1,0 | ≈ 0,64 |
        
        > **Relação com velocidade:** maior $\omega_{BW}$ → sistema mais rápido ($T_r \approx 1{,}8/\omega_{BW}$).
        
        ### 3.3 Frequência de cruzamento de ganho $\omega_{gc}$
        
        $$|G(j\omega_{gc})| = 1 \quad (0\;\text{dB})$$
        
        Fundamental para o cálculo da **margem de fase** (Seção 4).
        """)
        
        xi_vals3 = [0.1, 0.3, 0.5, 0.707, 1.0, 2.0]; wn_v3 = 2.0
        w3 = np.logspace(-1, 2, 3000)
        cols3 = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(xi_vals3)))
        fig3a, axes3a = plt.subplots(1, 2, figsize=(9.5, 3.4))
        bws3 = []
        for xi, col in zip(xi_vals3, cols3):
            xi_u = max(xi, 1e-5)
            _, H3 = signal.freqs([wn_v3**2], [1, 2*xi_u*wn_v3, wn_v3**2], worN=w3)
            mag3 = 20*np.log10(np.abs(H3)); dc3 = mag3[0]
            idx_bw = np.where(mag3 >= dc3-3)[0]
            wbw3 = w3[idx_bw[-1]] if len(idx_bw) else wn_v3
            bws3.append(wbw3)
            axes3a[0].semilogx(w3, mag3, color=col, lw=1.8, label=f"ξ={xi}")
            axes3a[0].axvline(wbw3, color=col, ls=":", lw=1.0, alpha=0.7)
        axes3a[0].axhline(-3, color="k", ls="--", lw=0.9, alpha=0.6, label="−3 dB")
        axes3a[0].set_xlabel("ω (rad/s)",fontsize=8); axes3a[0].set_ylabel("Magnitude (dB)",fontsize=8)
        axes3a[0].set_title(rf"Banda passante — $\omega_n={wn_v3}$", fontsize=8.5)
        axes3a[0].legend(fontsize=7, loc="lower left"); axes3a[0].spines[["right","top"]].set_visible(False)
        xi_c3 = np.linspace(0.05, 2.0, 300)
        bw_c3 = wn_v3*np.sqrt(1-2*xi_c3**2+np.sqrt(4*xi_c3**4-4*xi_c3**2+2))
        axes3a[1].plot(xi_c3, bw_c3/wn_v3, "b-", lw=2.2, label=r"$\omega_{BW}/\omega_n$")
        for xi, wbw, col in zip(xi_vals3, bws3, cols3):
            axes3a[1].plot(xi, wbw/wn_v3, "o", color=col, ms=8, zorder=5)
        axes3a[1].axhline(1.0, color="gray", ls="--", lw=0.8)
        axes3a[1].axvline(1/np.sqrt(2), color="gray", ls=":", lw=0.8)
        axes3a[1].text(1/np.sqrt(2)+0.03, 0.5, r"$\xi=1/\sqrt{2}$", fontsize=8, color="gray")
        axes3a[1].set_xlabel(r"$\xi$",fontsize=8); axes3a[1].set_ylabel(r"$\omega_{BW}/\omega_n$",fontsize=8)
        axes3a[1].set_title(r"Banda passante normalizada vs $\xi$", fontsize=8.5)
        axes3a[1].legend(fontsize=8); axes3a[1].spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig3a, 0.88)
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Margem de Ganho e Margem de Fase")
        
        st.markdown(r"""
        ### 4.1 Margem de fase $\phi_m$
        
        Na frequência de cruzamento de ganho $\omega_{gc}$ (onde $|G(j\omega_{gc})| = 0$ dB):
        
        $$\phi_m = 180° + \angle G(j\omega_{gc})$$
        
        | $\phi_m$ | Interpretação |
        |---|---|
        | $> 0°$ | Sistema estável em MF |
        | $= 0°$ | Limite (oscilatório) |
        | $< 0°$ | Sistema instável em MF |
        
        **Regra prática:** $30° \leq \phi_m \leq 60°$ para boa resposta transitória.
        
        ### 4.2 Margem de ganho $G_m$
        
        Na frequência de cruzamento de fase $\omega_{pc}$ (onde $\angle G(j\omega_{pc}) = -180°$):
        
        $$G_m = -|G(j\omega_{pc})|_{dB} = 20\log_{10}\!\left(\frac{1}{|G(j\omega_{pc})|}\right)$$
        
        | $G_m$ | Interpretação |
        |---|---|
        | $> 0$ dB | Estável — ganho pode crescer $G_m$ dB |
        | $= 0$ dB | Limite |
        | $< 0$ dB | Instável |
        
        **Regra prática:** $G_m \geq 6$ dB para estabilidade robusta.
        """)
        
        # Fig estática: G=10/[s(s+1)(s+2)]
        w4 = np.logspace(-1, 2, 4000)
        num_mg4=[10]; den_mg4=[1,3,2,0]
        mag_mg4, fase_mg4 = bode_calc(num_mg4, den_mg4, w4)
        idx_gc4 = np.argmin(np.abs(mag_mg4)); w_gc4 = w4[idx_gc4]
        pm4 = 180 + fase_mg4[idx_gc4]
        idx_pc4 = np.argmin(np.abs(fase_mg4+180)); w_pc4 = w4[idx_pc4]
        gm4_db = -mag_mg4[idx_pc4]
        
        fig4a, axes4a = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
        axes4a[0].semilogx(w4, mag_mg4, "b-", lw=2.2)
        axes4a[0].axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        axes4a[0].axvline(w_gc4, color="#2ca02c", lw=1.2, ls="--", alpha=0.8)
        axes4a[0].axvline(w_pc4, color="#d62728", lw=1.2, ls="--", alpha=0.8)
        axes4a[0].annotate("", xy=(w_pc4,0), xytext=(w_pc4, mag_mg4[idx_pc4]),
            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.8))
        axes4a[0].text(w_pc4*1.15, mag_mg4[idx_pc4]/2, f"$G_m={gm4_db:.1f}$ dB",
                       color="#d62728", fontsize=9)
        axes4a[0].plot(w_gc4, 0, "o", color="#2ca02c", ms=9, zorder=5)
        axes4a[0].plot(w_pc4, mag_mg4[idx_pc4], "s", color="#d62728", ms=9, zorder=5)
        axes4a[0].set_ylabel("Magnitude (dB)", fontsize=8)
        axes4a[0].set_title(r"$G(s)=10/[s(s+1)(s+2)]$ — Margens de estabilidade", fontsize=9)
        axes4a[0].spines[["right","top"]].set_visible(False)
        axes4a[1].semilogx(w4, fase_mg4, "b-", lw=2.2)
        axes4a[1].axhline(-180, color="k", lw=0.8, ls="--", alpha=0.4)
        axes4a[1].axvline(w_gc4, color="#2ca02c", lw=1.2, ls="--", alpha=0.8)
        axes4a[1].axvline(w_pc4, color="#d62728", lw=1.2, ls="--", alpha=0.8)
        axes4a[1].annotate("", xy=(w_gc4, fase_mg4[idx_gc4]), xytext=(w_gc4, -180),
            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.8))
        axes4a[1].text(w_gc4*1.12, (fase_mg4[idx_gc4]-180)/2-180,
                       rf"$\phi_m={pm4:.1f}°$", color="#2ca02c", fontsize=9)
        axes4a[1].plot(w_gc4, fase_mg4[idx_gc4], "o", color="#2ca02c", ms=9, zorder=5)
        axes4a[1].plot(w_pc4, -180, "s", color="#d62728", ms=9, zorder=5)
        axes4a[1].set_ylabel("Fase (°)", fontsize=8); axes4a[1].set_xlabel("ω (rad/s)", fontsize=8)
        axes4a[1].spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig4a, 0.78)
        est4 = "🟢 ESTÁVEL" if pm4>0 and gm4_db>0 else "🔴 INSTÁVEL/MARGINAL"
        st.info(f"$\\phi_m = {pm4:.2f}°$ em $\\omega_{{gc}}={w_gc4:.3f}$ rad/s · "
                f"$G_m = {gm4_db:.2f}$ dB em $\\omega_{{pc}}={w_pc4:.3f}$ rad/s · **{est4}**")
        
        # ── Explorador seção 4 ────────────────────────────────────────────────────────
        st.markdown("### 🎛️ Explorador — Margens de Estabilidade")
        st.caption("Insira num/den da malha aberta $G(s)$ e veja as margens")
        
        c4a, c4b = st.columns([1, 2])
        with c4a:
            num4_str = st.text_input("Numerador $N(s)$", value="10", key="num4mg")
            den4_str = st.text_input("Denominador $D(s)$", value="1, 3, 2, 0", key="den4mg")
            w4_min = st.slider("$\\log_{10}(\\omega_{min})$", -3.0, 0.0, -1.0, 0.5, key="wmin4")
            w4_max = st.slider("$\\log_{10}(\\omega_{max})$",  1.0, 5.0,  3.0, 0.5, key="wmax4")
        
        with c4b:
            try:
                num4_e = [float(x.strip()) for x in num4_str.split(",") if x.strip()]
                den4_e = [float(x.strip()) for x in den4_str.split(",") if x.strip()]
                w4_e = np.logspace(w4_min, w4_max, 4000)
                mag4_e, fase4_e = bode_calc(num4_e, den4_e, w4_e)
                idx_gc4e = np.argmin(np.abs(mag4_e))
                w_gc4e = w4_e[idx_gc4e]; pm4e = 180 + fase4_e[idx_gc4e]
                idx_pc4e = np.argmin(np.abs(fase4_e+180))
                w_pc4e = w4_e[idx_pc4e]; gm4e_db = -mag4_e[idx_pc4e]
        
                fig_e4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.55,0.45],
                                       subplot_titles=("Magnitude (dB)","Fase (°)"))
                fig_e4.add_trace(go.Scatter(x=w4_e, y=mag4_e, mode="lines",
                    line=dict(color="#1f77b4",width=2.2), name="Magnitude"), row=1, col=1)
                fig_e4.add_trace(go.Scatter(x=w4_e, y=fase4_e, mode="lines",
                    line=dict(color="#1f77b4",width=2.2), name="Fase", showlegend=False), row=2, col=1)
                fig_e4.add_hline(y=0,   line_color="black",   line_dash="dash", line_width=0.7, row=1, col=1)
                fig_e4.add_hline(y=-180,line_color="black",   line_dash="dash", line_width=0.7, row=2, col=1)
                fig_e4.add_vline(x=w_gc4e, line_color="#2ca02c", line_dash="dash", line_width=1.3)
                fig_e4.add_vline(x=w_pc4e, line_color="#d62728", line_dash="dash", line_width=1.3)
                fig_e4.add_trace(go.Scatter(x=[w_gc4e], y=[0], mode="markers",
                    marker=dict(symbol="circle",size=11,color="#2ca02c"),
                    name=f"ωgc={w_gc4e:.3f} φm={pm4e:.1f}°"), row=1, col=1)
                fig_e4.add_trace(go.Scatter(x=[w_pc4e], y=[mag4_e[idx_pc4e]], mode="markers",
                    marker=dict(symbol="square",size=11,color="#d62728"),
                    name=f"ωpc={w_pc4e:.3f} Gm={gm4e_db:.1f} dB"), row=1, col=1)
                est4e = "ESTÁVEL" if pm4e>0 and gm4e_db>0 else "INSTÁVEL/MARGINAL"
                fig_e4.update_xaxes(type="log", title_text="ω (rad/s)")
                fig_e4.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
                fig_e4.update_yaxes(title_text="Fase (°)", row=2, col=1)
                fig_e4.update_layout(
                    title=f"Bode — φm={pm4e:.1f}°  Gm={gm4e_db:.1f} dB  [{est4e}]",
                    height=460, margin=dict(t=50,b=20,l=60,r=20),
                    template="plotly_white",
                    legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig_e4, use_container_width=True)
        
                est_icon = "🟢" if est4e=="ESTÁVEL" else "🔴"
                st.info(f"**{est_icon} {est4e}**\n\n"
                        f"$\\phi_m = {pm4e:.2f}°$ em $\\omega_{{gc}}={w_gc4e:.4f}$ rad/s\n\n"
                        f"$G_m = {gm4e_db:.2f}$ dB em $\\omega_{{pc}}={w_pc4e:.4f}$ rad/s")
            except Exception as ex:
                st.error(f"Erro: {ex}")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Diagramas de Bode — Sistemas de 1ª Ordem")
        
        st.markdown(r"""
        $$G(s) = \frac{k}{s + a} = \frac{k/a}{1 + s/a}$$
        
        onde $k/a$ é o **ganho DC** e $1/a$ é a **constante de tempo** $\tau$.
        
        Avaliada em $s = j\omega$:
        
        $$|G(j\omega)| = \frac{k}{\sqrt{\omega^2 + a^2}}, \qquad \angle G(j\omega) = -\arctan\!\left(\frac{\omega}{a}\right)$$
        
        | Região | Magnitude assintótica | Inclinação |
        |---|---|---|
        | $\omega \ll a$ | $20\log_{10}(k/a)$ dB | **0 dB/década** |
        | $\omega \gg a$ | decai a partir de $\omega=a$ | **−20 dB/década** |
        | $\omega = a$ | assíntota **− 3 dB** | ponto de junção |
        
        **Fase:** de $0°$ a $-90°$, valendo $-45°$ em $\omega = a$. Transição em $[a/10,\; 10a]$.
        """)
        
        w5 = np.logspace(-2, 3, 2000)
        
        # Variação de k
        fig5a, ax5a = bode_fig(
            [([k],[1,1.0]) for k in [0.5,1,2,4]],
            [f"k={k}" for k in [0.5,1,2,4]],
            titulo=r"Bode — $G(s)=k/(s+1)$ — variação de $k$")
        ax5a[0].axvline(1.0, color="k", ls="--", lw=0.9, alpha=0.5)
        ax5a[0].text(1.1, ax5a[0].get_ylim()[0]+2, "ωc=1 rad/s", fontsize=7.5, color="k", alpha=0.7)
        show_fig(fig5a, 0.75)
        
        # Variação de a
        fig5b, ax5b = bode_fig(
            [([1.0],[1,a]) for a in [0.5,1,2,4]],
            [f"a={a}" for a in [0.5,1,2,4]],
            colors=plt.cm.tab10(np.linspace(0,0.8,4)),
            titulo=r"Bode — $G(s)=1/(s+a)$ — variação de $a$")
        for i, a in enumerate([0.5,1,2,4]):
            ax5b[0].axvline(a, color=plt.cm.tab10(i/4), ls=":", lw=1.0, alpha=0.6)
        show_fig(fig5b, 0.75)
        
        # Exemplo G=2/(s+3) com assíntota
        k_ex5=2.0; a_ex5=3.0
        mag_ex5, fase_ex5 = bode_calc([k_ex5],[1,a_ex5], w5)
        mag_bf5=20*np.log10(k_ex5/a_ex5)
        asint5=np.where(w5<a_ex5, mag_bf5, mag_bf5-20*np.log10(w5/a_ex5))
        fig5c, ax5c = plt.subplots(2,1,figsize=(8.5,5.5),sharex=True)
        ax5c[0].semilogx(w5, mag_ex5, "b-",  lw=2.0, label=r"$G(j\omega)$ exato")
        ax5c[0].semilogx(w5, asint5,  "r--", lw=1.5, label="Assíntota")
        ax5c[0].axvline(a_ex5, color="gray", ls=":", lw=1)
        ax5c[0].annotate(f"ωc={a_ex5} rad/s\n(erro=3 dB)",
            xy=(a_ex5, mag_bf5-3), xytext=(a_ex5*3, mag_bf5-8),
            fontsize=8, color="gray", arrowprops=dict(arrowstyle="->",color="gray",lw=0.8))
        ax5c[0].set_ylabel("Magnitude (dB)",fontsize=8); ax5c[0].legend(fontsize=8)
        ax5c[0].set_title(r"$G(s)=2/(s+3)$ — Bode com assíntota",fontsize=9)
        ax5c[1].semilogx(w5, fase_ex5, "b-", lw=2.0, label="Fase exata")
        for xv in [a_ex5/10, a_ex5, a_ex5*10]:
            ax5c[1].axvline(xv, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax5c[1].set_ylabel("Fase (°)",fontsize=8); ax5c[1].set_xlabel("ω (rad/s)",fontsize=8)
        ax5c[1].legend(fontsize=8)
        for ax in ax5c: ax.spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig5c, 0.72)
        
        # Explorador 1ª ordem
        st.markdown("### 🎛️ Explorador — Bode 1ª Ordem")
        st.caption("Sliders **azul** = $k$ · **vermelho** = $a$")
        c5a_, c5b_ = st.columns([1, 2])
        with c5a_:
            k5_sl = st.slider("Ganho $k$", 0.1, 10.0, 1.0, 0.1, key="k5sl")
            a5_sl = st.slider("Polo $a$",  0.1, 10.0, 1.0, 0.1, key="a5sl")
            mag_dc5_e = 20*np.log10(k5_sl/a5_sl)
            st.info(f"Ganho DC = ${k5_sl}/{a5_sl} = {k5_sl/a5_sl:.3f}$ ({mag_dc5_e:.2f} dB)\n\n"
                    f"$\\omega_c = {a5_sl}$ rad/s · fase em $\\omega_c$ = −45°")
        with c5b_:
            mag5_e, fase5_e = bode_calc([k5_sl], [1, a5_sl], w5)
            asint5_e = np.where(w5<=a5_sl, mag_dc5_e, mag_dc5_e-20*np.log10(w5/a5_sl))
            fig_e5_ = plotly_bode(w5, [mag5_e], [fase5_e],
                                  [rf"$G(s)={k5_sl:.1f}/(s+{a5_sl:.1f})$"],
                                  rf"Bode 1ª ordem — $k={k5_sl:.1f}$, $a={a5_sl:.1f}$",
                                  show_asint=[(asint5_e,"#d62728","assíntota")])
            fig_e5_.add_vline(x=a5_sl, line_dash="dot", line_color="gray")
            fig_e5_.add_trace(go.Scatter(x=[a5_sl], y=[mag_dc5_e-3], mode="markers+text",
                marker=dict(size=10,color="#2ca02c"),
                text=[f"ωc={a5_sl:.1f}  {mag_dc5_e-3:.1f} dB"],
                textposition="top right", showlegend=False), row=1, col=1)
            st.plotly_chart(fig_e5_, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Diagramas de Bode — Sistemas de 2ª Ordem")
        
        st.markdown(r"""
        $$G(s) = \frac{k\,\omega_n^2}{s^2 + 2\xi\omega_n s + \omega_n^2}$$
        
        ### Pico de ressonância (para $\xi < 1/\sqrt{2}$)
        
        $$\omega_r = \omega_n\sqrt{1 - 2\xi^2} \qquad M_r = \frac{k}{2\xi\sqrt{1-\xi^2}} \qquad |G(j\omega_r)|_{dB} = 20\log_{10}(M_r)$$
        
        > Para $\xi \geq 1/\sqrt{2} \approx 0{,}707$: sem pico — magnitude decresce monotonicamente.
        
        ### Assíntotas
        
        | Região | Magnitude | Inclinação |
        |---|---|---|
        | $\omega \ll \omega_n$ | $20\log_{10}(k)$ dB | **0 dB/década** |
        | $\omega \gg \omega_n$ | decai a partir de $\omega_n$ | **−40 dB/década** |
        
        A inclinação de −40 dB/dec reflete os **dois polos** do sistema.
        """)
        
        w6 = np.logspace(-2, 3, 2000)
        xi6_v=0.7; wn6_v=2.0
        
        # Variação de k
        fig6a, ax6a = bode_fig(
            [([k*wn6_v**2],[1,2*xi6_v*wn6_v,wn6_v**2]) for k in [0.5,1,2,4]],
            [f"k={k}" for k in [0.5,1,2,4]],
            titulo=rf"Bode — $\xi={xi6_v}$, $\omega_n={wn6_v}$ — variação de $k$")
        ax6a[0].axvline(wn6_v,color="k",ls="--",lw=0.8,alpha=0.5)
        show_fig(fig6a, 0.75)
        
        # Variação de xi
        xis6=[0.1,0.3,0.5,0.7,1.0,2.0]
        cols6=plt.cm.RdYlGn(np.linspace(0.1,0.9,len(xis6)))
        fig6b, ax6b = bode_fig(
            [([wn6_v**2],[1,2*max(xi,1e-5)*wn6_v,wn6_v**2]) for xi in xis6],
            [f"ξ={xi}" for xi in xis6], cols6,
            titulo=rf"Bode — $k=1$, $\omega_n={wn6_v}$ — variação de $\xi$")
        ax6b[0].axvline(wn6_v,color="k",ls="--",lw=0.8,alpha=0.5)
        show_fig(fig6b, 0.75)
        
        # Exemplo 1: G=50/(s²+10.25s+25)
        num_e16=[50]; den_e16=[1,10.25,25]
        wn_e16=np.sqrt(den_e16[2]); xi_e16=den_e16[1]/(2*wn_e16); k_e16=num_e16[0]/den_e16[2]
        fig6c, ax6c = bode_fig([([num_e16[0]],den_e16)],["$G(s)=50/(s^2+10.25s+25)$"],
            colors=["#1f77b4"],
            titulo=r"$G(s)=50/(s^2+10.25s+25)$ — Bode")
        ax6c[0].axvline(wn_e16,color="gray",ls="--",lw=0.9)
        ax6c[0].text(wn_e16*1.1, ax6c[0].get_ylim()[0]+1, f"ωn={wn_e16:.2f}", fontsize=8, color="gray")
        show_fig(fig6c, 0.72)
        st.info(f"$\\omega_n={wn_e16:.3f}$ · $\\xi={xi_e16:.3f}$ · $k={k_e16:.3f}$ — sem pico ($\\xi > 1/\\sqrt{{2}}$)")
        
        # Exemplo 2: G=50/(s²+5s+25)
        num_e26=[50]; den_e26=[1,5,25]
        wn_e26=np.sqrt(den_e26[2]); xi_e26=den_e26[1]/(2*wn_e26); k_e26=num_e26[0]/den_e26[2]
        fig6d, ax6d = bode_fig([([num_e26[0]],den_e26)],["$G(s)=50/(s^2+5s+25)$"],
            colors=["#d62728"],
            titulo=r"$G(s)=50/(s^2+5s+25)$ — Bode (pico de ressonância)")
        ax6d[0].axvline(wn_e26,color="gray",ls="--",lw=0.9)
        if xi_e26 < 1/np.sqrt(2):
            wr6=wn_e26*np.sqrt(1-2*xi_e26**2)
            Mr6=k_e26/(2*xi_e26*np.sqrt(1-xi_e26**2))
            ax6d[0].axvline(wr6,color="orange",ls=":",lw=1.2)
            ax6d[0].text(wr6*1.05,20*np.log10(Mr6)-2,
                         f"ωr={wr6:.2f}\n{20*np.log10(Mr6):.1f} dB",fontsize=7.5,color="darkorange")
        show_fig(fig6d, 0.72)
        if xi_e26 < 1/np.sqrt(2):
            st.info(f"$\\omega_n={wn_e26:.3f}$ · $\\xi={xi_e26:.3f}$ · $k={k_e26:.3f}$ · "
                    f"**pico**: $\\omega_r={wr6:.3f}$ rad/s → $M_r={20*np.log10(Mr6):.2f}$ dB")
        
        # Explorador 2ª ordem
        st.markdown("### 🎛️ Explorador — Bode 2ª Ordem")
        st.caption("Slider **azul** = $k$ · **vermelho** = $\\xi$ · **verde** = $\\omega_n$")
        c6a_, c6b_ = st.columns([1, 2])
        with c6a_:
            k6_sl  = st.slider("Ganho $k$", 0.1, 5.0, 1.0, 0.1, key="k6sl")
            xi6_sl = st.slider("Amortecimento $\\xi$", 0.05, 2.0, 0.7, 0.05, key="xi6sl")
            wn6_sl = st.slider("Freq. natural $\\omega_n$", 0.2, 10.0, 2.0, 0.2, key="wn6sl")
            pico_txt6=""
            if xi6_sl < 1/np.sqrt(2):
                wr6_e=wn6_sl*np.sqrt(1-2*xi6_sl**2)
                Mr6_e=k6_sl/(2*xi6_sl*np.sqrt(1-xi6_sl**2))
                pico_txt6=f"\n\nPico: $\\omega_r={wr6_e:.3f}$ rad/s → $M_r={20*np.log10(Mr6_e):.2f}$ dB"
            st.info(f"$G(0)={k6_sl:.2f}$ ({20*np.log10(k6_sl):.1f} dB)\n\n"
                    f"$\\omega_n={wn6_sl:.2f}$ · $\\xi={xi6_sl:.3f}$" + pico_txt6)
        with c6b_:
            xi6_u = max(xi6_sl, 1e-5)
            mag6_e, fase6_e = bode_calc([k6_sl*wn6_sl**2],[1,2*xi6_u*wn6_sl,wn6_sl**2], w6)
            asint6_e = np.where(w6<=wn6_sl, 20*np.log10(k6_sl),
                                20*np.log10(k6_sl)-40*np.log10(w6/wn6_sl))
            titulo6_e = f"Bode 2ª ordem — k={k6_sl:.1f}  ξ={xi6_sl:.2f}  ωn={wn6_sl:.1f}"
            fig_e6_ = plotly_bode(w6, [mag6_e], [fase6_e],
                                  [rf"$G(j\omega)$ exato"],
                                  titulo6_e,
                                  show_asint=[(asint6_e,"#d62728","assíntota")])
            fig_e6_.add_vline(x=wn6_sl, line_dash="dot", line_color="gray")
            if xi6_sl < 1/np.sqrt(2):
                fig_e6_.add_trace(go.Scatter(x=[wr6_e], y=[20*np.log10(Mr6_e)],
                    mode="markers+text",
                    marker=dict(size=12,color="#ff7f0e",symbol="star"),
                    text=[f"ωr={wr6_e:.2f}  {20*np.log10(Mr6_e):.1f} dB"],
                    textposition="top right", name="Pico"), row=1, col=1)
            st.plotly_chart(fig_e6_, use_container_width=True)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7 — NICHOLS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("7. Diagrama de Nichols")
        
        st.markdown(r"""
        ### 7.1 Definição
        
        O **Diagrama de Nichols** traça a curva paramétrica:
        
        $$\Bigl(\angle G(j\omega)\;[\text{eixo }x],\quad |G(j\omega)|_{dB}\;[\text{eixo }y]\Bigr), \quad \omega \in (0, +\infty)$$
        
        Condensa magnitude e fase num único plano, ao contrário do Bode (dois gráficos).
        
        ### 7.2 Margens de estabilidade — ponto crítico $(-180°,\; 0\text{ dB})$
        
        | Medida | Onde ler no Nichols |
        |---|---|
        | **Margem de fase** $\phi_m$ | Distância em fase até $-180°$ quando a curva cruza $0$ dB |
        | **Margem de ganho** $G_m$ | Distância em ganho até $0$ dB quando a curva cruza $-180°$ |
        """)
        
        w7 = np.logspace(-1, 2, 3000)
        num_n7=[50]; den_n7=[1,5,25]
        _, H7 = signal.freqs(num_n7, den_n7, worN=w7)
        mag_db7=20*np.log10(np.abs(H7)); fase_deg7=np.degrees(np.angle(H7))
        
        fig7a, ax7a = plt.subplots(figsize=(7.0, 5.0))
        sc7=ax7a.scatter(fase_deg7, mag_db7, c=np.log10(w7), cmap="plasma", s=8, zorder=3)
        ax7a.plot(fase_deg7, mag_db7, color="royalblue", lw=1.5, alpha=0.6)
        plt.colorbar(sc7, ax=ax7a, label="log₁₀(ω)")
        for w_mark7 in [0.5,1,2,5,10,20]:
            idx7=np.argmin(np.abs(w7-w_mark7))
            ax7a.annotate(f"ω={w_mark7}", xy=(fase_deg7[idx7],mag_db7[idx7]),
                          xytext=(fase_deg7[idx7]-5,mag_db7[idx7]+1.5), fontsize=7.5,color="#333")
        ax7a.axhline(0,    color="k", lw=0.8, ls="--", alpha=0.5, label="0 dB")
        ax7a.axvline(-180, color="r", lw=0.8, ls="--", alpha=0.5, label="−180°")
        ax7a.plot(-180, 0, "r*", ms=14, zorder=5, label="Ponto crítico")
        ax7a.set_xlabel("Fase (°)",fontsize=9); ax7a.set_ylabel("Magnitude (dB)",fontsize=9)
        ax7a.set_title(r"Nichols — $G(s)=50/(s^2+5s+25)$",fontsize=9)
        ax7a.legend(fontsize=8); ax7a.spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig7a, 0.55)
        
        # Explorador Nichols com xi
        st.markdown("### 🎛️ Explorador — Nichols: variação de $\\xi$")
        st.caption("Selecione $\\xi$ e $\\omega_n$ — observe como a curva se afasta ou aproxima do ponto crítico")
        
        c7a_, c7b_ = st.columns([1, 2])
        with c7a_:
            wn7_sl = st.slider("$\\omega_n$", 0.5, 10.0, 5.0, 0.5, key="wn7")
            k7_sl  = st.slider("$k$", 0.5, 5.0, 2.0, 0.5, key="k7")
            xis7_sel = st.multiselect(
                "Valores de $\\xi$", [0.1,0.2,0.3,0.5,0.7,1.0,2.0],
                default=[0.1,0.3,0.5,0.7], key="xis7")
        
        with c7b_:
            if xis7_sel:
                palette7=["#d62728","#ff7f0e","#2ca02c","#1f77b4","#9467bd","#8c564b","#17becf"]
                fig_n7 = go.Figure()
                for i, xi7 in enumerate(xis7_sel):
                    xi7u = max(xi7, 1e-5)
                    _, H7e = signal.freqs([k7_sl*wn7_sl**2],[1,2*xi7u*wn7_sl,wn7_sl**2],worN=w7)
                    fig_n7.add_trace(go.Scatter(
                        x=np.degrees(np.angle(H7e)), y=20*np.log10(np.abs(H7e)),
                        mode="lines", line=dict(color=palette7[i%len(palette7)],width=2.2),
                        name=f"ξ={xi7}",
                        hovertemplate="fase=%{x:.1f}°<br>mag=%{y:.2f} dB<extra>ξ="+str(xi7)+"</extra>"))
                fig_n7.add_hline(y=0,   line_color="black", line_dash="dash", line_width=0.8)
                fig_n7.add_vline(x=-180,line_color="red",   line_dash="dash", line_width=0.8)
                fig_n7.add_trace(go.Scatter(x=[-180],y=[0],mode="markers",
                    marker=dict(symbol="star",size=14,color="red"),name="Ponto crítico"))
                fig_n7.update_layout(
                    title=rf"Nichols — $k={k7_sl}$, $\omega_n={wn7_sl}$",
                    xaxis=dict(title="Fase (°)"), yaxis=dict(title="Magnitude (dB)"),
                    height=420, margin=dict(l=60,r=20,t=60,b=50),
                    template="plotly_white",
                    legend=dict(orientation="h",y=1.08))
                st.plotly_chart(fig_n7, use_container_width=True)
            else:
                st.info("Selecione ao menos um valor de ξ.")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 8 — FILTROS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("8. Filtros em Frequência")
        
        st.markdown(r"""
        ### 8.1 Tipos fundamentais
        
        | Tipo | Banda passante | Inclinação de rejeição | Aplicação |
        |---|---|---|---|
        | **Passa-baixa** | $\omega < \omega_c$ | $-n \times 20$ dB/dec | Remoção de ruído; suavização |
        | **Passa-alta** | $\omega > \omega_c$ | $+n \times 20$ dB/dec | Remoção de DC; diferenciação |
        | **Passa-faixa** | $\omega_1 < \omega < \omega_2$ | ±$n \times 20$ nas bordas | Seleção de canal |
        | **Rejeita-faixa** (*notch*) | $\omega \notin [\omega_1,\omega_2]$ | Atenuação na faixa rejeitada | Eliminação de 60 Hz |
        
        ### 8.2 Filtro de Butterworth
        
        $$|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}$$
        
        Maximalmente plano na banda passante; atenuação de $-n \times 20$ dB/dec na banda de rejeição. Para $n=1$: equivale a $G(s) = 1/(1+s/\omega_c)$.
        """)
        
        wc8=1.0; w8=np.logspace(-2,2,2000)
        ordens8=[1,2,4,8]; cols8_bp=plt.cm.Blues(np.linspace(0.4,1.0,len(ordens8)))
        fig8a, axes8a = plt.subplots(2,2,figsize=(10.5,7.0))
        fig8a.suptitle("Filtros Butterworth — Magnitude (ordens 1,2,4,8)", fontsize=9, y=1.01)
        for (titulo8,ax8,tipo8) in [("Passa-baixa",axes8a[0,0],"LP"),("Passa-alta",axes8a[0,1],"HP"),
                                      ("Passa-faixa",axes8a[1,0],"BP"),("Rejeita-faixa",axes8a[1,1],"BS")]:
            for n8,col8 in zip(ordens8,cols8_bp):
                try:
                    if tipo8=="LP":   b8,a8=butter(n8,wc8,btype="low",analog=True)
                    elif tipo8=="HP": b8,a8=butter(n8,wc8,btype="high",analog=True)
                    elif tipo8=="BP": b8,a8=butter(n8,[0.5*wc8,2.0*wc8],btype="band",analog=True)
                    else:             b8,a8=butter(n8,[0.5*wc8,2.0*wc8],btype="bandstop",analog=True)
                    _,H8=sp_freqs(b8,a8,worN=w8)
                    ax8.semilogx(w8,20*np.log10(np.abs(H8)+1e-15),color=col8,lw=1.8,label=f"n={n8}")
                except Exception:
                    pass
            ax8.axhline(-3,color="gray",ls="--",lw=0.8,alpha=0.6)
            if tipo8 in ["LP","HP"]: ax8.axvline(wc8,color="red",ls=":",lw=1.0,alpha=0.7)
            ax8.set_title(titulo8,fontsize=9); ax8.set_xlabel("ω (rad/s)",fontsize=7)
            ax8.set_ylabel("Magnitude (dB)",fontsize=7); ax8.legend(fontsize=7)
            ax8.spines[["right","top"]].set_visible(False); ax8.set_ylim(-80,5)
        plt.tight_layout()
        show_fig(fig8a, 0.88)
        
        # Explorador filtros
        st.markdown("### 🎛️ Explorador — Filtros Butterworth")
        st.caption("Selecione o tipo e a ordem do filtro")
        
        c8a_, c8b_ = st.columns([1, 2])
        with c8a_:
            tipo8_sl  = st.selectbox("Tipo de filtro",
                                      ["Passa-baixa","Passa-alta","Passa-faixa","Rejeita-faixa"],
                                      key="tipo8sl")
            ordem8_sl = st.slider("Ordem $n$", 1, 10, 2, 1, key="ord8sl")
            wc8_sl    = st.slider("Freq. de corte $\\omega_c$ (rad/s)", 0.5, 50.0, 5.0, 0.5, key="wc8sl")
            st.info(f"Tipo: {tipo8_sl} · Ordem: {ordem8_sl} · ωc≈{wc8_sl} rad/s\n\n"
                    f"Inclinação de rejeição: {ordem8_sl*20} dB/dec")
        
        with c8b_:
            w8_e = np.logspace(-2, 3, 2000)
            tcode8 = {"Passa-baixa":"low","Passa-alta":"high",
                       "Passa-faixa":"band","Rejeita-faixa":"bandstop"}[tipo8_sl]
            try:
                if tcode8 in ["band","bandstop"]:
                    b8_e,a8_e = butter(ordem8_sl,[wc8_sl*0.4,wc8_sl*2.5],btype=tcode8,analog=True)
                else:
                    b8_e,a8_e = butter(ordem8_sl, wc8_sl, btype=tcode8, analog=True)
                _,H8_e = sp_freqs(b8_e, a8_e, worN=w8_e)
                mag8_e = 20*np.log10(np.abs(H8_e)+1e-15)
                fig_e8_ = go.Figure()
                fig_e8_.add_trace(go.Scatter(x=w8_e, y=mag8_e, mode="lines",
                    line=dict(color="#1f77b4",width=2.5), name="Magnitude"))
                fig_e8_.add_hline(y=-3, line_color="gray", line_dash="dash", line_width=0.9)
                fig_e8_.add_hline(y=0,  line_color="black",line_dash="dash", line_width=0.6)
                fig_e8_.update_xaxes(type="log", title_text="ω (rad/s)")
                fig_e8_.update_yaxes(title_text="Magnitude (dB)", range=[-85,5])
                fig_e8_.update_layout(
                    title=f"Butterworth {tipo8_sl} — n={ordem8_sl}, ωc≈{wc8_sl} rad/s",
                    height=360, margin=dict(t=50,b=40,l=60,r=20),
                    template="plotly_white", showlegend=False)
                st.plotly_chart(fig_e8_, use_container_width=True)
            except Exception as ex:
                st.error(f"Erro: {ex}")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 9 — EXPLORADOR GERAL
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("9. Explorador Geral de Bode")
        
        st.markdown("""
        Insira os coeficientes de $G(s) = N(s)/D(s)$ em **ordem decrescente** de $s$.
        
        **Exemplos prontos:**
        
        | $G(s)$ | Numerador | Denominador |
        |:---|:---|:---|
        | $10/[s(s+1)(s+2)]$ | `10` | `1, 3, 2, 0` |
        | $50/(s^2+5s+25)$ | `50` | `1, 5, 25` |
        | $(s+2)/(s^3+6s^2+11s+6)$ | `1, 2` | `1, 6, 11, 6` |
        | $1/[s(s+2)(s+4)(s+6)]$ | `1` | `1, 12, 44, 48, 0` |
        """)
        
        c9a, c9b = st.columns([1, 2])
        with c9a:
            num9_str = st.text_input("Numerador $N(s)$", value="10", key="num9")
            den9_str = st.text_input("Denominador $D(s)$", value="1, 3, 2, 0", key="den9")
            wmin9 = st.slider("$\\log_{10}(\\omega_{min})$", -3.0,  0.0, -1.0, 0.5, key="wmin9")
            wmax9 = st.slider("$\\log_{10}(\\omega_{max})$",  1.0,  5.0,  3.0, 0.5, key="wmax9")
        
        with c9b:
            try:
                num9 = [float(x.strip()) for x in num9_str.split(",") if x.strip()]
                den9 = [float(x.strip()) for x in den9_str.split(",") if x.strip()]
                assert len(den9) > len(num9), "Grau de Den deve ser maior que Num."
                w9 = np.logspace(wmin9, wmax9, 5000)
                mag9, fase9 = bode_calc(num9, den9, w9)
                dc9 = mag9[0]
        
                # ω de corte (−3 dB)
                idx_3db9 = np.where(mag9 <= dc9-3)[0]
                w_3db9   = w9[idx_3db9[0]] if len(idx_3db9) else None
        
                # cruzamento de ganho
                cross_gc9 = np.where(np.diff(np.sign(mag9)) != 0)[0]
                w_gc9 = w9[cross_gc9[0]] if len(cross_gc9) else None
                pm9   = (180 + fase9[cross_gc9[0]]) if len(cross_gc9) else None
        
                # cruzamento de fase
                cross_pc9 = np.where(np.diff(np.sign(fase9+180)) != 0)[0]
                w_pc9   = w9[cross_pc9[0]] if len(cross_pc9) else None
                gm9_db  = -mag9[cross_pc9[0]] if len(cross_pc9) else None
        
                fig_g9 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.55,0.45],
                                       subplot_titles=("Magnitude (dB)","Fase (°)"))
                fig_g9.add_trace(go.Scatter(x=w9,y=mag9,mode="lines",
                    line=dict(color="#1f77b4",width=2.2),name="Magnitude",
                    hovertemplate="ω=%{x:.3f}<br>%{y:.2f} dB<extra></extra>"),row=1,col=1)
                fig_g9.add_trace(go.Scatter(x=w9,y=fase9,mode="lines",
                    line=dict(color="#1f77b4",width=2.2),name="Fase",showlegend=False,
                    hovertemplate="ω=%{x:.3f}<br>%{y:.1f}°<extra></extra>"),row=2,col=1)
                fig_g9.add_hline(y=0,   line_color="black",line_dash="dash",line_width=0.7,row=1,col=1)
                fig_g9.add_hline(y=-180,line_color="black",line_dash="dash",line_width=0.7,row=2,col=1)
        
                info9=[]
                if w_3db9:
                    fig_g9.add_vline(x=w_3db9,line_color="seagreen",line_dash="dot",line_width=1.5)
                    fig_g9.add_trace(go.Scatter(x=[w_3db9],y=[dc9-3],mode="markers+text",
                        marker=dict(size=11,color="seagreen"),
                        text=[f"ωc={w_3db9:.3f}"],textposition="top right",
                        name=f"ωc={w_3db9:.3f} rad/s"),row=1,col=1)
                    fig_g9.add_vrect(x0=w9[0],x1=w_3db9,fillcolor="rgba(44,160,44,0.08)",
                                     line_width=0,row=1,col=1)
                    info9.append(f"$\\omega_c = {w_3db9:.4f}$ rad/s (−3 dB)")
                if w_gc9:
                    fig_g9.add_vline(x=w_gc9,line_color="#2ca02c",line_dash="dash",line_width=1.3)
                    fig_g9.add_trace(go.Scatter(x=[w_gc9],y=[0],mode="markers",
                        marker=dict(size=11,color="#2ca02c",symbol="diamond"),
                        name=f"ωgc={w_gc9:.3f}  φm={pm9:.1f}°"),row=1,col=1)
                    info9.append(f"$\\omega_{{gc}} = {w_gc9:.4f}$ · $\\phi_m = {pm9:.2f}°$")
                if w_pc9:
                    fig_g9.add_vline(x=w_pc9,line_color="#d62728",line_dash="dash",line_width=1.3)
                    fig_g9.add_trace(go.Scatter(x=[w_pc9],y=[-180],mode="markers",
                        marker=dict(size=11,color="#d62728",symbol="square"),
                        name=f"ωpc={w_pc9:.3f}  Gm={gm9_db:.1f} dB"),row=2,col=1)
                    info9.append(f"$\\omega_{{pc}} = {w_pc9:.4f}$ · $G_m = {gm9_db:.2f}$ dB")
        
                est9=""
                if pm9 is not None and gm9_db is not None:
                    est9 = "🟢 ESTÁVEL" if pm9>0 and gm9_db>0 else "🔴 INSTÁVEL/MARGINAL"
        
                title9 = f"Bode — N=[{num9_str}] / D=[{den9_str}]"
                if est9: title9 += f"  {est9}"
                fig_g9.update_xaxes(type="log",title_text="ω (rad/s)")
                fig_g9.update_yaxes(title_text="Magnitude (dB)",row=1,col=1)
                fig_g9.update_yaxes(title_text="Fase (°)",row=2,col=1)
                fig_g9.update_layout(title=dict(text=title9,font=dict(size=11)),
                    height=500,margin=dict(l=60,r=20,t=65,b=50),
                    template="plotly_white",legend=dict(orientation="h",y=1.08))
                st.plotly_chart(fig_g9,use_container_width=True)
                if info9:
                    st.info("\n\n".join(info9))
            except AssertionError as e:
                st.error(str(e))
            except Exception as ex:
                st.error(f"Erro: {ex}")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 10 — REFERÊNCIAS
        # ═══════════════════════════════════════════════════════════════════════════════
        with st.expander("10. Referências", expanded=False):
            st.markdown("""
        - **LATHI, B. P.; GREEN, R.** *Sinais e Sistemas Lineares*. 3ª ed. Oxford University Press, 2018.
        - **DORF, R. C.; BISHOP, R. H.** *Sistemas de Controle Modernos*. 13ª ed. LTC, 2017.
        - **OGATA, K.** *Engenharia de Controle Moderno*. 5ª ed. Pearson, 2014.
        - **NISE, N. S.** *Engenharia de Sistemas de Controle*. 7ª ed. Wiley / LTC, 2018.
        - **SEDRA, A. S.; SMITH, K. C.** *Microeletrônica*. 8ª ed. Pearson, 2020.
        - **DE SOUZA, A. C. Z.; PINHEIRO, C. A. M.** *Introdução à Modelagem, Análise e Simulação de Sistemas Dinâmicos*. 1ª ed. Interciência, 2008.
        - **CASTRUCCI, P. B. de L.; BITTAR, A.; SALES, R. M.** *Controle Automático*. 2ª ed. LTC, 2018.
        """)
        
        st.divider()
        
        st.markdown(
            "<div style='text-align:center;color:gray;font-size:12px'>"
            "📉 Resposta em Frequência &nbsp;·&nbsp; 🎛️ SINTONIA — Sistemas de Controle<br>"
            "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
            " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
            "</div>",
            unsafe_allow_html=True,
        )
