"""
⚖️ Estabilidade de Sistemas com Realimentação
Disciplina: Modelagem e Sistemas Lineares
Curso: Engenharia de Energia
Instituição: IFRN — Campus Natal-Central (CNAT)
Autor: Marcus V A Fernandes · marcus.fernandes@ifrn.edu.br · v1.0
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.signal import lti, lsim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
        
def run():

        warnings.filterwarnings("ignore")
        
        # ── Configuração da Página ────────────────────────────────────────────────────
        st.set_page_config(
            page_title="Estabilidade de Sistemas com Realimentação",
            page_icon="⚖️",
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
        
        COR = dict(est="royalblue", inst="crimson", marg="darkorange", ref="gray")
        
        # ── Helpers matplotlib ────────────────────────────────────────────────────────
        def estilo(ax, xlabel="t (s)", ylabel="y(t)"):
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        def plano_s_ax(ax, xlim=(-5, 3), ylim=(-5, 5)):
            ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
            ax.fill_betweenx([ylim[0], ylim[1]], xlim[0], 0,
                             alpha=0.07, color="seagreen", label="SPE (estável)")
            ax.fill_betweenx([ylim[0], ylim[1]], 0, xlim[1],
                             alpha=0.07, color="crimson", label="SPD (instável)")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\sigma$", fontsize=8)
            ax.set_ylabel(r"$j\omega$", fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
        
        def polo_x(ax, x, y, cor="#d62728", ms=12):
            ax.plot(x, y, "x", color=cor, ms=ms, mew=2.5, zorder=5)
        
        # ── Helpers Plotly ────────────────────────────────────────────────────────────
        def plotly_plano_s(fig, row, col, xlim=(-15, 5), ylim=(-12, 12)):
            fig.add_vrect(x0=xlim[0], x1=0, fillcolor="seagreen",
                          opacity=0.05, layer="below", line_width=0, row=row, col=col)
            fig.add_vrect(x0=0, x1=xlim[1], fillcolor="crimson",
                          opacity=0.05, layer="below", line_width=0, row=row, col=col)
            fig.update_xaxes(title_text="σ", range=list(xlim),
                             zeroline=True, zerolinecolor="black", row=row, col=col)
            fig.update_yaxes(title_text="jω", range=list(ylim),
                             zeroline=True, zerolinecolor="black", row=row, col=col)
        
        # ── Routh numérico ────────────────────────────────────────────────────────────
        def routh_table(coeffs):
            """Calcula a tabela de Routh e retorna (tabela, mudanças de sinal na 1ª col)."""
            n = len(coeffs)
            ncols = (n + 1) // 2
            EPS = 1e-10
            eps_sub = 1e-9  # valor para substituir zero isolado
            r1 = list(coeffs[0::2]) + [0] * (ncols - len(list(coeffs[0::2])))
            r2 = list(coeffs[1::2]) + [0] * (ncols - len(list(coeffs[1::2])))
            table = [r1[:ncols], r2[:ncols]]
            for _ in range(n - 2):
                prev, curr = table[-2], table[-1]
                pv = curr[0]
                if abs(pv) < EPS:
                    pv = eps_sub
                new = []
                for j in range(ncols - 1):
                    v = (curr[0] * prev[j + 1] - prev[0] * curr[j + 1]) / pv
                    new.append(v)
                new += [0] * (ncols - len(new))
                table.append(new[:ncols])
                if len(table) >= n:
                    break
            col1 = [row[0] for row in table]
            changes = 0
            prev_sign = None
            for v in col1:
                sg = 1 if v > EPS else (-1 if v < -EPS else 0)
                if sg == 0:
                    sg = 1  # epsilon já foi substituído
                if prev_sign is not None and sg != prev_sign:
                    changes += 1
                prev_sign = sg
            return table, changes, col1
        
        def estabilidade_str(roots):
            n_spd = sum(1 for r in roots if r.real > 1e-6)
            n_eim = sum(1 for r in roots if abs(r.real) < 1e-6)
            if n_spd > 0:
                return f"🔴 INSTÁVEL ({n_spd} polo(s) no SPD)"
            elif n_eim > 0:
                return f"🟡 MARGINALMENTE ESTÁVEL ({n_eim} polo(s) no eixo Im.)"
            return "🟢 ESTÁVEL (todos os polos no SPE)"
        
        def cor_polo(r):
            if r.real > 1e-6:   return COR["inst"]
            elif abs(r.real) < 1e-6: return COR["marg"]
            return COR["est"]
        
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
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CABEÇALHO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.title("⚖️ Estabilidade de Sistemas com Realimentação")
        st.subheader("Critério de Routh-Hurwitz e Região de Estabilidade")
        st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
        st.markdown("---")
        
        # ── Índice ────────────────────────────────────────────────────────────────────
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[1. Conceitos de Estabilidade](#1-conceitos-de-estabilidade)**
        - 1.1 Pontos de equilíbrio: estável, marginalmente estável, instável
        - 1.2 Estabilidade BIBO vs. Assintótica (Lyapunov)
        - 1.3 Condição de estabilidade para sistemas LTI causais
        
        **[2. Classificação pela Posição dos Polos](#2-classifica-o-pela-posi-o-dos-polos)**
        - Modos temporais: SPE, eixo imaginário (simples e repetido), SPD
        - Polo na origem: integrador simples vs. repetido
        
        **[3. Critério de Routh-Hurwitz](#3-crit-rio-de-routh-hurwitz)**
        - 3.1 Motivação: determinar estabilidade sem calcular raízes
        - 3.2 Construção da tabela de Routh
        - 3.3 Interpretação da 1ª coluna: mudanças de sinal → raízes no SPD
        - 3.4 Casos especiais: zero isolado ($\varepsilon \to 0^+$) e linha de zeros (polinômio auxiliar)
        
        **[4. Exemplo 1 — Sistema Instável](#4-exemplo-1-sistema-inst-vel)**
        - $D_{MF}(s)=s^3+10s^2+31s+1030$ — 2 raízes no SPD
        
        **[5. Exemplo 2 — Zero na Primeira Coluna](#5-exemplo-2-zero-na-primeira-coluna)**
        - $D(s)=s^5+2s^4+3s^3+6s^2+5s+3$ — substituição por $\varepsilon$
        
        **[6. Exemplo 3 — Linha de Zeros](#6-exemplo-3-linha-de-zeros)**
        - $D(s)=s^5+7s^4+6s^3+42s^2+8s+56$ — polinômio auxiliar, marginalmente estável
        
        **[7. Exemplo 4 — Região de Estabilidade (Ganho Crítico)](#7-exemplo-4-regi-o-de-estabilidade-ganho-cr-tico)**
        - $G(s)=1/[s(s+7)(s+11)]$: condição $0 < k < k_{crit}=1386$
        
        **[8. Explorador Interativo — Região de Estabilidade](#8-explorador-interativo-regi-o-de-estabilidade)**
        - 8.1 🎛️ Polos de MF em função de $k$ — 🟢 estável / 🟡 marginal / 🔴 instável
        - 8.2 Região de estabilidade no plano $(k, a_2)$
        
        **[9. Referências](#9-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Conceitos de Estabilidade")
        
        st.markdown(r"""
        ### 1.1 Pontos de equilíbrio
        
        Um sistema está em **equilíbrio** quando, na ausência de entrada, sua saída permanece constante.
        
        | Tipo | Comportamento após perturbação | Analogia física |
        |---|---|---|
        | **Estável** (assintótico) | Retorna ao equilíbrio original | Bola no fundo de uma tigela |
        | **Marginalmente estável** | Oscila com amplitude constante | Bola sobre superfície plana |
        | **Instável** | Afasta-se indefinidamente | Bola no topo de uma colina |
        
        ### 1.2 Estabilidade BIBO vs. Assintótica
        
        | Conceito | Definição |
        |---|---|
        | **BIBO** (*Bounded-Input Bounded-Output*) | Para qualquer entrada limitada, a saída é limitada |
        | **Assintótica** (interna / Lyapunov) | Todos os modos naturais tendem a zero quando $t \to \infty$ |
        
        Estabilidade **assintótica** $\Rightarrow$ estabilidade **BIBO**. O inverso não é garantido:
        um cancelamento polo-zero pode ocultar um modo instável na FT observada.
        
        ### 1.3 Condição de estabilidade para sistemas LTI causais
        
        | Condição sobre os polos | Estabilidade |
        |---|---|
        | Todos no **SPE**: $\text{Re}(s_i) < 0$ | **Assintoticamente estável** |
        | Nenhum no SPD; polos imaginários **simples** | **Marginalmente estável** |
        | $\geq 1$ polo no **SPD**, ou polos **repetidos** no eixo Im. | **Instável** |
        
        > Polos repetidos no eixo imaginário produzem termos $t\,e^{j\omega t}$ (amplitude crescente) — o sistema é **instável**, não marginalmente estável.
        """)
        
        # Figura 1.1 — Paisagem de energia
        def _landscape(xv):
            return (-1.2*np.exp(-2*(xv+1.3)**2) +
                     2.0*np.exp(-1.2*(xv-0.3)**2) +
                    -1.8*np.exp(-1.5*(xv-2.2)**2) + 2.5)
        
        xp = np.linspace(-2.0, 3.5, 800)
        yp = _landscape(xp)
        pts = {
            "meta":  (-1.30, float(_landscape(np.array([-1.30]))[0])),
            "inst":  ( 0.30, float(_landscape(np.array([ 0.30]))[0])),
            "noneq": ( 1.50, float(_landscape(np.array([ 1.50]))[0])),
            "est":   ( 2.20, float(_landscape(np.array([ 2.20]))[0])),
        }
        MS = 18
        
        fig1a, ax1a = plt.subplots(figsize=(7.5, 4.0))
        ax1a.plot(xp, yp, "k-", lw=2.5, zorder=2)
        ax1a.fill_between(xp, yp, yp.min()-0.4, alpha=0.06, color="steelblue", zorder=1)
        ax1a.plot(pts["meta"][0],  pts["meta"][1],  "o", ms=MS, color="#1f77b4", zorder=5, mec="#333", mew=0.8)
        ax1a.plot(pts["inst"][0],  pts["inst"][1],  "o", ms=MS, color="#d62728", zorder=5, mec="#333", mew=0.8)
        ax1a.plot(pts["noneq"][0], pts["noneq"][1], "o", ms=MS, color="#ff7f0e", zorder=5, mec="#333", mew=0.8)
        ax1a.plot(pts["est"][0],   pts["est"][1],   "o", ms=MS, color="#2ca02c", zorder=5, mec="#333", mew=0.8)
        ax1a.text(pts["inst"][0],  pts["inst"][1]  + 0.45, "Equilíbrio\ninstável",
                  ha="center", va="bottom", fontsize=8.5, color="#d62728", fontweight="bold")
        ax1a.text(pts["noneq"][0]+0.12, pts["noneq"][1]+0.42, "Não-equilíbrio\n(transitório)",
                  ha="left",   va="bottom", fontsize=8.5, color="#ff7f0e")
        ax1a.text(pts["meta"][0]+0.15,  pts["meta"][1]-0.44, "Equilíbrio\nmetaestável",
                  ha="left",   va="top",   fontsize=8.5, color="#1f77b4")
        ax1a.text(pts["est"][0],  pts["est"][1]-0.44, "Equilíbrio\nestável",
                  ha="center", va="top",   fontsize=8.5, color="#2ca02c", fontweight="bold")
        y_floor = yp.min() - 0.35
        for key, lbl in [("meta","Estado A"),("est","Estado B")]:
            ax1a.text(pts[key][0], y_floor, lbl, ha="center", fontsize=8.5, color="gray")
            ax1a.axvline(pts[key][0], color="gray", ls=":", lw=0.7, ymin=0, ymax=0.09)
        ax1a.set_xlabel("Coordenada de fase (estado do sistema)", fontsize=10)
        ax1a.set_ylabel("Energia potencial", fontsize=10)
        ax1a.set_xlim(-2.1, 3.6); ax1a.set_ylim(yp.min()-0.55, yp.max()+1.2)
        ax1a.spines[["right","top"]].set_visible(False)
        ax1a.set_xticks([]); ax1a.set_yticks([])
        ax1a.set_title("Paisagem de energia e pontos de equilíbrio", fontsize=9, pad=8)
        plt.tight_layout()
        show_fig(fig1a, 0.60)
        
        # Figura 1.2 — Analogia da bola
        Rb = 0.20
        fig1b, axes1b = plt.subplots(1, 3, figsize=(10.5, 3.8))
        fig1b.suptitle("Tipos de equilíbrio — analogia da bola na superfície", fontsize=9)
        for ax in axes1b:
            ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.7, 3.1)
            ax.set_aspect("equal"); ax.axis("off")
        
        # Instável
        ax1 = axes1b[0]
        xm = np.linspace(-2.0, 2.0, 400); ym = 2.2*np.exp(-0.55*xm**2)
        ax1.plot(xm, ym, "k-", lw=2.5)
        ax1.fill_between(xm, ym, -0.7, alpha=0.13, color="crimson")
        y_topo = float(np.interp(0.0, xm, ym))
        ax1.plot(0, y_topo+Rb, "o", ms=MS, color="#d62728", zorder=5, mec="#222", mew=0.8)
        for sign in [-1, +1]:
            ax1.annotate("", xy=(sign*1.3, y_topo+Rb), xytext=(sign*0.23, y_topo+Rb),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8))
        ax1.set_title("Equilíbrio Instável", fontsize=9, pad=5, color="#d62728")
        ax1.text(0, -0.58, "Perturbação → afasta-se", ha="center", fontsize=8, color="#d62728", style="italic")
        
        # Marginal
        ax2 = axes1b[1]
        xpl = np.linspace(-2.0, 2.0, 10)
        ax2.plot(xpl, np.zeros(10), "k-", lw=2.5)
        ax2.fill_between(xpl, np.zeros(10), -0.7, alpha=0.13, color="darkorange")
        ax2.plot(-0.6, Rb, "o", ms=MS, color="#888", alpha=0.4, zorder=4, mec="#888", mew=0.8)
        ax2.plot( 0.7, Rb, "o", ms=MS, color="#ff7f0e", zorder=5, mec="#222", mew=0.8)
        ax2.annotate("", xy=(0.7-Rb-0.04, Rb), xytext=(-0.6+Rb+0.04, Rb),
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.8))
        ax2.set_title("Equilíbrio Neutro (Marginal)", fontsize=9, pad=5, color="#e07000")
        ax2.text(0, -0.58, "Perturbação → novo equilíbrio", ha="center", fontsize=8, color="#e07000", style="italic")
        
        # Estável
        ax3 = axes1b[2]
        xb = np.linspace(-2.0, 2.0, 400); yb = 0.5*xb**2
        ax3.plot(xb, yb, "k-", lw=2.5)
        ax3.fill_between(xb, yb, -0.7, alpha=0.13, color="seagreen")
        y_fundo = float(np.interp(0.0, xb, yb))
        ax3.plot(0, y_fundo+Rb, "o", ms=MS, color="#2ca02c", zorder=5, mec="#222", mew=0.8)
        y_pert = float(np.interp(1.0, xb, yb))
        ax3.plot(1.0, y_pert+Rb, "o", ms=MS, color="#888", alpha=0.4, zorder=4, mec="#888", mew=0.8)
        ax3.annotate("", xy=(0.12, y_fundo+Rb+0.08), xytext=(1.0, y_pert+Rb+0.08),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.8,
                            connectionstyle="arc3,rad=-0.35"))
        ax3.set_title("Equilíbrio Estável", fontsize=9, pad=5, color="#2ca02c")
        ax3.text(0, -0.58, "Perturbação → retorna ao equilíbrio", ha="center", fontsize=8, color="#2ca02c", style="italic")
        plt.tight_layout()
        show_fig(fig1b, 0.82)
        
        # Figura 1.3 — Analogia do cone
        h_c=2.5; rx_c=1.2; GND=0.45; GND1=GND+rx_c*0.28
        L_c=np.sqrt(h_c**2+rx_c**2); alpha_c=np.arctan(rx_c/h_c)
        
        def _ground_c(ax, gnd):
            ax.plot([-2.0,2.0],[gnd,gnd],color="#555",lw=2.5,zorder=5)
            ax.fill_between([-2.0,2.0],[gnd]*2,[-0.05]*2,color="#ccc",alpha=0.4,zorder=0)
            ax.text(0,-0.03,"superfície",ha="center",fontsize=7,color="#888")
        
        fig1c, axes1c = plt.subplots(1,3,figsize=(10.5,4.6))
        fig1c.suptitle("Graus de equilíbrio — analogia do sólido cônico", fontsize=9)
        for ax in axes1c:
            ax.set_xlim(-2.2,2.2); ax.set_ylim(-0.05,4.3)
            ax.set_aspect("equal"); ax.axis("off")
        _ground_c(axes1c[0], GND1); _ground_c(axes1c[1], GND); _ground_c(axes1c[2], GND)
        
        # Estável: base ao solo
        ax1c = axes1c[0]
        by1,ty1 = GND,GND+h_c; ry1=rx_c*0.28
        ax1c.add_patch(plt.Polygon([(0-rx_c,by1),(0+rx_c,by1),(0,ty1)],fc="#e8ddc8",ec="#555",lw=1.6,zorder=3))
        ax1c.add_patch(Ellipse((0,by1),2*rx_c,2*ry1,fc="#c8b898",ec="#555",lw=1.6,zorder=4))
        ball_y1c=GND+ry1
        ax1c.plot(0,ball_y1c,"o",ms=MS,color="#2ca02c",mec="#222",mew=0.8,zorder=6)
        for sign in [-1,+1]:
            ax1c.annotate("",xy=(sign*0.1,ball_y1c),xytext=(sign*0.8,ball_y1c+0.3),
                arrowprops=dict(arrowstyle="->",color="#2ca02c",lw=1.7,connectionstyle="arc3,rad=-0.4"))
        ax1c.set_title("Equilíbrio Estável\n(Assintótico)",fontsize=8.5,color="#2ca02c",fontweight="bold",pad=5)
        ax1c.text(0,-0.04,"Base ao solo — c.g. baixo",ha="center",fontsize=8,color="#2ca02c",style="italic",va="top")
        
        # Metaestável: face lateral ao solo
        ax2c = axes1c[1]
        axis_v = np.array([np.cos(alpha_c),np.sin(alpha_c)])
        perp_v = np.array([-np.sin(alpha_c),np.cos(alpha_c)])
        tip2c  = np.array([-L_c/2,GND]); rim_gnd2=np.array([+L_c/2,GND])
        base_c2c = tip2c + h_c*axis_v; rim_top2 = base_c2c + rx_c*perp_v
        ax2c.add_patch(plt.Polygon([tip2c,rim_gnd2,rim_top2],fc="#e8ddc8",ec="#555",lw=1.6,zorder=3))
        ell_angle2=np.degrees(np.arctan2(perp_v[1],perp_v[0]))
        ax2c.add_patch(Ellipse(base_c2c,2*rx_c,2*rx_c*0.28,angle=ell_angle2,fc="#c8b898",ec="#555",lw=1.6,zorder=4))
        mid2c=tip2c*0.45+rim_top2*0.55; bp2c=mid2c+perp_v*0.22
        ax2c.plot(bp2c[0],bp2c[1],"o",ms=MS,color="#1f77b4",mec="#222",mew=0.8,zorder=6)
        topple=np.array([0.55,-0.45])
        ax2c.annotate("",xy=(bp2c+topple).tolist(),xytext=(bp2c+topple*0.08).tolist(),
            arrowprops=dict(arrowstyle="->",color="#1f77b4",lw=1.7))
        ax2c.set_title("Equilíbrio Metaestável",fontsize=8.5,color="#1f77b4",fontweight="bold",pad=5)
        ax2c.text(0,-0.04,"Face lateral ao solo",ha="center",fontsize=8,color="#1f77b4",style="italic",va="top")
        
        # Instável: ponta ao solo
        ax3c = axes1c[2]
        ty3,by3=GND,GND+h_c; ry3=rx_c*0.28
        ax3c.add_patch(plt.Polygon([(0-rx_c,by3),(0+rx_c,by3),(0,ty3)],fc="#e8ddc8",ec="#555",lw=1.6,zorder=3))
        ax3c.add_patch(Ellipse((0,by3),2*rx_c,2*ry3,fc="#c8b898",ec="#555",lw=1.6,zorder=4))
        ball_y3c=by3+ry3
        ax3c.plot(0,ball_y3c,"o",ms=MS,color="#d62728",mec="#222",mew=0.8,zorder=6)
        for sign in [-1,+1]:
            ax3c.annotate("",xy=(sign*1.22,ball_y3c),xytext=(sign*0.23,ball_y3c),
                arrowprops=dict(arrowstyle="->",color="#d62728",lw=1.8))
        ax3c.set_title("Equilíbrio Instável",fontsize=8.5,color="#d62728",fontweight="bold",pad=5)
        ax3c.text(0,-0.04,"Ponta ao solo — c.g. alto",ha="center",fontsize=8,color="#d62728",style="italic",va="top")
        plt.tight_layout()
        show_fig(fig1c, 0.82)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Classificação pela Posição dos Polos")
        
        st.markdown(r"""
        Para um polo $s_i = \sigma_i + j\omega_i$, o modo correspondente é:
        
        $$e^{s_i t} = e^{\sigma_i t}\!\left(\cos\omega_i t + j\sin\omega_i t\right)$$
        
        | Posição do polo | $\sigma_i$ | Comportamento temporal |
        |---|---|---|
        | SPE real | $< 0$ | Decaimento exponencial puro |
        | SPE complexo conjugado | $< 0$ | Oscilação amortecida |
        | Eixo Im. — **simples** | $= 0$ | Oscilação sustentada — **marginalmente estável** |
        | Eixo Im. — **repetido** | $= 0$ | Modo $t\cos\omega t$: crescente — **instável** |
        | Origem $s=0$ — simples | $= 0$ | Modo constante (integrador) — **marginalmente estável** |
        | Origem $s=0$ — repetido | $= 0$ | Modo $t$ (rampa): diverge — **instável** |
        | SPD real | $> 0$ | Crescimento exponencial puro |
        | SPD complexo conjugado | $> 0$ | Oscilação com amplitude crescente |
        
        > **Polo na origem simples:** a resposta ao **impulso** é uma constante (marginalmente estável), mas a resposta ao **degrau** é uma rampa (BIBO instável para essa entrada específica).
        """)
        
        t_s2 = np.linspace(0, 8, 600)
        casos2 = [
            ("SPE real $s=-2$",             -2, 0, COR["est"],  "-"),
            ("SPE complexo $s=-1\\pm3j$",   -1, 3, COR["est"],  "--"),
            ("Eixo Im. $s=\\pm3j$ (simples)", 0, 3, COR["marg"], "-."),
            ("SPD real $s=+1$",             +1, 0, COR["inst"], "-"),
            ("SPD complexo $s=+1\\pm3j$",   +1, 3, COR["inst"], "--"),
        ]
        fig2a, axes2a = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax = axes2a[0]
        plano_s_ax(ax, xlim=(-4,3), ylim=(-5,5))
        ax.set_title(r"Posição dos polos no plano $s$", fontsize=8.5)
        for lbl, sig, wd, col, ls in casos2:
            if wd == 0:
                polo_x(ax, sig, 0, cor=col)
            else:
                polo_x(ax, sig,  wd, cor=col)
                polo_x(ax, sig, -wd, cor=col)
        ax.legend(handles=[mpatches.Patch(color=c,label=l.split(" $")[0])
                           for l,_,_,c,_ in casos2], fontsize=7, loc="upper left")
        ax2 = axes2a[1]
        for lbl, sig, wd, col, ls in casos2:
            y = np.exp(sig*t_s2)*np.cos(wd*t_s2) if wd > 0 else np.exp(sig*t_s2)
            y = np.clip(y, -5, 5)
            ax2.plot(t_s2, y, color=col, ls=ls, lw=1.8, label=lbl)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.legend(fontsize=7, ncol=2)
        estilo(ax2, xlabel="t (s)", ylabel="modo")
        ax2.set_title("Modos temporais correspondentes", fontsize=8.5)
        ax2.set_xlim(0, 8); ax2.set_ylim(-5.2, 5.2)
        plt.tight_layout()
        show_fig(fig2a, 0.88)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Critério de Routh-Hurwitz")
        
        st.markdown(r"""
        ### 3.1 Motivação
        
        O critério permite, **sem calcular as raízes**:
        - Determinar quantos polos estão no semiplano direito
        - Verificar estabilidade marginal
        - Encontrar a **faixa de um parâmetro** $k$ que garante estabilidade
        
        ### 3.2 Construção da Tabela de Routh
        
        Dado $D(s) = a_n s^n + a_{n-1}s^{n-1} + \cdots + a_1 s + a_0$:
        
        **Passo 0 — Condição necessária:** todos os coeficientes $a_i$ devem ter o **mesmo sinal** e ser **não nulos**. Se algum for zero ou negativo, o sistema é imediatamente instável.
        
        **Passo 1 — Primeiras duas linhas** (coeficientes alternados):
        
        $$\begin{array}{c|cccc} s^n & a_n & a_{n-2} & a_{n-4} & \cdots \\ s^{n-1} & a_{n-1} & a_{n-3} & a_{n-5} & \cdots \end{array}$$
        
        **Passo 2 — Linhas seguintes** (determinantes cruzados):
        
        $$b_1 = \frac{a_{n-1}a_{n-2} - a_n a_{n-3}}{a_{n-1}}, \quad b_2 = \frac{a_{n-1}a_{n-4} - a_n a_{n-5}}{a_{n-1}}, \quad \ldots$$
        
        ### 3.3 Interpretação da 1ª coluna
        
        | Observação | Conclusão |
        |---|---|
        | Todos com o **mesmo sinal** | Sistema **estável** — zero raízes no SPD |
        | $N$ **mudanças de sinal** | Sistema **instável** — exatamente $N$ raízes no SPD |
        | **Linha inteira de zeros** | Há raízes simétricas — analisar o **polinômio auxiliar** |
        | **Zero isolado** na 1ª coluna | Substituir por $\varepsilon \to 0^+$ e continuar |
        
        ### 3.4 Casos especiais
        
        **Zero isolado na 1ª coluna:** substituir por $\varepsilon > 0$, avaliar o sinal no limite $\varepsilon \to 0^+$. Cada troca de sinal indica uma raiz no SPD.
        
        **Linha inteira de zeros:** a linha **imediatamente anterior** fornece o **polinômio auxiliar** $P(s)$. Substituir a linha de zeros por $P'(s) = dP/ds$, continuar a tabela e resolver $P(s)=0$ para localizar as raízes simétricas.
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Exemplo 1 — Sistema Instável")
        
        st.markdown(r"""
        ### Problema
        
        Planta em malha aberta: $G(s) = 1000/[(s+2)(s+3)(s+5)]$, $C(s)=1$
        
        Denominador de MF:
        
        $$D_{MF}(s) = (s+2)(s+3)(s+5) + 1000 = s^3 + 10s^2 + 31s + 1030$$
        
        ### Tabela de Routh
        
        $$\begin{array}{c|cc} s^3 & 1 & 31 \\ s^2 & 10 & 1030 \\ s^1 & (10\times31 - 1\times1030)/10 = -72 & 0 \\ s^0 & 1030 & \end{array}$$
        
        **1ª coluna:** $\{1,\;10,\;-72,\;1030\}$ — **2 mudanças de sinal** ($10\to-72$ e $-72\to1030$).
        
        **Conclusão:** sistema **instável** com **2 raízes no SPD**.
        
        > Todos os coeficientes são positivos (condição necessária satisfeita), mas a condição **suficiente** falha. Isso ilustra que a condição necessária **não é suficiente** para $n \geq 3$.
        """)
        
        coeffs1 = [1, 10, 31, 1030]
        roots1 = np.roots(coeffs1)
        t_arr1 = np.linspace(0, 2.5, 2000)
        table1, ch1, col1_1 = routh_table(coeffs1)
        
        col4a, col4b = st.columns(2)
        with col4a:
            st.markdown("**Tabela de Routh:**")
            headers = ["Linha"] + [f"col {j+1}" for j in range(len(table1[0]))]
            rows_disp = []
            for i, row in enumerate(table1):
                exp = len(coeffs1)-1-i
                rows_disp.append([f"s^{exp}"] + [f"{v:.4g}" for v in row])
            import pandas as pd
            df1 = pd.DataFrame(rows_disp, columns=headers)
            st.dataframe(df1, hide_index=True)
            st.info(f"**1ª coluna:** {[f'{v:.3g}' for v in col1_1]}\n\n"
                    f"**Mudanças de sinal:** {ch1} → {estabilidade_str(roots1)}")
        
        with col4b:
            fig4a, axes4a = plt.subplots(1, 2, figsize=(6.5, 3.2))
            ax = axes4a[0]
            plano_s_ax(ax, xlim=(-6,4), ylim=(-35,35))
            for r in roots1:
                polo_x(ax, r.real, r.imag, cor=cor_polo(r))
            ax.legend(fontsize=7); ax.set_title("Polos de MF", fontsize=8.5)
            ax2 = axes4a[1]
            try:
                _, y1, _ = lsim(lti([1000], coeffs1), np.ones(len(t_arr1)), t_arr1)
                y1 = np.clip(y1, -50, 50)
                ax2.plot(t_arr1, y1, color=COR["inst"], lw=2.0)
            except Exception:
                pass
            ax2.axhline(0, color="k", lw=0.5, ls="--")
            estilo(ax2); ax2.set_xlim(0, 2.5)
            ax2.set_title("Resposta ao degrau", fontsize=8.5)
            plt.tight_layout()
            show_fig(fig4a, 1.0)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Exemplo 2 — Zero na Primeira Coluna")
        
        st.markdown(r"""
        ### Problema
        
        $$D(s) = s^5 + 2s^4 + 3s^3 + 6s^2 + 5s + 3$$
        
        Ao calcular a linha $s^3$, o elemento da 1ª coluna é exatamente **zero** (mas a linha não é toda zero).
        
        ### Procedimento — substituição por $\varepsilon \to 0^+$
        
        $$\begin{array}{c|ccc} s^5 & 1 & 3 & 5 \\ s^4 & 2 & 6 & 3 \\ s^3 & \varepsilon & 3{,}5 & 0 \\ s^2 & 6 - 7/\varepsilon & 3 & 0 \\ s^1 & \cdots(+) & 0 & \\ s^0 & 3 & & \end{array}$$
        
        Quando $\varepsilon \to 0^+$: $6 - 7/\varepsilon \to -\infty$ (negativo).
        
        **1ª coluna:** $\{1^+,\;2^+,\;\varepsilon^+,\;{-\infty}^-,\;\ldots^+,\;3^+\}$ — **2 mudanças de sinal**.
        
        **Conclusão:** sistema **instável** com **2 raízes no SPD**.
        """)
        
        coeffs2 = [1, 2, 3, 6, 5, 3]
        roots2 = np.roots(coeffs2)
        t_arr2 = np.linspace(0, 5, 1000)
        table2, ch2, col1_2 = routh_table(coeffs2)
        
        col5a, col5b = st.columns(2)
        with col5a:
            st.markdown("**Tabela de Routh (com $\\varepsilon$):**")
            rows_disp2 = []
            for i, row in enumerate(table2):
                exp = len(coeffs2)-1-i
                rows_disp2.append([f"s^{exp}"] + [f"{v:.4g}" for v in row])
            df2 = pd.DataFrame(rows_disp2, columns=["Linha"]+[f"col {j+1}" for j in range(len(table2[0]))])
            st.dataframe(df2, hide_index=True)
            n_spd2 = sum(1 for r in roots2 if r.real > 1e-6)
            st.info(f"**Mudanças de sinal:** {ch2} → {estabilidade_str(roots2)}\n\n"
                    + "\n".join([f"s = {r:.4f}" for r in sorted(roots2, key=lambda x: x.real)]))
        
        with col5b:
            fig5a, axes5a = plt.subplots(1, 2, figsize=(6.5, 3.2))
            ax = axes5a[0]
            plano_s_ax(ax, xlim=(-3,2), ylim=(-4,4))
            for r in roots2:
                polo_x(ax, r.real, r.imag, cor=cor_polo(r))
            ax.legend(fontsize=7); ax.set_title("Polos", fontsize=8.5)
            ax2 = axes5a[1]
            try:
                _, y2, _ = lsim(lti([3], coeffs2), np.ones(len(t_arr2)), t_arr2)
                y2 = np.clip(y2, -20, 20)
                ax2.plot(t_arr2, y2, color=COR["inst"] if n_spd2 > 0 else COR["est"], lw=2.0)
            except Exception:
                pass
            ax2.axhline(0, color="k", lw=0.5, ls="--")
            estilo(ax2); ax2.set_xlim(0, 5)
            ax2.set_title("Resposta ao degrau", fontsize=8.5)
            plt.tight_layout()
            show_fig(fig5a, 1.0)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Exemplo 3 — Linha de Zeros")
        
        st.markdown(r"""
        ### Problema
        
        $$D(s) = s^5 + 7s^4 + 6s^3 + 42s^2 + 8s + 56$$
        
        Ao calcular a linha $s^3$, **todos os elementos são zero** — linha de zeros.
        
        ### Procedimento — Polinômio Auxiliar
        
        1. Linha anterior à de zeros ($s^4$): $P(s) = 7s^4 + 42s^2 + 56$ → **polinômio auxiliar**
        2. Derivar: $P'(s) = 28s^3 + 84s$ → coeficientes substituem a linha de zeros
        3. Continuar a tabela normalmente
        4. Resolver $P(s)=0$: $7(s^2+2)(s^2+4)$ → raízes $s = \pm j\sqrt{2}$ e $s = \pm 2j$ — **imaginárias puras**
        
        $$\begin{array}{c|ccc} s^5 & 1 & 6 & 8 \\ s^4 & 7 & 42 & 56 \\ s^3 & \mathbf{28} & \mathbf{84} & 0 \quad\leftarrow P'(s) \\ s^2 & 21 & 56 & 0 \\ s^1 & \ldots(+) & 0 & \\ s^0 & 56 & & \end{array}$$
        
        **Conclusão:** sistema **marginalmente estável** — nenhuma raiz no SPD; duas raízes simples no eixo Im.
        """)
        
        coeffs3 = [1, 7, 6, 42, 8, 56]
        roots3 = np.roots(coeffs3)
        t_arr3 = np.linspace(0, 15, 3000)
        
        col6a, col6b = st.columns(2)
        with col6a:
            st.info(f"**{estabilidade_str(roots3)}**\n\n"
                    + "\n".join([f"s = {r:.4f}" for r in sorted(roots3, key=lambda x: x.real)]))
            st.markdown("Polinômio auxiliar $P(s) = 7s^4 + 42s^2 + 56 = 7(s^2+2)(s^2+4)$")
            st.markdown("Raízes: $s = \\pm j\\sqrt{2} \\approx \\pm 1.414j$ e $s = \\pm 2j$")
        
        with col6b:
            fig6a, axes6a = plt.subplots(1, 2, figsize=(6.5, 3.2))
            ax = axes6a[0]
            plano_s_ax(ax, xlim=(-4,2), ylim=(-4,4))
            for r in roots3:
                polo_x(ax, r.real, r.imag, cor=cor_polo(r))
            ax.legend(fontsize=7); ax.set_title("Polos (eixo Im.)", fontsize=8.5)
            ax2 = axes6a[1]
            try:
                _, y3, _ = lsim(lti([10], coeffs3), np.ones(len(t_arr3)), t_arr3)
                y3 = np.clip(y3, -5, 5)
                ax2.plot(t_arr3, y3, color=COR["marg"], lw=2.0, label="oscilação sustentada")
            except Exception:
                pass
            ax2.axhline(0, color="k", lw=0.5)
            estilo(ax2); ax2.set_xlim(0, 15)
            ax2.set_title("Resposta ao degrau (marginal)", fontsize=8.5)
            ax2.legend(fontsize=7)
            plt.tight_layout()
            show_fig(fig6a, 1.0)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("7. Exemplo 4 — Região de Estabilidade (Ganho Crítico)")
        
        st.markdown(r"""
        ### Problema
        
        $$G(s) = \frac{1}{s(s+7)(s+11)}, \qquad C(s) = k$$
        
        $$D_{MF}(s) = s^3 + 18s^2 + 77s + k$$
        
        ### Tabela de Routh com $k$ simbólico
        
        $$\begin{array}{c|cc} s^3 & 1 & 77 \\ s^2 & 18 & k \\ s^1 & (18\times77 - k)/18 = (1386-k)/18 & 0 \\ s^0 & k & \end{array}$$
        
        Para todos os elementos da 1ª coluna positivos:
        
        $$k > 0 \quad\text{e}\quad 1386 - k > 0 \quad\Rightarrow\quad \boxed{0 < k < k_{crit} = 1386}$$
        
        ### Ganho crítico e frequência de oscilação
        
        Para $k = 1386$: polinômio auxiliar $P(s) = 18s^2 + 1386 = 18(s^2+77)$
        
        Raízes: $s = \pm j\sqrt{77} \approx \pm j\,8{,}77$ rad/s
        
        > **Interpretação:** para $k < 1386$ os polos de MF estão no SPE. Em $k = 1386$ dois polos cruzam o eixo imaginário. Para $k > 1386$ entram no SPD e a amplitude cresce sem limite.
        """)
        
        k_crit7 = 18 * 77  # = 1386
        wn_crit7 = np.sqrt(77)
        ks7 = [100, 500, 1000, 1386, 1600, 2000]
        t_arr7 = np.linspace(0, 20, 4000)
        cols7 = plt.cm.RdYlGn(np.linspace(0.05, 0.95, len(ks7)))
        
        fig7a, axes7a = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax = axes7a[0]
        plano_s_ax(ax, xlim=(-15,5), ylim=(-12,12))
        ax2 = axes7a[1]
        for kv, col in zip(ks7, cols7):
            den7 = [1, 18, 77, kv]
            roots7 = np.roots(den7)
            lbl = f"k={kv}" + (" (crit)" if kv==k_crit7 else " (inst)" if kv>k_crit7 else "")
            for r in roots7:
                polo_x(ax, r.real, r.imag, cor=col, ms=10)
            try:
                _, y7, _ = lsim(lti([kv], den7), np.ones(len(t_arr7)), t_arr7)
                y7 = np.clip(y7, -4, 4)
                ax2.plot(t_arr7, y7, color=col, lw=1.7, label=lbl)
            except Exception:
                pass
        ax.set_title(r"Exemplo 4 — polos MF para vários $k$", fontsize=8.5)
        ax.legend(handles=[mpatches.Patch(color=c,label=f"k={k}") for k,c in zip(ks7,cols7)],
                  fontsize=7, loc="upper right")
        ax2.axhline(1.0, color=COR["ref"], lw=0.8, ls="--")
        ax2.axhline(0.0, color="k", lw=0.5)
        estilo(ax2); ax2.set_xlim(0, 20); ax2.set_ylim(-3, 4)
        ax2.set_title(r"Resposta ao degrau — $G(s)=1/[s(s+7)(s+11)]$", fontsize=8.5)
        ax2.legend(fontsize=7, ncol=2)
        plt.suptitle(rf"$k_{{crit}}=1386$ · $\omega_n=\sqrt{{77}}\approx{wn_crit7:.2f}$ rad/s",
                     fontsize=9, fontweight="bold")
        plt.tight_layout()
        show_fig(fig7a, 0.88)
        
        st.info(f"$k_{{crit}} = 18 \\times 77 = {k_crit7}$ · "
                f"$\\omega_n = \\sqrt{{77}} \\approx {wn_crit7:.4f}$ rad/s · "
                f"$T = 2\\pi/\\omega_n \\approx {2*np.pi/wn_crit7:.4f}$ s")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 8 — EXPLORADOR INTERATIVO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("8. Explorador Interativo — Região de Estabilidade")
        
        st.markdown(r"""
        ### 8.1 Polos de MF em função do ganho $k$
        
        Para $G(s) = 1/[s(s+a_1)(s+a_2)]$ com controlador proporcional $C(s)=k$:
        
        $$D_{MF}(s) = s^3 + (a_1+a_2)s^2 + a_1 a_2\,s + k, \qquad k_{crit} = a_1\,a_2\,(a_1+a_2)$$
        
        Varie $k$ e observe os polos cruzarem o eixo imaginário em $k = k_{crit}$.
        """)
        st.caption("🟢 Estável (todos polos no SPE) · 🟡 Marginalmente estável (polos no eixo Im.) · 🔴 Instável (polo(s) no SPD)")
        
        a1_e8 = 7.0; a2_e8 = 11.0
        k_crit_e8 = a1_e8 * a2_e8 * (a1_e8 + a2_e8)
        
        c8a, c8b = st.columns([1, 2])
        with c8a:
            k8_sl = st.slider("Ganho $k$", 50.0, float(k_crit_e8 + 400),
                              500.0, 10.0, key="k8sl")
            den8_e = [1, a1_e8+a2_e8, a1_e8*a2_e8, k8_sl]
            roots8_e = np.roots(den8_e)
            n_spd8 = sum(1 for r in roots8_e if r.real > 1e-6)
            n_eim8 = sum(1 for r in roots8_e if abs(r.real) < 1e-6)
            if n_spd8 > 0:
                cor8 = "#d62728"; st8 = "🔴 INSTÁVEL"
            elif n_eim8 > 0:
                cor8 = "#ff7f0e"; st8 = "🟡 MARGINAL"
            else:
                cor8 = "#2ca02c"; st8 = "🟢 ESTÁVEL"
            st.info(f"**{st8}**\n\n"
                    f"$k={k8_sl:.0f}$ · $k_{{crit}}={k_crit_e8:.0f}$\n\n" +
                    "\n".join([f"polo: $s={r.real:.3f}{r.imag:+.3f}j$"
                               for r in sorted(roots8_e, key=lambda x: x.real)]))
        
        with c8b:
            t_e8 = np.linspace(0, 20, 2000)
            try:
                _, y_e8, _ = lsim(lti([k8_sl], den8_e), np.ones(len(t_e8)), t_e8)
                y_e8 = np.clip(y_e8, -4, 4)
            except Exception:
                y_e8 = np.zeros_like(t_e8)
        
            fig_e8 = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Plano s — polos de MF","Resposta ao degrau"),
                                   horizontal_spacing=0.10)
            plotly_plano_s(fig_e8, 1, 1, xlim=(-15,5), ylim=(-12,12))
            fig_e8.add_trace(go.Scatter(
                x=[r.real for r in roots8_e], y=[r.imag for r in roots8_e],
                mode="markers",
                marker=dict(symbol="x", size=14, color=cor8, line=dict(width=3, color=cor8)),
                showlegend=False,
                hovertemplate="polo=(%{x:.3f}, %{y:.3f}j)<extra></extra>"), row=1, col=1)
            fig_e8.add_trace(go.Scatter(
                x=t_e8, y=y_e8, mode="lines",
                line=dict(color=cor8, width=2.2), showlegend=False), row=1, col=2)
            fig_e8.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_e8.add_hline(y=0.0, line_width=0.6, line_color="black", row=1, col=2)
            fig_e8.update_xaxes(title_text="t (s)", row=1, col=2)
            fig_e8.update_yaxes(title_text="y(t)", range=[-3,4], row=1, col=2)
            fig_e8.update_layout(height=320, margin=dict(t=30,b=10,l=15,r=10),
                                 template="plotly_white")
            st.plotly_chart(fig_e8, use_container_width=True)
        
        # ── 8.2 Região de estabilidade no plano (k, a2) ───────────────────────────────
        st.markdown("### 8.2 Região de Estabilidade no Plano $(k,\\ a_2)$")
        st.markdown(r"""
        Com $a_1 = 7$ fixo, a fronteira de estabilidade é:
        
        $$k_{crit}(a_2) = a_1\,a_2\,(a_1 + a_2) = 7\,a_2\,(7 + a_2)$$
        """)
        
        c8c, c8d = st.columns([1, 2])
        with c8c:
            a2_mark = st.slider("Marcar ponto $a_2$", 0.5, 20.0, 11.0, 0.5, key="a2mark8")
            k_mark  = st.slider("Marcar ponto $k$",   0.0, 5000.0, 500.0, 50.0, key="kmark8")
            k_crit_mark = 7.0 * a2_mark * (7.0 + a2_mark)
            if k_mark < k_crit_mark:
                reg_mark = "🟢 ESTÁVEL"
            elif abs(k_mark - k_crit_mark) < 30:
                reg_mark = "🟡 FRONTEIRA (marginal)"
            else:
                reg_mark = "🔴 INSTÁVEL"
            st.info(f"Ponto: $a_2={a2_mark:.1f}$, $k={k_mark:.0f}$\n\n"
                    f"$k_{{crit}}={k_crit_mark:.1f}$\n\n**{reg_mark}**")
        
        with c8d:
            a2_arr8 = np.linspace(0.5, 20, 500)
            kc_arr8 = 7.0 * a2_arr8 * (7.0 + a2_arr8)
            fig_reg8 = go.Figure()
            fig_reg8.add_trace(go.Scatter(
                x=np.concatenate([a2_arr8, a2_arr8[::-1]]),
                y=np.concatenate([kc_arr8, np.zeros(len(a2_arr8))]),
                fill="toself", fillcolor="rgba(44,160,44,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="Região estável"))
            fig_reg8.add_trace(go.Scatter(
                x=a2_arr8, y=kc_arr8, mode="lines",
                line=dict(color="darkorange", width=2.5, dash="dash"),
                name=r"k_crit = 7·a2·(7+a2)",
                hovertemplate="a2=%{x:.2f}<br>k_crit=%{y:.1f}<extra></extra>"))
            # ponto marcado
            cor_m = "#2ca02c" if "ESTÁVEL" in reg_mark and "INST" not in reg_mark else (
                    "#ff7f0e" if "FRONT" in reg_mark else "#d62728")
            fig_reg8.add_trace(go.Scatter(
                x=[a2_mark], y=[k_mark], mode="markers+text",
                marker=dict(size=14, color=cor_m, symbol="circle"),
                text=[reg_mark.split(" ")[0]],
                textposition="top right", textfont=dict(size=10, color=cor_m),
                name="Ponto selecionado"))
            fig_reg8.add_annotation(x=6, y=500, text="ESTÁVEL<br>(k < k_crit)",
                showarrow=False, font=dict(size=12, color="seagreen"))
            fig_reg8.add_annotation(x=16, y=3500, text="INSTÁVEL<br>(k > k_crit)",
                showarrow=False, font=dict(size=12, color="crimson"))
            fig_reg8.update_layout(
                title=dict(text=r"Região de estabilidade — G(s)=1/[s(s+7)(s+a2)] com ganho k",
                           font=dict(size=11)),
                xaxis=dict(title="a2 (polo da planta)", range=[0,21]),
                yaxis=dict(title="k (ganho)", range=[0,5000]),
                height=380, margin=dict(l=60,r=30,t=60,b=50),
                template="plotly_white",
                legend=dict(x=0.02, y=0.95))
            st.plotly_chart(fig_reg8, use_container_width=True)
        
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
            "⚖️ Estabilidade de Sistemas com Realimentação &nbsp;·&nbsp; 🎛️ SINTONIA — Sistemas de Controle<br>"
            "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
            " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
            "</div>",
            unsafe_allow_html=True,
        )
