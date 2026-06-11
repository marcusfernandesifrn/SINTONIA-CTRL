"""
🧮 Análise de Sistemas no Espaço de Estados
Disciplina: Modelagem e Sistemas Lineares
Curso: Engenharia de Energia
Instituição: IFRN — Campus Natal-Central (CNAT)
Autor: Marcus V A Fernandes · marcus.fernandes@ifrn.edu.br · v1.0
"""
        
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import expm
from scipy import signal
import control
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
        
        
def run():
        
        warnings.filterwarnings("ignore")
        
        # ── Configuração da Página ────────────────────────────────────────────────────
        st.set_page_config(
            page_title="Espaço de Estados",
            page_icon="🧮",
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
        
        # ── Helpers de formatação ─────────────────────────────────────────────────────
        def mat_str(M, fmt=".4g", name="M"):
            """Formata numpy matrix como LaTeX bmatrix."""
            M = np.atleast_2d(np.array(M, dtype=float))
            rows = []
            for row in M:
                def fv(v):
                    return "0" if abs(v) < 1e-10 else f"{v:{fmt}}"
                rows.append(" & ".join(fv(v) for v in row))
            return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"
        
        def poly_str(coeffs, var="s"):
            """Converte coeficientes em string LaTeX de polinômio."""
            n = len(coeffs) - 1; terms = []
            for i, c in enumerate(coeffs):
                p = n - i
                if abs(c) < 1e-10: continue
                ca = abs(float(c))
                sg = "-" if c < 0 else ("+" if terms else "")
                cs = f"{ca:.4g}" if (ca != 1 or p == 0) else ""
                if p == 0:   terms.append(f"{sg}{ca:.4g}")
                elif p == 1: terms.append(f"{sg}{cs}{var}")
                else:        terms.append(f"{sg}{cs}{var}^{{{p}}}")
            return "".join(terms) or "0"
        
        def parse_matrix(s):
            """Parseia string 'a,b;c,d' em numpy array."""
            rows = [[float(v.strip()) for v in row.split(",")]
                    for row in s.strip().split(";")]
            return np.array(rows, dtype=float)
        
        def ss_diagnostics(A, B, C):
            """Retorna (eigvals, stable, ctrl, obsv, Wc, Wo)."""
            ev = np.linalg.eigvals(A)
            stable = bool(all(v.real < 0 for v in ev))
            Wc = control.ctrb(A, B)
            Wo = control.obsv(A, C)
            n = A.shape[0]
            ctrl = (np.linalg.matrix_rank(Wc) == n)
            obsv = (np.linalg.matrix_rank(Wo) == n)
            return ev, stable, ctrl, obsv, Wc, Wo
        
        def render_diagnostics(A, B, C, D):
            """Exibe diagnóstico completo de um sistema em espaço de estados."""
            n = A.shape[0]
            ev, stable, ctrl, obsv, Wc, Wo = ss_diagnostics(A, B, C)
            sys_ss = control.ss(A, B, C, D)
            sys_tf = control.ss2tf(sys_ss)
            num_tf = np.array(sys_tf.num[0][0]); den_tf = np.array(sys_tf.den[0][0])
            tol = 1e-10 * max(np.abs(den_tf)); num_tf[np.abs(num_tf)<tol]=0.0
            char = np.real(np.poly(ev))
            polos_str = ", ".join(
                f"{v.real:.4f}" if abs(v.imag)<1e-8 else f"{v.real:.4f}{v.imag:+.4f}j"
                for v in sorted(ev, key=lambda z: z.real))
            zeros_tf = np.roots(num_tf)
            zeros_str = (", ".join(
                f"{z.real:.4f}" if abs(z.imag)<1e-8 else f"{z.real:.4f}{z.imag:+.4f}j"
                for z in sorted(zeros_tf, key=lambda z: z.real)) if len(zeros_tf)>0 else "—")
            est_icon = "🟢 ESTÁVEL" if stable else "🔴 INSTÁVEL"
            ctrl_icon = "✅ Controlável" if ctrl else "❌ Não controlável"
            obsv_icon = "✅ Observável" if obsv else "❌ Não observável"
            st.info(
                f"**Ordem:** $n={n}$ · **{est_icon}** · **{ctrl_icon}** · **{obsv_icon}**\n\n"
                f"**Polos:** $[{polos_str}]$\n\n"
                f"**Zeros:** $[{zeros_str}]$\n\n"
                f"**Pol. característico:** $\\det(sI-A)={poly_str(np.round(char,6))}$\n\n"
                f"**$H(s) = $** $\\dfrac{{{poly_str(num_tf)}}}{{{poly_str(den_tf)}}}$\n\n"
                f"$\\text{{rank}}(W_c)={np.linalg.matrix_rank(Wc)}/{n}$ · "
                f"$\\text{{rank}}(W_o)={np.linalg.matrix_rank(Wo)}/{n}$"
            )
            return ev, stable, sys_tf, sys_ss
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CABEÇALHO
        # ═══════════════════════════════════════════════════════════════════════════════
        st.title("🧮 Análise de Sistemas no Espaço de Estados")
        st.subheader("Representação, Solução, Controlabilidade e Observabilidade")
        st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
        st.markdown("---")
        
        with st.expander("📋 Índice — clique para expandir", expanded=False):
            st.markdown(r"""
        **[1. Descrição Interna e Externa](#1-descri-o-interna-e-externa-de-um-sistema)**
        - Descrição externa (FT, caixa preta) vs. interna (espaço de estados)
        - Quando as duas descrições são equivalentes
        
        **[2. Espaço de Estados — Conceitos Fundamentais](#2-espa-o-de-estados-conceitos-fundamentais)**
        - 2.1 Equações de estado e saída: $\dot{\mathbf{x}}=A\mathbf{x}+B\mathbf{u}$, $\mathbf{y}=C\mathbf{x}+D\mathbf{u}$
        - 2.2 Dimensões das matrizes $A, B, C, D$
        - 2.3 Escolha das variáveis de estado
        - 2.4 Vantagens da representação
        
        **[3. Equações de Estado a partir de Funções de Transferência](#3-equa-es-de-estado-a-partir-de-fun-es-de-transfer-ncia)**
        - 3.1 Forma canônica do controlador (CCF)
        - 3.2 Forma canônica do observador (OCF)
        - 3.3 Forma em cascata
        - 3.4 Forma paralela (frações parciais) — variantes I e II
        - Exemplos: $G(s)=s(s+2)/[(s+1)(s^2+2s+5)]$ e sistema de 5ª ordem
        
        **[4. Solução no Domínio de Laplace](#4-solu-o-no-dom-nio-de-laplace)**
        - 4.1 Transformada das equações de estado: $(sI-A)\mathbf{X}(s) = \mathbf{x}(0)+B\mathbf{U}(s)$
        - 4.2 Componentes entrada nula e estado nulo
        - 4.3 Função de transferência: $H(s) = C(sI-A)^{-1}B+D$
        - Verificação numérica: decomposição da resposta
        
        **[5. Solução no Domínio do Tempo — Matriz de Transição](#5-solu-o-no-dom-nio-do-tempo-matriz-de-transi-o-de-estado)**
        - 5.1 Solução geral com $e^{At}$
        - 5.2 Propriedades de $\Phi(t)=e^{At}$
        - 5.3 Cálculo via Teorema de Cayley-Hamilton
        - 5.4 Transformação de similaridade e forma modal (diagonalização)
        - Visualização: trajetória no espaço de estados e resposta temporal
        
        **[6. Controlabilidade e Observabilidade](#6-controlabilidade-e-observabilidade)**
        - 6.1 Critério de Kalman: $\text{rank}(W_c)=n$ e $\text{rank}(W_o)=n$
        - 6.2 Interpretação física e exemplos com cancelamento polo-zero
        - 6.3 Formas canônicas e dualidade
        
        **[7. Explorador Interativo de Espaço de Estados](#7-explorador-interativo-de-espa-o-de-estados)**
        - 7.1 🎛️ Explorador SS — insira $A,B,C,D$ e veja resposta, trajetória, diagnóstico
        - 7.2 🎛️ Conversor FT → SS — insira $N(s)/D(s)$
        - 7.3 🎛️ Conversor SS → FT — insira $A,B,C,D$
        
        **[8. Referências](#8-refer-ncias)**
        """)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("1. Descrição Interna e Externa de um Sistema")
        
        st.markdown(r"""
        ### 1.1 Descrição externa
        
        A **descrição externa** é a relação de entrada-saída — tipicamente $H(s)=Y(s)/U(s)$ para
        sistemas SISO, ou a **matriz de transferência** $\mathbf{H}(s)$ para MIMO.
        Pode ser obtida por medições nos terminais, mesmo quando o interior é inacessível (caixa preta).
        
        ### 1.2 Descrição interna
        
        A **descrição interna** fornece informação completa sobre todos os sinais do sistema.
        Cada modo natural corresponde a um autovalor da matriz $A$ — o conjunto de todos os modos
        constitui o **espaço de estados**.
        
        | Característica | Descrição Externa | Descrição Interna |
        |---|---|---|
        | Informação | Relação entrada-saída | Todos os sinais, incluindo estados internos |
        | Obtenção | Medição de terminais | Modelo matemático completo |
        | Sistemas MIMO | Matriz $\mathbf{H}(s)$ de dimensão $q\times p$ | Matrizes $A,B,C,D$ — representação unificada |
        | Deriva da outra? | Sim (da interna) | Em geral, não (da externa) |
        
        > Uma descrição externa pode sempre ser determinada a partir de uma interna, mas o inverso
        > não é necessariamente válido — modos não controláveis ou não observáveis são invisíveis
        > na descrição externa.
        
        ### 1.3 Quando as descrições são equivalentes?
        
        As descrições interna e externa são equivalentes **apenas** quando o sistema é simultaneamente
        **controlável** e **observável**. Caso contrário, ocorrem **cancelamentos polo-zero** na
        função de transferência: o polo do modo problemático é cancelado por um zero correspondente.
        
        > **Implicação prática:** sistemas com modos não controláveis ou não observáveis podem ser
        > instáveis mesmo que a FT aparente estabilidade.
        """)
        
        fig1a, axes1a = plt.subplots(1, 2, figsize=(10.5, 3.8))
        for ax, title, has_internal in [
                (axes1a[0], "Descrição Externa\n(caixa preta)", False),
                (axes1a[1], "Descrição Interna\n(espaço de estados)", True)]:
            ax.set_xlim(0,10); ax.set_ylim(0,5); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(title, fontsize=9, pad=6)
            ax.add_patch(mpatches.FancyBboxPatch((2,1.5),6,2,
                boxstyle="square,pad=0.0",fc="white",ec="#333",lw=1.8,zorder=2))
            if not has_internal:
                ax.text(5,2.5,r"$H(s)$",ha="center",va="center",fontsize=16,zorder=3)
                ax.text(5,1.85,"caixa preta",ha="center",va="center",fontsize=8,color="gray",zorder=3)
            else:
                for xp, yp, lbl in [(3.5,2.5,r"$x_1$"),(5.0,2.5,r"$x_2$"),(6.5,2.5,r"$x_3$")]:
                    ax.add_patch(plt.Circle((xp,yp),0.35,fc="white",ec="#1f77b4",lw=1.4,zorder=4))
                    ax.text(xp,yp,lbl,ha="center",va="center",fontsize=9,zorder=5)
                for x1, x2 in [(3.85,4.65),(5.35,6.15)]:
                    ax.annotate("",xy=(x2,2.5),xytext=(x1,2.5),
                        arrowprops=dict(arrowstyle="->",color="#1f77b4",lw=1.3))
                ax.text(5.0,1.85,r"$\dot{\mathbf{x}}=A\mathbf{x}+B\mathbf{u}$",
                        ha="center",va="center",fontsize=9,color="#1f77b4",zorder=3)
            ax.annotate("",xy=(2.0,2.5),xytext=(0.5,2.5),
                arrowprops=dict(arrowstyle="->",color="#333",lw=1.6))
            ax.text(0.55,2.78,r"$u(t)$",fontsize=9)
            ax.annotate("",xy=(9.5,2.5),xytext=(8.0,2.5),
                arrowprops=dict(arrowstyle="->",color="#333",lw=1.6))
            ax.text(8.05,2.78,r"$y(t)$",fontsize=9)
        fig1a.suptitle("Descrição de sistemas: externa vs. interna",fontsize=9)
        plt.tight_layout()
        show_fig(fig1a, 0.82)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("2. Espaço de Estados — Conceitos Fundamentais")
        
        st.markdown(r"""
        ### 2.1 Equações de estado e saída
        
        Para sistemas **LTI** (Lineares Invariantes no Tempo):
        
        $$\boxed{\dot{\mathbf{x}}(t) = A\,\mathbf{x}(t) + B\,\mathbf{u}(t)} \qquad\text{(equação de estado)}$$
        
        $$\boxed{\mathbf{y}(t) = C\,\mathbf{x}(t) + D\,\mathbf{u}(t)} \qquad\text{(equação de saída)}$$
        
        ### 2.2 Dimensões das matrizes
        
        | Símbolo | Nome | Dimensão |
        |---|---|---|
        | $\mathbf{x}(t)\in\mathbb{R}^n$ | Vetor de estado | $n\times 1$ |
        | $\mathbf{u}(t)\in\mathbb{R}^p$ | Vetor de entrada | $p\times 1$ |
        | $\mathbf{y}(t)\in\mathbb{R}^q$ | Vetor de saída | $q\times 1$ |
        | $A\in\mathbb{R}^{n\times n}$ | Matriz de sistema (dinâmica) | $n\times n$ |
        | $B\in\mathbb{R}^{n\times p}$ | Matriz de entrada | $n\times p$ |
        | $C\in\mathbb{R}^{q\times n}$ | Matriz de saída | $q\times n$ |
        | $D\in\mathbb{R}^{q\times p}$ | Matriz de transmissão direta (*feedthrough*) | $q\times p$ |
        
        > **Ordem do sistema:** $n$ = número de variáveis de estado = grau de $\det(sI-A)$.
        
        ### 2.3 Escolha das variáveis de estado
        
        A representação não é única — qualquer transformação linear invertível $\mathbf{z}=T\mathbf{x}$
        produz outra representação válida. Escolhas comuns:
        
        - **Variáveis físicas:** tensão em capacitores, corrente em indutores, posição/velocidade
        - **Variáveis de fase:** derivadas sucessivas da saída
        - **Formas canônicas:** controlador, observador, modal (diagonal)
        
        ### 2.4 Vantagens da representação
        
        - Trata sistemas **SISO e MIMO** de forma uniforme
        - Permite simulação computacional direta via integração de $\dot{\mathbf{x}}=A\mathbf{x}+Bu$
        - Fornece análise de controlabilidade, observabilidade e estabilidade diretamente de $A$, $B$, $C$
        - Extensão natural para sistemas não lineares e controle moderno
        """)
        
        # Diagrama de blocos em espaço de estados
        fig2a, ax2a = plt.subplots(figsize=(12.5, 4.2))
        ax2a.set_xlim(0.3,13.2); ax2a.set_ylim(0.2,4.3)
        ax2a.set_aspect("equal"); ax2a.axis("off")
        
        def _blk(ax,cx,cy,w,h,txt,fs=10):
            ax.add_patch(mpatches.FancyBboxPatch((cx-w/2,cy-h/2),w,h,
                boxstyle="square,pad=0.0",fc="white",ec="#333",lw=1.5,zorder=4))
            ax.text(cx,cy,txt,ha="center",va="center",fontsize=fs,zorder=5)
        def _seta(ax,x1,y1,x2,y2):
            ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->",color="#333",lw=1.5,mutation_scale=14),zorder=3)
        def _linha(ax,x1,y1,x2,y2):
            ax.plot([x1,x2],[y1,y2],"-",color="#333",lw=1.5,zorder=3)
        
        xB2,yB2,wB2,hB2=2.0,3.0,1.4,0.9
        xS1_2,yS1_2=3.8,3.0; xI2,yI2,wI2,hI2=6.0,3.0,1.8,0.9
        xBif2,yBif2=7.8,3.0; xC2,yC2,wC2,hC2=9.5,3.0,1.4,0.9
        xS2_2,yS2_2=11.4,3.0; xA2,yA2,wA2,hA2=5.0,1.5,1.8,0.9
        xD2,yD2,wD2,hD2=9.5,0.9,1.4,0.9; R2=0.28
        
        for xc,yc in [(xS1_2,yS1_2),(xS2_2,yS2_2)]:
            ax2a.add_patch(plt.Circle((xc,yc),R2,fc="white",ec="#333",lw=1.5,zorder=4))
        _blk(ax2a,xB2,yB2,wB2,hB2,r"$B$"); _blk(ax2a,xI2,yI2,wI2,hI2,r"$\int$",fs=15)
        _blk(ax2a,xC2,yC2,wC2,hC2,r"$C$"); _blk(ax2a,xA2,yA2,wA2,hA2,r"$A$")
        _blk(ax2a,xD2,yD2,wD2,hD2,r"$D$")
        for xc,yc,lbl in [(xS1_2-R2-0.05,yS1_2+0.28,"+"),(xS1_2+0.08,yS1_2-R2-0.24,"+"),
                           (xS2_2-R2-0.05,yS2_2+0.28,"+"),(xS2_2+0.08,yS2_2-R2-0.24,"+")]:
            ax2a.text(xc,yc,lbl,fontsize=11,ha="right" if "−" in lbl else "center")
        _seta(ax2a,0.5,yB2,xB2-wB2/2,yB2); ax2a.text(0.55,yB2+0.22,r"$\mathbf{u}(t)$",fontsize=9)
        _seta(ax2a,xB2+wB2/2,yB2,xS1_2-R2,yS1_2)
        _seta(ax2a,xS1_2+R2,yS1_2,xI2-wI2/2,yI2)
        _linha(ax2a,xI2+wI2/2,yI2,xBif2,yBif2)
        ax2a.plot(xBif2,yBif2,"o",color="#333",ms=6,zorder=5)
        ax2a.text(xBif2+0.05,yBif2+0.22,r"$\mathbf{x}(t)$",fontsize=9)
        _seta(ax2a,xBif2,yBif2,xC2-wC2/2,yC2)
        _seta(ax2a,xC2+wC2/2,yC2,xS2_2-R2,yS2_2)
        _seta(ax2a,xS2_2+R2,yS2_2,12.8,yS2_2); ax2a.text(12.3,yS2_2+0.22,r"$\mathbf{y}(t)$",fontsize=9)
        _linha(ax2a,xBif2,yBif2,xBif2,yA2)
        _seta(ax2a,xBif2,yA2,xA2+wA2/2,yA2)
        _linha(ax2a,xA2-wA2/2,yA2,xS1_2,yA2)
        _seta(ax2a,xS1_2,yA2,xS1_2,yS1_2-R2)
        _linha(ax2a,0.5,yB2,0.5,yD2)
        _seta(ax2a,0.5,yD2,xD2-wD2/2,yD2)
        _linha(ax2a,xD2+wD2/2,yD2,xS2_2,yD2)
        _seta(ax2a,xS2_2,yD2,xS2_2,yS2_2-R2)
        ax2a.set_title(r"Diagrama de blocos — $\dot{\mathbf{x}}=A\mathbf{x}+B\mathbf{u}$,  $\mathbf{y}=C\mathbf{x}+D\mathbf{u}$",
                       fontsize=9,pad=6)
        plt.tight_layout()
        show_fig(fig2a, 0.85)
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("3. Equações de Estado a partir de Funções de Transferência")
        
        st.markdown(r"""
        Dada $H(s) = \dfrac{b_N s^N + \cdots + b_0}{s^N + a_{N-1}s^{N-1} + \cdots + a_0}$
        
        ### 3.1 Forma canônica do controlador (CCF)
        
        Variáveis de fase: $q_k = \dot{q}_{k-1}$.
        
        $$A = \begin{bmatrix}0&1&0&\cdots&0\\0&0&1&\cdots&0\\\vdots&&&&\vdots\\0&0&0&\cdots&1\\-a_0&-a_1&-a_2&\cdots&-a_{N-1}\end{bmatrix}, \quad B=\begin{bmatrix}0\\0\\\vdots\\0\\1\end{bmatrix}, \quad C=\begin{bmatrix}b_0&b_1&\cdots&b_{N-1}\end{bmatrix}, \quad D=[b_N]$$
        
        > Última linha de $A$: coeficientes do denominador em ordem **crescente** com sinal negativo.
        
        ### 3.2 Forma canônica do observador (OCF)
        
        Transposta da CCF. Controlabilidade depende de $B$; observabilidade garantida.
        
        ### 3.3 Forma em cascata
        
        Para $H(s) = k/[(s-\lambda_1)\cdots(s-\lambda_N)]$ com polos distintos — matriz triangular superior.
        
        ### 3.4 Forma paralela (frações parciais)
        
        $$H(s) = b_N + \frac{k_1}{s-\lambda_1} + \cdots + \frac{k_N}{s-\lambda_N}$$
        
        $A = \text{diag}(\lambda_1,\ldots,\lambda_N)$ — desacoplada, ideal para análise modal.
        
        > Para polos **repetidos**, usar blocos de Jordan $\begin{bmatrix}\lambda&1\\0&\lambda\end{bmatrix}$.
        """)
        
        # Exemplo numérico G=s(s+2)/[(s+1)(s²+2s+5)]
        num_e1=[1,2,0]; den_e1=[1,3,7,5]
        sys_tf1=control.tf(num_e1,den_e1)
        sys_cc1=control.tf2ss(sys_tf1)
        A_cc1=np.array(sys_cc1.A); B_cc1=np.array(sys_cc1.B)
        C_cc1=np.array(sys_cc1.C); D_cc1=np.array(sys_cc1.D)
        
        st.markdown("#### Exemplo 1 — $G(s)=s(s+2)/[(s+1)(s^2+2s+5)]$")
        col3a, col3b = st.columns(2)
        with col3a:
            st.markdown("**Forma Canônica do Controlador:**")
            st.latex(r"\dot{\mathbf{x}} = " + mat_str(A_cc1) +
                     r"\,\mathbf{x} + " + mat_str(B_cc1) + r"\,\mathbf{u}")
            st.latex(r"\mathbf{y} = " + mat_str(C_cc1) +
                     r"\,\mathbf{x} + " + mat_str(D_cc1) + r"\,\mathbf{u}")
        with col3b:
            ev3, stable3, _, obsv3, Wc3, Wo3 = ss_diagnostics(A_cc1, B_cc1, C_cc1)
            n3 = A_cc1.shape[0]
            polos_str3 = ", ".join(
                f"{v.real:.4f}" if abs(v.imag)<1e-8 else f"{v.real:.4f}{v.imag:+.4f}j"
                for v in sorted(ev3, key=lambda z: z.real))
            st.info(f"**{'🟢 ESTÁVEL' if stable3 else '🔴 INSTÁVEL'}**\n\n"
                    f"**Polos:** $[{polos_str3}]$\n\n"
                    f"$\\text{{rank}}(W_c)={np.linalg.matrix_rank(Wc3)}/{n3}$ · "
                    f"$\\text{{rank}}(W_o)={np.linalg.matrix_rank(Wo3)}/{n3}$")
        
        st.markdown("#### Exemplo 2 — Comparação de três realizações para $G(s)=(2s+6)/(s^2+3s+2)$")
        num_r=[2,6]; den_r=[1,3,2]
        sys_tf_r=control.tf(num_r,den_r)
        sys_dir=control.tf2ss(sys_tf_r)
        A_dir=np.array(sys_dir.A); B_dir=np.array(sys_dir.B)
        C_dir=np.array(sys_dir.C); D_dir=np.array(sys_dir.D)
        A_par=np.array([[-1.,0.],[0.,-2.]]); B_par=np.array([[1.],[1.]])
        C_par=np.array([[4.,-2.]]); D_par=np.array([[0.]])
        A_cas=np.array([[-2.,0.],[1.,-1.]]); B_cas=np.array([[1.],[0.]])
        C_cas=np.array([[4.,4.]]); D_cas=np.array([[0.]])
        
        tabs3=st.tabs(["Forma Direta","Forma Paralela","Forma Cascata"])
        for tab, (nome, Av, Bv, Cv, Dv) in zip(tabs3, [
            ("Forma Direta",   A_dir,B_dir,C_dir,D_dir),
            ("Forma Paralela", A_par,B_par,C_par,D_par),
            ("Forma Cascata",  A_cas,B_cas,C_cas,D_cas)]):
            with tab:
                c1t, c2t = st.columns(2)
                with c1t:
                    st.latex(r"\dot{\mathbf{x}}=" + mat_str(Av) +
                             r"\mathbf{x}+" + mat_str(Bv) + r"\mathbf{u}")
                    st.latex(r"\mathbf{y}=" + mat_str(Cv) +
                             r"\mathbf{x}+" + mat_str(Dv) + r"\mathbf{u}")
                with c2t:
                    sys_v=control.ss(Av,Bv,Cv,Dv); tf_v=control.ss2tf(sys_v)
                    n_v=np.array(tf_v.num[0][0]); d_v=np.array(tf_v.den[0][0])
                    ev_v=np.linalg.eigvals(Av)
                    p_v=", ".join(f"{v.real:.3f}" if abs(v.imag)<1e-8 else f"{v.real:.3f}{v.imag:+.3f}j"
                                  for v in sorted(ev_v,key=lambda z:z.real))
                    st.info(f"$H(s)=\\dfrac{{{poly_str(n_v)}}}{{{poly_str(d_v)}}}$\n\nPolos: $[{p_v}]$")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("4. Solução no Domínio de Laplace")
        
        st.markdown(r"""
        ### 4.1 Transformada de Laplace das equações de estado
        
        $$s\,\mathbf{X}(s) - \mathbf{x}(0) = A\,\mathbf{X}(s) + B\,\mathbf{U}(s)$$
        
        $$\boxed{\mathbf{X}(s) = (sI-A)^{-1}\mathbf{x}(0) + (sI-A)^{-1}B\,\mathbf{U}(s)}$$
        
        ### 4.2 Equação de saída
        
        $$\boxed{\mathbf{Y}(s) = \underbrace{C(sI-A)^{-1}\mathbf{x}(0)}_{\text{entrada nula}} + \underbrace{\bigl[C(sI-A)^{-1}B+D\bigr]\mathbf{U}(s)}_{\text{estado nulo}}}$$
        
        | Componente | Terminologia | Condição |
        |---|---|---|
        | $C(sI-A)^{-1}\mathbf{x}(0)$ | **Entrada nula** (*zero-input*) | $\mathbf{u}(t)=0$, $\mathbf{x}(0)\neq 0$ |
        | $[C(sI-A)^{-1}B+D]\mathbf{U}(s)$ | **Estado nulo** (*zero-state*) | $\mathbf{x}(0)=0$, $\mathbf{u}(t)\neq 0$ |
        
        ### 4.3 Função de transferência
        
        $$\boxed{H(s) = C(sI-A)^{-1}B + D}$$
        
        O **polinômio característico** $\det(sI-A)$ é o denominador comum de $(sI-A)^{-1}$.
        Suas raízes (autovalores $\lambda_i$ de $A$) coincidem com os **polos** para realizações mínimas.
        
        > **Relação com $e^{At}$:** $\;e^{At} = \mathcal{L}^{-1}\{(sI-A)^{-1}\}$
        """)
        
        # Demonstração numérica: decomposição da resposta
        A4=np.array([[-3.,1.],[0.,-1.]])
        B4=np.array([[0.],[1.]])
        C4=np.array([[1.,2.]]); D4=np.array([[0.]])
        sys4=control.ss(A4,B4,C4,D4)
        t4=np.linspace(0,5,2000); u4=np.ones_like(t4); x0_4=np.array([1.0,0.5])
        t_tot,y_tot,_=control.forced_response(sys4,T=t4,U=u4,X0=x0_4,return_x=True)
        t_zi, y_zi, _=control.forced_response(sys4,T=t4,U=np.zeros_like(t4),X0=x0_4,return_x=True)
        t_zs, y_zs, _=control.forced_response(sys4,T=t4,U=u4,X0=np.zeros(2),return_x=True)
        
        fig4a, ax4a = plt.subplots(figsize=(8.5,3.4))
        ax4a.plot(t_tot, y_tot, "k-",  lw=2.5, label=r"Resposta total $y(t)$")
        ax4a.plot(t_zi,  y_zi,  "b--", lw=2.0, label=r"Entrada nula $y_{zi}$  [$\mathbf{x}(0)\neq0$, $u=0$]")
        ax4a.plot(t_zs,  y_zs,  "r-.", lw=2.0, label=r"Estado nulo $y_{zs}$  [$\mathbf{x}(0)=0$, $u\neq0$]")
        ax4a.set_xlabel("t (s)",fontsize=8); ax4a.set_ylabel("y(t)",fontsize=8)
        ax4a.legend(fontsize=7); ax4a.spines[["right","top"]].set_visible(False)
        ax4a.set_title(r"Decomposição da resposta: $y=y_{zi}+y_{zs}$  ($\mathbf{x}(0)=[1,\;0.5]^T$, degrau)",
                       fontsize=9)
        plt.tight_layout()
        show_fig(fig4a, 0.72)
        err4 = np.max(np.abs(y_tot-(y_zi+y_zs)))
        st.info(f"Verificação: max|y_total - (y_zi + y_zs)| = {err4:.2e}")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("5. Solução no Domínio do Tempo — Matriz de Transição de Estado")
        
        st.markdown(r"""
        ### 5.1 Solução geral
        
        $$\boxed{\mathbf{x}(t) = \underbrace{e^{At}\mathbf{x}(0)}_{\text{entrada nula}} + \underbrace{\int_0^t e^{A(t-\tau)}B\,\mathbf{u}(\tau)\,d\tau}_{\text{estado nulo (integral de Duhamel)}}}$$
        
        ### 5.2 Propriedades de $\Phi(t) = e^{At}$
        
        $$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$
        
        | Propriedade | Expressão |
        |---|---|
        | Valor inicial | $\Phi(0) = I$ |
        | Derivada | $\dot{\Phi}(t) = A\,\Phi(t) = \Phi(t)\,A$ |
        | Inversa | $\Phi^{-1}(t) = \Phi(-t) = e^{-At}$ |
        | Composição | $\Phi(t_1+t_2) = \Phi(t_1)\Phi(t_2)$ |
        | Relação com Laplace | $\Phi(t) = \mathcal{L}^{-1}\{(sI-A)^{-1}\}$ |
        
        ### 5.3 Cálculo via Cayley-Hamilton
        
        $$e^{At} = \beta_0(t)I + \beta_1(t)A + \cdots + \beta_{N-1}(t)A^{N-1}$$
        
        Para autovalores distintos, os $\beta_i(t)$ são determinados pelo sistema de Vandermonde: $e^{\lambda_i t} = \beta_0 + \beta_1\lambda_i + \cdots + \beta_{N-1}\lambda_i^{N-1}$.
        
        ### 5.4 Transformação de similaridade e forma modal
        
        $$\bar{A} = P A P^{-1}, \qquad \bar{B} = P B, \qquad \bar{C} = C P^{-1}$$
        
        FT e autovalores são **invariantes**. Se $P$ = matriz de autovetores, $\bar{A}=\Lambda$ (diagonal):
        modo $m$ é **controlável** se linha $m$ de $\bar{B}\neq 0$; **observável** se coluna $m$ de $\bar{C}\neq 0$.
        """)
        
        # Verificação de propriedades
        A5=np.array([[-1.,1.],[0.,-2.]])
        eigvals5=np.linalg.eigvals(A5)
        p5_str=", ".join(f"{v:.4f}" for v in eigvals5)
        t1_5,t2_5=0.5,0.7
        P1_5=expm(A5*t1_5); P2_5=expm(A5*t2_5)
        P12_5=expm(A5*(t1_5+t2_5)); Pinv_5=expm(-A5*t1_5)
        
        st.markdown(f"**Verificação das propriedades de $e^{{At}}$ para $A=[[-1,1],[0,-2]]$:**  "
                    f"autovalores $\\lambda=[{p5_str}]$")
        col5a, col5b, col5c = st.columns(3)
        for col, prop, val, ok in [
            (col5a, r"$\Phi(0)=I$",                 None, np.allclose(expm(A5*0),np.eye(2))),
            (col5b, r"$\Phi(t_1)\Phi(t_2)=\Phi(t_1+t_2)$",None,np.allclose(P1_5@P2_5,P12_5)),
            (col5c, r"$\Phi(t)\Phi(-t)=I$",         None, np.allclose(P1_5@Pinv_5,np.eye(2))),
        ]:
            col.info(f"{'✅' if ok else '❌'} {prop}")
        
        # Resposta oscilatória amortecida
        wn5,xi5=3.0,0.3
        A5b=np.array([[0.,1.],[-wn5**2,-2*xi5*wn5]])
        B5b=np.array([[0.],[wn5**2]]); C5b=np.array([[1.,0.]]); D5b=np.array([[0.]])
        t5b=np.linspace(0,5,1000); x0_5=np.array([1.0,0.0])
        x_traj5=np.array([expm(A5b*ti)@x0_5 for ti in t5b])
        
        fig5a, axes5a = plt.subplots(1,2,figsize=(9.5,3.6))
        ax = axes5a[0]
        ax.plot(x_traj5[:,0],x_traj5[:,1],"b-",lw=2)
        ax.plot(x0_5[0],x0_5[1],"go",ms=10,label=r"$\mathbf{x}(0)$")
        ax.plot(0,0,"r*",ms=12,label="Equilíbrio")
        mid5=len(t5b)//4
        ax.annotate("",xy=(x_traj5[mid5+5,0],x_traj5[mid5+5,1]),
            xytext=(x_traj5[mid5,0],x_traj5[mid5,1]),
            arrowprops=dict(arrowstyle="->",color="blue",lw=1.5))
        ax.set_xlabel(r"$x_1$ (posição)",fontsize=8); ax.set_ylabel(r"$x_2$ (velocidade)",fontsize=8)
        ax.set_title("Trajetória no espaço de estados",fontsize=8.5)
        ax.legend(fontsize=7); ax.spines[["right","top"]].set_visible(False)
        
        ax2 = axes5a[1]
        ax2.plot(t5b,x_traj5[:,0],"b-",lw=2,label=r"$x_1(t)$ — posição")
        ax2.plot(t5b,x_traj5[:,1],"r--",lw=2,label=r"$x_2(t)$ — velocidade")
        ax2.axhline(0,color="k",lw=0.5,ls="--",alpha=0.4)
        ax2.set_xlabel("t (s)",fontsize=8); ax2.set_ylabel("Estado",fontsize=8)
        ax2.set_title(rf"Resposta de entrada nula: $\omega_n={wn5}$, $\xi={xi5}$",fontsize=8.5)
        ax2.legend(fontsize=7); ax2.spines[["right","top"]].set_visible(False)
        plt.tight_layout()
        show_fig(fig5a, 0.88)
        
        # Transformação de similaridade — forma modal
        A5m=np.array([[-3.,1.,0.],[-7.,0.,1.],[-5.,0.,0.]])
        B5m=np.array([[1.],[2.],[0.]]); C5m=np.array([[1.,0.,0.]]); D5m=np.array([[0.]])
        ev5m,Pinv5m=np.linalg.eig(A5m); P5m=np.linalg.inv(Pinv5m)
        A_mod=np.real(P5m@A5m@Pinv5m); B_mod=np.real(P5m@B5m); C_mod=np.real(C5m@Pinv5m)
        
        st.markdown("#### Transformação de similaridade — forma modal")
        col5c1, col5c2 = st.columns(2)
        with col5c1:
            st.markdown("**Sistema original:**")
            st.latex(r"A=" + mat_str(A5m) + r",\quad B=" + mat_str(B5m))
            ev5m_str=", ".join(f"{v.real:.4f}" if abs(v.imag)<1e-8 else f"{v.real:.4f}{v.imag:+.4f}j"
                               for v in sorted(ev5m,key=lambda z:z.real))
            st.info(f"Autovalores: $\\lambda=[{ev5m_str}]$")
        with col5c2:
            st.markdown("**Forma modal** $\\bar{A}=PAP^{-1}$:")
            st.latex(r"\bar{A}=" + mat_str(np.round(A_mod,4)))
            rows_co5=[]
            for m in range(len(ev5m)):
                ctrl_m5=np.any(np.abs(B_mod[m,:])>1e-10)
                obsv_m5=np.any(np.abs(C_mod[:,m])>1e-10)
                lam5=f"{ev5m[m].real:.3f}" if abs(ev5m[m].imag)<1e-8 else f"{ev5m[m].real:.3f}{ev5m[m].imag:+.3f}j"
                rows_co5.append(f"$\\lambda={lam5}$: {'✅ctrl' if ctrl_m5 else '❌ctrl'} · {'✅obs' if obsv_m5 else '❌obs'}")
            st.info("\n\n".join(rows_co5))
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("6. Controlabilidade e Observabilidade")
        
        st.markdown(r"""
        ### 6.1 Controlabilidade — Critério de Kalman
        
        Um sistema é **completamente controlável** se para qualquer estado inicial existe uma entrada de
        duração finita que transfere o sistema a qualquer estado desejado.
        
        $$\mathbf{W}_c = \begin{bmatrix}B & AB & A^2B & \cdots & A^{n-1}B\end{bmatrix} \in \mathbb{R}^{n\times np}$$
        
        $$\text{rank}(\mathbf{W}_c) = n \quad\Leftrightarrow\quad\text{sistema controlável}$$
        
        ### 6.2 Observabilidade — Critério de Kalman
        
        Um sistema é **completamente observável** se o estado inicial pode ser determinado
        univocamente a partir de entrada e saída em intervalo finito.
        
        $$\mathbf{W}_o = \begin{bmatrix}C\\ CA\\ CA^2\\ \vdots\\ CA^{n-1}\end{bmatrix} \in \mathbb{R}^{nq\times n}$$
        
        $$\text{rank}(\mathbf{W}_o) = n \quad\Leftrightarrow\quad\text{sistema observável}$$
        
        ### 6.3 Formas canônicas e dualidade
        
        | Forma | Controlabilidade | Observabilidade |
        |---|---|---|
        | **CCF** (controlador) | $\text{rank}(W_c)=n$ garantido | Depende de $C$ |
        | **OCF** (observador) | Depende de $B$ | $\text{rank}(W_o)=n$ garantido |
        | **Modal** | Colunas de $\bar{B}$ não nulas | Linhas de $\bar{C}$ não nulas |
        
        > **Cancelamento polo-zero:** autovalor $\lambda_k$ que também é zero de $H(s)$ indica modo
        > **não controlável ou não observável** — existe fisicamente mas não aparece na FT.
        
        > **Dualidade:** $(A,B)$ controlável $\Leftrightarrow$ $(A^T,B^T)$ observável.
        """)
        
        # Exemplos numéricos
        A_nc6=np.array([[-1.,0.],[0.,-2.]]); B_nc6=np.array([[1.],[0.]])
        C_nc6=np.array([[1.,1.]]); D_nc6=np.array([[0.]])
        A_no6=np.array([[-1.,0.],[0.,-2.]]); B_no6=np.array([[1.],[1.]])
        C_no6=np.array([[1.,0.]]); D_no6=np.array([[0.]])
        
        fig6a, axes6a = plt.subplots(1,3,figsize=(11.5,4.0))
        A_co6=np.array([[-1.,1.],[-2.,-3.]]); B_co6=np.array([[0.],[1.]])
        C_co6=np.array([[1.,0.]])
        t6=np.linspace(0,6,500); u6=np.ones_like(t6)
        configs6=[(axes6a[0],A_co6,B_co6,C_co6,np.zeros((1,1)),"Controlável e Observável","seagreen"),
                   (axes6a[1],A_nc6,B_nc6,C_nc6,D_nc6,"Não Controlável","darkorange"),
                   (axes6a[2],A_no6,B_no6,C_no6,D_no6,"Não Observável","crimson")]
        for ax,Av,Bv,Cv,Dv,titulo,cor in configs6:
            sv=control.ss(Av,Bv,Cv,Dv)
            t_r,y_r,x_r=control.forced_response(sv,T=t6,U=u6,X0=np.array([0.5,-0.5]),return_x=True)
            ax.plot(t_r,x_r[0],"b-",lw=1.8,label=r"$x_1$")
            ax.plot(t_r,x_r[1],"r--",lw=1.8,label=r"$x_2$")
            ax.plot(t_r,y_r,"k-",lw=2.5,label=r"$y$",alpha=0.7)
            Wc6=control.ctrb(Av,Bv); Wo6=control.obsv(Av,Cv); n6=Av.shape[0]
            rc6=np.linalg.matrix_rank(Wc6); ro6=np.linalg.matrix_rank(Wo6)
            ax.set_title(f"{titulo}\nrank($W_c$)={rc6}/{n6}  rank($W_o$)={ro6}/{n6}",fontsize=8.5,color=cor)
            ax.set_xlabel("t (s)",fontsize=7); ax.legend(fontsize=7)
            ax.spines[["right","top"]].set_visible(False)
        fig6a.suptitle("Resposta de sistemas controláveis/observáveis vs. não",fontsize=9)
        plt.tight_layout()
        show_fig(fig6a, 0.92)
        
        for nome, Av, Bv, Cv, Dv in [
            ("Não Controlável (polo s=−2 oculto)", A_nc6,B_nc6,C_nc6,D_nc6),
            ("Não Observável (polo s=−2 oculto)",  A_no6,B_no6,C_no6,D_no6)]:
            sys_v=control.ss(Av,Bv,Cv,Dv); tf_v=control.ss2tf(sys_v)
            n_v=np.array(tf_v.num[0][0]); d_v=np.array(tf_v.den[0][0])
            Wc6v=control.ctrb(Av,Bv); Wo6v=control.obsv(Av,Cv); nv=Av.shape[0]
            st.info(f"**{nome}:** $H(s)=\\dfrac{{{poly_str(n_v)}}}{{{poly_str(d_v)}}}$ — "
                    f"polo $s=-2$ cancelado · "
                    f"rank$(W_c)={np.linalg.matrix_rank(Wc6v)}/{nv}$ · "
                    f"rank$(W_o)={np.linalg.matrix_rank(Wo6v)}/{nv}$")
        
        st.divider()
        
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7 — EXPLORADORES INTERATIVOS
        # ═══════════════════════════════════════════════════════════════════════════════
        st.header("7. Explorador Interativo de Espaço de Estados")
        
        tabs7 = st.tabs(["7.1 Explorador SS",
                          "7.2 Conversor FT → SS",
                          "7.3 Conversor SS → FT"])
        
        # ── 7.1 Explorador SS ────────────────────────────────────────────────────────
        with tabs7[0]:
            st.markdown(r"""
            Insira as matrizes $A$, $B$, $C$, $D$ e visualize resposta, trajetória e diagnóstico.
        
            **Exemplos prontos:**
        
            | Sistema | A | B | C | D |
            |:---|:---|:---|:---|:---|
            | 2ª ordem estável | `-1,1;-2,-3` | `0;1` | `1,0` | `0` |
            | Oscilador amortecido | `0,1;-9,-1.2` | `0;9` | `1,0` | `0` |
            | 3ª ordem | `0,1,0;0,0,1;-5,-7,-3` | `0;0;1` | `2,2,1` | `0` |
            | Instável | `1,1;-2,-1` | `0;1` | `1,0` | `0` |
            """)
        
            c71a, c71b = st.columns([1, 2])
            with c71a:
                A71_str = st.text_input("Matriz $A$", value="-1,1;-2,-3", key="A71")
                B71_str = st.text_input("Matriz $B$", value="0;1",        key="B71")
                C71_str = st.text_input("Matriz $C$", value="1,0",        key="C71")
                D71_str = st.text_input("Matriz $D$", value="0",          key="D71")
                t71_max = st.slider("Tempo de simulação (s)", 2.0, 30.0, 8.0, 1.0, key="t71")
                x0_71_str = st.text_input("CI $\\mathbf{x}(0)$ (separados por vírgula)", value="0,0", key="x071")
        
            with c71b:
                try:
                    A71 = parse_matrix(A71_str)
                    B71 = parse_matrix(B71_str)
                    C71 = (parse_matrix(C71_str) if ";" in C71_str
                           else np.array([[float(v.strip()) for v in C71_str.split(",")]]))
                    D71 = (np.array([[float(D71_str.strip())]]) if ";" not in D71_str and "," not in D71_str
                           else parse_matrix(D71_str))
                    n71 = A71.shape[0]
                    x0_71 = np.array([float(v.strip()) for v in x0_71_str.split(",")])
                    if len(x0_71) != n71:
                        x0_71 = np.zeros(n71)
        
                    assert A71.shape == (n71,n71), "A deve ser quadrada"
                    assert B71.shape[0] == n71, "B: n linhas"
                    assert C71.shape[1] == n71, "C: n colunas"
        
                    sys71 = control.ss(A71, B71, C71, D71)
                    t71 = np.linspace(0, t71_max, 1000)
                    u71 = np.ones_like(t71)
        
                    t_stp71,y_stp71,x_stp71 = control.forced_response(sys71,T=t71,U=u71,X0=x0_71,return_x=True)
                    t_imp71,y_imp71 = control.impulse_response(sys71,T=t71)
        
                    # Plano de estados (só n==2)
                    n_cols71 = 3 if n71 == 2 else 2
                    sub_ttl71 = ["Resposta ao Degrau","Resposta ao Impulso"]
                    if n71 == 2: sub_ttl71 += ["Plano de Estados"]
        
                    fig71 = make_subplots(rows=1,cols=n_cols71,subplot_titles=sub_ttl71,
                                           horizontal_spacing=0.10)
                    fig71.add_trace(go.Scatter(x=t_stp71,y=y_stp71,mode="lines",
                        line=dict(color="#1f77b4",width=2.5),name="Degrau"),row=1,col=1)
                    fig71.add_trace(go.Scatter(x=t_imp71,y=y_imp71,mode="lines",
                        line=dict(color="#d62728",width=2.5),name="Impulso"),row=1,col=2)
                    if n71 == 2:
                        fig71.add_trace(go.Scatter(x=x_stp71[0],y=x_stp71[1],mode="lines",
                            line=dict(color="#2ca02c",width=2),name="Trajetória"),row=1,col=3)
                        fig71.add_trace(go.Scatter(x=[x0_71[0]],y=[x0_71[1]],mode="markers",
                            marker=dict(size=12,color="green",symbol="circle"),name="x(0)"),row=1,col=3)
                        fig71.update_xaxes(title_text="x₁",row=1,col=3)
                        fig71.update_yaxes(title_text="x₂",row=1,col=3)
                    fig71.update_xaxes(title_text="t (s)",row=1,col=1)
                    fig71.update_xaxes(title_text="t (s)",row=1,col=2)
                    fig71.update_yaxes(title_text="y(t)",row=1,col=1)
                    fig71.update_yaxes(title_text="y(t)",row=1,col=2)
                    fig71.update_layout(height=340,margin=dict(t=40,b=40,l=50,r=10),
                                         template="plotly_white",
                                         legend=dict(orientation="h",y=1.10))
                    st.plotly_chart(fig71, use_container_width=True)
                    render_diagnostics(A71, B71, C71, D71)
        
                except AssertionError as e:
                    st.error(str(e))
                except Exception as ex:
                    st.error(f"Erro: {ex}")
        
        # ── 7.2 Conversor FT → SS ─────────────────────────────────────────────────────
        with tabs7[1]:
            st.markdown(r"""
            Insira $H(s) = N(s)/D(s)$ (coeficientes em **ordem decrescente** de $s$).
        
            | Exemplo | Numerador | Denominador |
            |:---|:---|:---|
            | $s(s+2)/[(s+1)(s^2+2s+5)]$ | `1, 2, 0` | `1, 3, 7, 5` |
            | $1/(s^2+2s+5)$ | `1` | `1, 2, 5` |
            | $(s+2)/[s(s+1)(s+3)]$ | `1, 2` | `1, 4, 3, 0` |
            | $(2s+6)/(s^2+3s+2)` | `2, 6` | `1, 3, 2` |
            """)
        
            c72a, c72b = st.columns([1, 2])
            with c72a:
                num72_str = st.text_input("Numerador $N(s)$", value="1, 2, 0", key="num72")
                den72_str = st.text_input("Denominador $D(s)$", value="1, 3, 7, 5", key="den72")
                t72_max = st.slider("Tempo de simulação (s)", 2.0, 30.0, 8.0, 1.0, key="t72")
        
            with c72b:
                try:
                    num72=[float(x.strip()) for x in num72_str.split(",") if x.strip()]
                    den72=[float(x.strip()) for x in den72_str.split(",") if x.strip()]
                    assert len(den72)>len(num72), "Grau Den deve ser > Num"
        
                    sys_tf72=control.tf(num72,den72)
                    sys_ss72=control.tf2ss(sys_tf72)
                    A72=np.array(sys_ss72.A); B72=np.array(sys_ss72.B)
                    C72=np.array(sys_ss72.C); D72=np.array(sys_ss72.D)
                    n72=A72.shape[0]
        
                    st.latex(r"\dot{\mathbf{x}} = " + mat_str(A72) +
                             r"\,\mathbf{x} + " + mat_str(B72) + r"\,\mathbf{u}")
                    st.latex(r"\mathbf{y} = " + mat_str(C72) +
                             r"\,\mathbf{x} + " + mat_str(D72) + r"\,\mathbf{u}")
        
                    sys72=control.ss(A72,B72,C72,D72)
                    t72=np.linspace(0,t72_max,1000); u72=np.ones_like(t72)
                    t_stp72,y_stp72=control.forced_response(sys72,T=t72,U=u72,return_x=False)[:2]
                    t_imp72,y_imp72=control.impulse_response(sys72,T=t72)
        
                    fig72=make_subplots(rows=1,cols=2,
                                         subplot_titles=["Resposta ao Degrau","Resposta ao Impulso"])
                    fig72.add_trace(go.Scatter(x=t_stp72,y=y_stp72,mode="lines",
                        line=dict(color="#1f77b4",width=2.5),name="Degrau"),row=1,col=1)
                    fig72.add_trace(go.Scatter(x=t_imp72,y=y_imp72,mode="lines",
                        line=dict(color="#d62728",width=2.5),name="Impulso"),row=1,col=2)
                    fig72.update_xaxes(title_text="t (s)")
                    fig72.update_yaxes(title_text="y(t)",row=1,col=1)
                    fig72.update_yaxes(title_text="y(t)",row=1,col=2)
                    fig72.update_layout(height=300,margin=dict(t=40,b=30,l=50,r=10),
                                         template="plotly_white",legend=dict(orientation="h",y=1.1))
                    st.plotly_chart(fig72, use_container_width=True)
                    render_diagnostics(A72,B72,C72,D72)
        
                except AssertionError as e:
                    st.error(str(e))
                except Exception as ex:
                    st.error(f"Erro: {ex}")
        
        # ── 7.3 Conversor SS → FT ─────────────────────────────────────────────────────
        with tabs7[2]:
            st.markdown(r"""
            Insira $A$, $B$, $C$, $D$ e obtenha $H(s) = C(sI-A)^{-1}B+D$.
        
            | Exemplo | A | B | C | D |
            |:---|:---|:---|:---|:---|
            | 2ª ordem | `-1,1;-2,-3` | `0;1` | `1,0` | `0` |
            | Oscilatório | `0,1;-9,-1.2` | `0;9` | `1,0` | `0` |
            | 3ª ordem | `0,1,0;0,0,1;-5,-7,-3` | `0;0;1` | `2,2,1` | `0` |
            """)
        
            c73a, c73b = st.columns([1, 2])
            with c73a:
                A73_str = st.text_input("Matriz $A$", value="-1,1;-2,-3", key="A73")
                B73_str = st.text_input("Matriz $B$", value="0;1",        key="B73")
                C73_str = st.text_input("Matriz $C$", value="1,0",        key="C73")
                D73_str = st.text_input("Matriz $D$", value="0",          key="D73")
        
            with c73b:
                try:
                    A73 = parse_matrix(A73_str)
                    B73 = parse_matrix(B73_str)
                    C73 = (parse_matrix(C73_str) if ";" in C73_str
                           else np.array([[float(v.strip()) for v in C73_str.split(",")]]))
                    D73 = (np.array([[float(D73_str.strip())]]) if ";" not in D73_str and "," not in D73_str
                           else parse_matrix(D73_str))
                    n73 = A73.shape[0]
                    assert A73.shape==(n73,n73),"A quadrada"
                    assert B73.shape[0]==n73,"B: n linhas"
                    assert C73.shape[1]==n73,"C: n colunas"
        
                    sys_ss73=control.ss(A73,B73,C73,D73)
                    sys_tf73=control.ss2tf(sys_ss73)
                    num73=np.array(sys_tf73.num[0][0]); den73=np.array(sys_tf73.den[0][0])
                    tol73=1e-10*max(np.abs(den73))
                    num73[np.abs(num73)<tol73]=0.0
        
                    st.latex(r"H(s) = C(sI-A)^{-1}B + D = \frac{" +
                             poly_str(num73) + r"}{" + poly_str(den73) + r"}")
                    zeros73=np.roots(num73)
                    if len(zeros73)>0:
                        z_str=", ".join(f"{z.real:.4f}" if abs(z.imag)<1e-8
                                        else f"{z.real:.4f}{z.imag:+.4f}j"
                                        for z in sorted(zeros73,key=lambda z:z.real))
                        st.latex(r"\text{Zeros de } H(s): \quad " + z_str)
                    render_diagnostics(A73,B73,C73,D73)
        
                except AssertionError as e:
                    st.error(str(e))
                except Exception as ex:
                    st.error(f"Erro: {ex}")
        
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
            "🧮 Análise de Sistemas no Espaço de Estados &nbsp;·&nbsp; 🎛️ SINTONIA — Sistemas de Controle<br>"
            "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
            " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
            "</div>",
            unsafe_allow_html=True,
        )
        
            # TODO: todo o restante do código original
            # exatamente como está no arquivo
