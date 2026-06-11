"""
🔁 Critério de Nyquist
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
   
def run():
   
      warnings.filterwarnings("ignore")
      
      # ── Configuração da Página ────────────────────────────────────────────────────
      st.set_page_config(
          page_title="Critério de Nyquist",
          page_icon="🔁",
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
              f'<img src="data:image/png;base64,{b64}" '
              f'style="width:100%;height:auto;display:block;"/>'
              f'</div></div>', unsafe_allow_html=True)
      
      # ── Helpers matplotlib ────────────────────────────────────────────────────────
      def polo_x(ax, x, y, cor="#d62728", ms=11):
          ax.plot(x, y, "x", color=cor, ms=ms, mew=2.5, zorder=5)
      
      def zero_o(ax, x, y, cor="#1f77b4", ms=10):
          ax.plot(x, y, "o", color=cor, ms=ms, mfc="white", mew=2.0, zorder=5)
      
      def plano_s_base(ax, xlim=(-6,2), ylim=(-5,5), titulo="plano s"):
          ax.axhline(0,color="k",lw=0.8); ax.axvline(0,color="k",lw=0.8)
          ax.fill_betweenx([ylim[0],ylim[1]],xlim[0],0,alpha=0.06,color="seagreen")
          ax.fill_betweenx([ylim[0],ylim[1]],0,xlim[1],alpha=0.06,color="crimson")
          ax.set_xlim(xlim); ax.set_ylim(ylim)
          ax.set_xlabel(r"$\sigma$",fontsize=8); ax.set_ylabel(r"$j\omega$",fontsize=8)
          ax.set_title(titulo,fontsize=8.5)
          ax.spines[["right","top"]].set_visible(False)
      
      def estilo(ax, xlabel="Re", ylabel="Im"):
          ax.set_xlabel(xlabel,fontsize=8); ax.set_ylabel(ylabel,fontsize=8)
          ax.spines[["right","top"]].set_visible(False)
      
      # ── Helpers Nyquist ───────────────────────────────────────────────────────────
      def nyquist_data(num, den, w_min=1e-3, w_max=1e3, n=4000):
          w = np.logspace(np.log10(w_min), np.log10(w_max), n)
          _, H = signal.freqs(num, den, worN=w)
          return H.real, H.imag, w
      
      def plot_nyquist_ax(ax, num, den, color="#1f77b4", label="",
                          w_min=1e-2, w_max=1e3, lw=2.0):
          Re, Im, w = nyquist_data(num, den, w_min, w_max)
          ax.plot(Re,  Im,  color=color, lw=lw, label=label or r"$\omega>0$")
          ax.plot(Re, -Im,  color=color, lw=lw, ls="--", alpha=0.5,
                  label=r"$\omega<0$ (conj.)")
          ax.plot(-1, 0, "r*", ms=14, zorder=6, label="Ponto crítico (−1,0)")
          ax.axhline(0,color="k",lw=0.5,ls="--",alpha=0.4)
          ax.axvline(0,color="k",lw=0.5,ls="--",alpha=0.4)
          mid = len(w)//3
          ax.annotate("", xy=(Re[mid+1],Im[mid+1]), xytext=(Re[mid],Im[mid]),
              arrowprops=dict(arrowstyle="->",color=color,lw=1.8))
          estilo(ax)
          return Re, Im, w
      
      def nyquist_margins(num, den, w_min=1e-3, w_max=1e3):
          """Retorna (phi_m, w_gc, gm_db, re_pc) para um sistema."""
          w = np.logspace(np.log10(w_min), np.log10(w_max), 8000)
          _, H = signal.freqs(num, den, worN=w)
          mag = np.abs(H); Re = H.real; Im = H.imag
          # Cruzamento círculo unitário → φm
          idx_gc = np.argmin(np.abs(mag - 1.0))
          pm = 180 + np.degrees(np.angle(H[idx_gc]))
          w_gc = float(w[idx_gc])
          # Cruzamento eixo real negativo → Gm
          sign_im = np.sign(Im)
          crosses = [i for i in range(1,len(Im))
                     if sign_im[i-1]*sign_im[i]<0 and Re[i]<0]
          re_pc = Re[crosses[0]] if crosses else None
          gm_db = float(-20*np.log10(abs(re_pc))) if re_pc is not None else float("inf")
          return pm, w_gc, gm_db, re_pc
      
      def count_N(Re, Im):
          """Conta N = envolvimentos horários ao redor de (-1,0)."""
          sign_im = np.sign(Im)
          N_cw=0; N_ccw=0
          for i in range(1,len(Im)):
              if sign_im[i-1]*sign_im[i]<0 and min(Re[i-1],Re[i])<-1:
                  if sign_im[i-1]>0: N_cw+=1
                  else:               N_ccw+=1
          return N_cw - N_ccw
      
      # ── LGR helper ────────────────────────────────────────────────────────────────
      def calc_lgr(num, den, k_max=200.0, n_k=1500):
          k_arr = np.unique(np.concatenate([
              np.linspace(0, 0.01, 100),
              np.geomspace(0.01, k_max, n_k)
          ]))
          n_p = len(den)-1
          pd = np.array(den,float); pn = np.pad(np.array(num,float),(n_p-len(num)+1,0))
          ramos=[[] for _ in range(n_p)]; prev=None
          for k in k_arr:
              roots=np.roots(pd + k*pn)
              if prev is None:
                  ordered=sorted(roots,key=lambda z:(z.real,z.imag))
              else:
                  ordered=list(prev); rem=list(roots)
                  for ri in range(n_p):
                      b=int(np.argmin([abs(ordered[ri]-r) for r in rem]))
                      ordered[ri]=rem.pop(b)
              for ri in range(n_p): ramos[ri].append(ordered[ri])
              prev=ordered
          return ramos, k_arr
      
      CORES_LGR=["#1f77b4","#d62728","#2ca02c","#9467bd","#e07000","#17becf"]
      PHASE_COLORS={-90:"#ff7f0e",-180:"#d62728",-270:"#9467bd",-360:"#8c564b"}
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # CABEÇALHO
      # ═══════════════════════════════════════════════════════════════════════════════
      st.title("🔁 Critério de Nyquist")
      st.subheader("Diagramas de Nyquist")
      st.caption("🎛️ SINTONIA · Sistemas de Controle · 👤 Marcus V A Fernandes · ✉️ marcus.fernandes@ifrn.edu.br")
      st.markdown("---")
      
      with st.expander("📋 Índice — clique para expandir", expanded=False):
          st.markdown(r"""
      **[1. Introdução e Motivação](#1-introdu-o-e-motiva-o)**
      - 1.1 O problema de estabilidade em malha fechada
      - 1.2 Comparação: Routh-Hurwitz × Nyquist × LGR
      - 1.3 Vantagem prática: resposta em frequência experimental
      
      **[2. Mapeamento de Contornos e Princípio do Argumento](#2-mapeamento-de-contornos-e-princ-pio-do-argumento)**
      - 2.1 Teorema do Princípio do Argumento: $N_{ah} = Z - P$
      - 2.2 Contribuições de zeros e polos
      - 2.3 Conexão com estabilidade: zeros de $F=1+GH$ = polos de MF
      
      **[3. Contorno de Nyquist no Plano $s$](#3-contorno-de-nyquist-no-plano-s)**
      - 3.1 Definição: percorre todo o SPD no sentido horário
      - 3.2 Por que avaliar no eixo imaginário: $G(j\omega)H(j\omega)$
      - 3.3 Diagrama completo: curva principal, conjugada e fechamento
      - 3.4 Comportamento por tipo do sistema
      
      **[4. Critério de Nyquist — Enunciado e Procedimento](#4-crit-rio-de-nyquist-enunciado-e-procedimento)**
      - 4.1 Enunciado: $Z = N + P = 0$ para estabilidade
      - 4.2 Tabela de casos práticos
      - 4.3 Procedimento passo a passo
      
      **[5. Exemplos de Diagramas de Nyquist](#5-exemplos-de-diagramas-de-nyquist)**
      - 5.1 Exemplo 1: $500/[(s+1)(s+3)(s+10)]$ — tipo 0, $P=0$
      - 5.2 Efeito do ganho $K$ e ganho crítico $K_{crit}$
      - 5.3 Exemplo 2: sistema com $P=2$ polos no SPD
      
      **[6. Desvio em Polos sobre o Eixo Imaginário](#6-desvio-em-polos-sobre-o-eixo-imagin-rio)**
      - 6.1 Problema: polo sobre o eixo Imaginário → $\lvert GH\rvert \to \infty$
      - 6.2 Solução: semicírculo de raio $\varepsilon \to 0$
      - 6.3 Efeito por tipo de polo: simples, duplo, par $\pm j\omega_0$
      - 6.4 Exemplo com integrador: $1/[s(s+3)(s+5)]$
      
      **[7. Margens de Ganho e Fase pelo Critério de Nyquist](#7-margens-de-ganho-e-fase-pelo-crit-rio-de-nyquist)**
      - 7.1 Margem de fase $\phi_m$: ângulo até $(-1, j0)$ no círculo unitário
      - 7.2 Margem de ganho $G_m$: fator de escala até atingir $(-1, j0)$
      - 7.3 Interpretação geométrica
      - 7.4 🎛️ Comparação Nyquist × Bode × LGR
      
      **[8. Explorador Interativo de Nyquist](#8-explorador-interativo-de-nyquist)**
      - 🎛️ Inserir $G(s)H(s)=N(s)/D(s)$, ajustar $K$ e $\omega_{max}$
      - Diagnóstico automático: $P$, $N$, $Z$, margens, estabilidade
      
      **[9. Referências](#9-refer-ncias)**
      """)
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 1
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("1. Introdução e Motivação")
      
      st.markdown(r"""
      ### 1.1 O problema de estabilidade em malha fechada
      
      $$T(s) = \frac{G(s)}{1 + G(s)H(s)}, \qquad F(s) = 1 + G(s)H(s)$$
      
      | Singularidade de $F(s)$ | Relação com o sistema |
      |---|---|
      | **Zeros** de $F(s)$: $F(s_i)=0$ | **Polos de malha fechada** — determinam estabilidade |
      | **Polos** de $F(s)$: denom. de $F$ | **Polos de malha aberta** $G(s)H(s)$ — conhecidos |
      
      O **Critério de Nyquist** responde, sem calcular raízes explicitamente:
      
      > *Quantos zeros de $F(s)=1+G(s)H(s)$ estão no semiplano direito?*
      
      usando apenas a **resposta em frequência de malha aberta** $G(j\omega)H(j\omega)$.
      
      ### 1.2 Comparação com outros métodos
      
      | Critério | Informação obtida | Entrada necessária |
      |---|---|---|
      | **Routh-Hurwitz** | Nº de polos no SPD | Polinômio característico |
      | **Critério de Nyquist** | Nº de polos no SPD + **margens** | Resposta em frequência de MA |
      | **LGR** | Trajetória dos polos vs. ganho | Polos/zeros de MA |
      
      ### 1.3 Vantagem prática
      
      $G(j\omega)H(j\omega)$ pode ser obtido **experimentalmente** medindo a resposta em frequência de
      malha aberta — sem necessidade de modelo matemático explícito.
      
      > **Atenção:** a medição direta pressupõe malha aberta **estável** ($P=0$). Para $P>0$ é necessário
      > garantir estabilidade do arranjo de medição.
      """)
      
      # Diagrama de blocos
      fig1a, ax1a = plt.subplots(figsize=(9.0, 3.4))
      ax1a.set_xlim(0,12); ax1a.set_ylim(0,5); ax1a.axis("off")
      R_s=0.30; xS1=2.5; yS1=2.5
      xG1=6.2; yG1=2.5; wG1=2.2; hG1=0.9
      xH1=6.2; yH1=1.0; wH1=2.2; hH1=0.9; xBif1=9.5
      
      def _bloco(ax, cx, cy, w, h, txt, fs=10):
          ax.add_patch(mpatches.FancyBboxPatch((cx-w/2,cy-h/2),w,h,
              boxstyle="square,pad=0.0",fc="white",ec="#333",lw=1.6,zorder=4))
          ax.text(cx,cy,txt,ha="center",va="center",fontsize=fs,zorder=5)
      
      def _seta(ax,x1,y1,x2,y2):
          ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
              arrowprops=dict(arrowstyle="->",color="#333",lw=1.6,mutation_scale=14),zorder=3)
      
      def _linha(ax,x1,y1,x2,y2):
          ax.plot([x1,x2],[y1,y2],"-",color="#333",lw=1.6,zorder=3)
      
      ax1a.add_patch(plt.Circle((xS1,yS1),R_s,fc="white",ec="#333",lw=1.6,zorder=4))
      _bloco(ax1a,xG1,yG1,wG1,hG1,r"$G(s)$")
      _bloco(ax1a,xH1,yH1,wH1,hH1,r"$H(s)$")
      _seta(ax1a,0.8,yS1,xS1-R_s,yS1); _seta(ax1a,xS1+R_s,yS1,xG1-wG1/2,yG1)
      _linha(ax1a,xG1+wG1/2,yG1,xBif1,yG1); _seta(ax1a,xBif1,yG1,11.0,yG1)
      ax1a.plot(xBif1,yG1,"o",color="#333",ms=5,zorder=5)
      _linha(ax1a,xBif1,yG1,xBif1,yH1); _seta(ax1a,xBif1,yH1,xH1+wH1/2,yH1)
      _linha(ax1a,xH1-wH1/2,yH1,xS1,yH1); _seta(ax1a,xS1,yH1,xS1,yS1-R_s)
      ax1a.text(xS1-R_s-0.06,yS1+0.28,"+",fontsize=13,color="#333",ha="right")
      ax1a.text(xS1+0.10,yS1-R_s-0.22,"−",fontsize=14,color="#333",ha="center")
      ax1a.text(0.9,yS1+0.22,r"$R(s)$",fontsize=9)
      ax1a.text((xS1+R_s+xG1-wG1/2)/2,yS1+0.22,r"$E(s)$",fontsize=9,ha="center")
      ax1a.text(10.2,yG1+0.22,r"$C(s)$",fontsize=9)
      ax1a.set_title(r"$T(s)=G(s)/[1+G(s)H(s)]$ — polinômio característico: $1+G(s)H(s)=0$",
                     fontsize=9, pad=6)
      plt.tight_layout()
      show_fig(fig1a, 0.78)
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 2
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("2. Mapeamento de Contornos e Princípio do Argumento")
      
      st.markdown(r"""
      ### 2.1 Teorema do Princípio do Argumento
      
      $$\boxed{N_{ah} = Z - P}$$
      
      onde $N_{ah}$ é o número de envolvimentos **anti-horários** de $B = F(A)$ ao redor da origem,
      $Z$ = zeros de $F$ dentro de $A$, $P$ = polos de $F$ dentro de $A$.
      
      > **Convenção de controle (Nyquist):** contorno percorrido no sentido **horário** (SPD fica
      > à esquerda). Nessa convenção $N$ conta envolvimentos **horários** e $N = Z - P$.
      
      ### 2.2 Contribuições de cada singularidade
      
      | Singularidade | Posição relativa a $A$ | Contribuição ao ângulo total |
      |---|---|---|
      | **Zero** de $F$ em $z_1$ | **Dentro** | +360° (envolvimento anti-horário) |
      | **Zero** de $F$ em $z_1$ | **Fora** | 0° (variação líquida nula) |
      | **Polo** de $F$ em $p_1$ | **Dentro** | −360° (envolvimento horário) |
      | **Polo** de $F$ em $p_1$ | **Fora** | 0° |
      
      ### 2.3 Conexão com estabilidade
      
      Para $F(s) = 1 + G(s)H(s)$, usando o contorno de Nyquist (envolve todo o SPD):
      
      $$Z = N + P$$
      
      Como $G(s)H(s) = F(s)-1$, o contorno mapeado de $G(s)H(s)$ envolve o **ponto $(-1, j0)$**
      o mesmo número $N$ de vezes que $F(s)$ envolve a origem.
      
      > **Chave do método:** contar envolvimentos de $G(j\omega)H(j\omega)$ ao redor de $(-1, j0)$
      > equivale a contar zeros de $1+G(s)H(s)$ no SPD.
      """)
      
      fig2a, axes2a = plt.subplots(1, 3, figsize=(11.5, 4.0))
      configs2 = [
          dict(zeros=[complex(-0.5, 0.5)], polos=[],
               N=1,  desc="N = Z−P = 1−0 = +1\n(1 envolvimento\nanti-horário)"),
          dict(zeros=[], polos=[complex(-0.5, 0.5)],
               N=-1, desc="N = Z−P = 0−1 = −1\n(1 envolvimento\nhorário)"),
          dict(zeros=[complex(-0.5, 0.5)], polos=[complex(-0.3, 0.2)],
               N=0,  desc="N = Z−P = 1−1 = 0\n(não envolve\na origem)"),
      ]
      titulos2 = ["Zero dentro do contorno","Polo dentro do contorno","Zero e polo dentro"]
      for ax, cfg, ttl in zip(axes2a, configs2, titulos2):
          th = np.linspace(0, 2*np.pi, 300)
          cx2,cy2,rx2,ry2 = -0.5, 0.5, 0.8, 0.7
          ax.plot(cx2+rx2*np.cos(th), cy2+ry2*np.sin(th), "k-", lw=1.8, label="Contorno A")
          ax.annotate("", xy=(cx2+rx2*np.cos(0.3),cy2+ry2*np.sin(0.3)),
              xytext=(cx2+rx2*np.cos(0.1),cy2+ry2*np.sin(0.1)),
              arrowprops=dict(arrowstyle="->",color="k",lw=1.5))
          for z in cfg["zeros"]: zero_o(ax, z.real, z.imag, ms=10)
          for p in cfg["polos"]: polo_x(ax, p.real, p.imag, ms=10)
          ax.axhline(0,color="k",lw=0.5); ax.axvline(0,color="k",lw=0.5)
          ax.set_xlim(-1.6, 0.6); ax.set_ylim(-0.4, 1.6); ax.set_aspect("equal")
          ax.set_title(ttl, fontsize=8.5)
          col2 = "#2ca02c" if cfg["N"]>=0 else "#d62728"
          ax.text(-1.5, -0.25, cfg["desc"], fontsize=8, color=col2,
                  bbox=dict(boxstyle="round,pad=0.3",fc="#f8f8f8",ec="gray",alpha=0.8))
          ax.text(-1.5, 1.45, "plano s", fontsize=7.5, style="italic", color="gray")
          ax.spines[["right","top"]].set_visible(False)
      fig2a.suptitle("Princípio do Argumento: N = Z − P", fontsize=9)
      plt.tight_layout()
      show_fig(fig2a, 0.88)
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 3
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("3. Contorno de Nyquist no Plano $s$")
      
      st.markdown(r"""
      ### 3.1 Definição
      
      O contorno de Nyquist envolve todo o SPD percorrido no sentido **horário**:
      
      $$A:\;\underbrace{s=j\omega,\;\omega:0^+\to+\infty}_{\text{eixo Im positivo}}\;\cup\;\underbrace{s=Re^{j\theta},\;\theta:+90°\to-90°\;(R\to\infty)}_{\text{semicírculo}}\;\cup\;\underbrace{s=j\omega,\;\omega:-\infty\to0^-}_{\text{eixo Im negativo}}$$
      
      ### 3.2 Por que avaliar no eixo imaginário?
      
      $G(s)H(s)$ avaliada em $s=j\omega$ produz $G(j\omega)H(j\omega)$ — a **resposta em frequência de MA**,
      obtível por medição experimental.
      
      Para sistemas estritos próprios, a imagem do semicírculo $R\to\infty$ colapsa na **origem** do
      plano GH, sem contribuir para envolvimentos ao redor de $(-1, j0)$.
      
      ### 3.3 Diagrama de Nyquist completo
      
      | Trecho | Origem | Representação |
      |---|---|---|
      | Curva principal | $G(j\omega)H(j\omega)$, $\omega:0^+\to+\infty$ | Traçada explicitamente |
      | Curva conjugada | $[G(j\omega)H(j\omega)]^*$, $\omega:0^-\to-\infty$ | Espelho no eixo real |
      | Fechamento | Semicírculo no infinito (ou arco de desvio) | Geralmente na origem |
      
      ### 3.4 Comportamento por tipo do sistema
      
      | Tipo | $\omega\to 0$ | $\omega\to\infty$ |
      |---|---|---|
      | **0** | $GH \to$ cte finita | $GH \to 0$ |
      | **1** (1 integrador) | $GH \to \infty\angle{-90°}$ | $GH \to 0$ — exige desvio |
      | **2** (2 integradores) | $GH \to \infty\angle{-180°}$ | $GH \to 0$ |
      """)
      
      fig3a, ax3a = plt.subplots(figsize=(5.0, 6.0))
      R_c3=4.0; w_c3=np.linspace(0.01, R_c3*0.95, 300)
      ax3a.plot([0]*len(w_c3),  w_c3, "b-", lw=2.5)
      ax3a.plot([0]*len(w_c3), -w_c3, "b-", lw=2.5)
      th3 = np.linspace(np.pi/2, -np.pi/2, 300)
      ax3a.plot(R_c3*np.cos(th3), R_c3*np.sin(th3), "b-", lw=2.5)
      for (x1,y1,x2,y2) in [
          (0,1.0,0,1.5),(0,-1.0,0,-1.5),
          (R_c3*np.cos(0.3),R_c3*np.sin(0.3),R_c3*np.cos(0.2),R_c3*np.sin(0.2))]:
          ax3a.annotate("",xy=(x2,y2),xytext=(x1,y1),
              arrowprops=dict(arrowstyle="->",color="blue",lw=2.0))
      th_f=np.linspace(np.pi/2,-np.pi/2,200)
      ax3a.fill(np.concatenate([[0]*200, R_c3*np.cos(th_f[::-1])]),
                np.concatenate([np.linspace(-R_c3,R_c3,200), R_c3*np.sin(th_f[::-1])]),
                alpha=0.07, color="crimson")
      ax3a.text(0.15, 3.5, r"$j\omega:0^+\to+\infty$", fontsize=8.5, color="blue")
      ax3a.text(0.15,-3.8, r"$j\omega:-\infty\to0^-$", fontsize=8.5, color="blue")
      ax3a.text(R_c3+0.1, 0.2, r"$R\to\infty$", fontsize=8.5, color="blue")
      ax3a.text(-0.4, 4.3, r"$j\omega$", fontsize=11)
      ax3a.text(4.6, 0.1, r"$\sigma$", fontsize=11)
      ax3a.text(1.8, 0.3, "SPD\n(instável)", fontsize=8.5, color="crimson", ha="center")
      ax3a.axhline(0,color="k",lw=0.8); ax3a.axvline(0,color="k",lw=0.8)
      ax3a.set_xlim(-4.8,5.2); ax3a.set_ylim(-5.0,5.0)
      ax3a.set_aspect("equal"); ax3a.axis("off")
      ax3a.set_title("Contorno de Nyquist (sentido horário)", fontsize=9)
      plt.tight_layout()
      show_fig(fig3a, 0.38)
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 4
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("4. Critério de Nyquist — Enunciado e Procedimento")
      
      st.markdown(r"""
      ### 4.1 Enunciado
      
      Seja $G(s)H(s)$ com $P$ polos no SPD. O sistema de malha fechada é **estável** se e somente se:
      
      $$\boxed{Z = N + P = 0}$$
      
      onde $N$ é o número de envolvimentos **horários** do diagrama de Nyquist ao redor de $(-1, j0)$.
      
      **Condição equivalente:** o diagrama deve dar exatamente $P$ envolvimentos **anti-horários**
      ao redor de $(-1, j0)$.
      
      ### 4.2 Casos práticos
      
      | $P$ | $N$ | $Z=N+P$ | Estabilidade | Interpretação |
      |---|---|---|---|---|
      | 0 | 0 | 0 | ✅ Estável | Curva não envolve $(-1,j0)$ |
      | 0 | 1 | 1 | ❌ Instável | 1 envolvimento horário |
      | 0 | −1 | −1 | — Impossível | $Z < 0$: erro de traçado |
      | 1 | −1 | 0 | ✅ Estável | 1 envolvimento anti-horário compensa $P=1$ |
      | 2 | −2 | 0 | ✅ Estável | 2 envolvimentos anti-horários compensam $P=2$ |
      
      > **Por que $Z=-1$ é impossível?** $Z$ conta polos de MF no SPD — inteiro não-negativo.
      > $Z<0$ indica erro no traçado ou na contagem de $N$.
      
      ### 4.3 Procedimento passo a passo
      
      1. **Identificar** os polos de MA no SPD → $P$; verificar polos no eixo Im (requer desvio)
      2. **Traçar** $G(j\omega)H(j\omega)$ para $\omega\in(0,+\infty)$; adicionar arcos de fechamento
      3. **Espelhar** em relação ao eixo real para $\omega\in(-\infty,0)$ (conjugado)
      4. **Contar** $N$: cruzamentos do raio $(-\infty,-1]$ no eixo real — +1 (horário), −1 (anti-horário)
      5. **Calcular** $Z=N+P$: $Z=0$ → estável; $Z>0$ → instável com $Z$ polos de MF no SPD
      
      > **Caso $P=0$:** sistema de MF é estável se e somente se a curva **não envolve** $(-1, j0)$.
      """)
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 5
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("5. Exemplos de Diagramas de Nyquist")
      
      st.markdown(r"""
      ### 5.1 Exemplo 1 — $G(s)H(s)=500/[(s+1)(s+3)(s+10)]$
      
      Sistema tipo 0, $P=0$. $G(0)H(0)=500/(1\cdot3\cdot10)=50/3\approx16{,}67$.
      """)
      
      num1_e=[500]; den1_e=np.polymul([1,1],np.polymul([1,3],[1,10]))
      fig5a, axes5a = plt.subplots(1,2,figsize=(9.5,4.5))
      ax_s1=axes5a[0]; ax_n1=axes5a[1]
      plano_s_base(ax_s1,xlim=(-12,2),ylim=(-5,5),titulo="Plano s — polos de MA")
      for p in np.roots(den1_e): polo_x(ax_s1,p.real,p.imag)
      ax_s1.text(-11.5,4.5,"P=0 (todos no SPE)",fontsize=8,color="seagreen")
      Re1_e,Im1_e,w1_e=plot_nyquist_ax(ax_n1,num1_e,den1_e,label="G(jω)H(jω)")
      sign1_im=np.sign(Im1_e)
      cross1=[i for i in range(1,len(Im1_e))
              if sign1_im[i-1]*sign1_im[i]<0 and Re1_e[i]<0]
      if cross1:
          re_pc1=Re1_e[cross1[0]]
          ax_n1.plot(re_pc1,0,"s",color="#ff7f0e",ms=10,zorder=5,label=f"ωpc: Re={re_pc1:.3f}")
      ax_n1.set_title("G(jω)H(jω)=500/[(jω+1)(jω+3)(jω+10)]",fontsize=8.5)
      ax_n1.legend(fontsize=7)
      plt.suptitle("Exemplo 1 — P=0, N=0 → Z=0 (ESTÁVEL)",fontsize=9,fontweight="bold")
      plt.tight_layout()
      show_fig(fig5a, 0.88)
      if cross1:
          st.info(f"$G(0)H(0)={500/(1*3*10):.4f}$ · Cruzamento eixo real: $Re={re_pc1:.4f}$ · $N=0$ → $Z=0$ → **ESTÁVEL**")
      
      st.markdown(r"""
      ### 5.2 Efeito do ganho $K$
      
      Para $G(s)H(s)=K/[(s+1)(s+3)(s+10)]$ com $P=0$:
      
      - $K<K_{crit}$: diagrama não envolve $(-1,j0)$ → $N=0$, $Z=0$ → **estável**
      - $K=K_{crit}\approx574$: passa por $(-1,j0)$ → **marginalmente estável**
      - $K>K_{crit}$: envolve $(-1,j0)$ duas vezes (horário) → $N=2$, $Z=2$ → **instável**
      
      $$K_{crit}=\frac{1}{\lvert G(j\omega_{pc})H(j\omega_{pc})/K\rvert}\approx 574$$
      
      > Ao variar $K$, o diagrama se escala proporcionalmente — como um balão que infla ou desinfla.
      """)
      
      # Cálculo de K_crit
      den_base5=np.polymul([1,1],np.polymul([1,3],[1,10]))
      w_base5=np.logspace(-1,3,3000)
      _,H_base5=signal.freqs([1],den_base5,worN=w_base5)
      sign_ib5=np.sign(H_base5.imag)
      cross_b5=[i for i in range(1,len(H_base5))
                if sign_ib5[i-1]*sign_ib5[i]<0 and H_base5.real[i]<0]
      K_crit5=float(1.0/abs(H_base5[cross_b5[0]])) if cross_b5 else 572.0
      
      st.markdown("### 🎛️ Explorador — Efeito do Ganho $K$")
      st.caption(f"$K_{{crit}} \\approx {K_crit5:.1f}$ · Verde = estável · Laranja = crítico · Vermelho = instável")
      
      c5a, c5b = st.columns([1, 2])
      with c5a:
          K5_sl = st.slider("Ganho $K$", 10.0, float(K_crit5*1.8),
                            200.0, 10.0, key="K5sl")
          ratio5 = K5_sl / K_crit5
          if abs(ratio5 - 1.0) < 0.05:
              st5 = "🟡 MARGINAL"; cor5 = "#ff7f0e"
          elif K5_sl < K_crit5:
              st5 = "🟢 ESTÁVEL"; cor5 = "#2ca02c"
          else:
              st5 = "🔴 INSTÁVEL"; cor5 = "#d62728"
          pm5_e, w_gc5, gm5_db, re_pc5 = nyquist_margins([K5_sl], den_base5)
          st.info(f"**{st5}**\n\n$K={K5_sl:.0f}$ · $K_{{crit}}={K_crit5:.1f}$\n\n"
                  f"$\\phi_m={pm5_e:.1f}°$ · $G_m={gm5_db:.1f}$ dB")
      
      with c5b:
          w5_e = np.logspace(-1, 3, 3000)
          _, H5_e = signal.freqs([K5_sl], den_base5, worN=w5_e)
          th5 = np.linspace(0, 2*np.pi, 200)
          fig_e5 = go.Figure()
          fig_e5.add_trace(go.Scatter(x=H5_e.real, y=H5_e.imag, mode="lines",
              line=dict(color=cor5, width=2.5), name="ω>0"))
          fig_e5.add_trace(go.Scatter(x=H5_e.real, y=-H5_e.imag, mode="lines",
              line=dict(color=cor5, width=2.5, dash="dot"), name="ω<0", showlegend=False))
          fig_e5.add_trace(go.Scatter(x=[-1], y=[0], mode="markers",
              marker=dict(symbol="star", size=16, color="red"), name="(−1,0)"))
          fig_e5.add_trace(go.Scatter(x=np.cos(th5), y=np.sin(th5), mode="lines",
              line=dict(color="lightgray", width=0.8, dash="dot"), showlegend=False))
          fig_e5.update_layout(
              title=f"Nyquist K={K5_sl:.0f}/[(s+1)(s+3)(s+10)] → {st5}",
              xaxis=dict(title="Re", zeroline=True, zerolinecolor="gray"),
              yaxis=dict(title="Im", zeroline=True, zerolinecolor="gray",
                         scaleanchor="x", scaleratio=1),
              height=360, margin=dict(l=60,r=20,t=50,b=50),
              template="plotly_white",
              legend=dict(orientation="h", y=1.08))
          st.plotly_chart(fig_e5, use_container_width=True)
      
      st.markdown(r"""
      ### 5.3 Exemplo 2 — Sistema com polos no SPD
      
      $$G(s)H(s) = \frac{K(s+3)(s+5)}{(s-2)(s-4)}, \qquad P=2$$
      
      Para $Z=N+P=0$: $N=-2$ (2 envolvimentos **anti-horários**).
      """)
      
      num3_e=np.polymul([1,3],[1,5]); den3_e=np.polymul([1,-2],[1,-4])
      fig5b, axes5b = plt.subplots(1,2,figsize=(9.5,4.5))
      ax_s2=axes5b[0]; ax_n2=axes5b[1]
      plano_s_base(ax_s2,xlim=(-7,6),ylim=(-5,5),titulo="Plano s — P=2 (polos no SPD)")
      for p in np.roots(den3_e): polo_x(ax_s2,p.real,p.imag)
      for z in np.roots(num3_e): zero_o(ax_s2,z.real,z.imag)
      ax_s2.text(-6.5,4.5,"P=2 (polos em +2, +4)",fontsize=8,color="#d62728")
      Re3_e,Im3_e,_=plot_nyquist_ax(ax_n2,num3_e,den3_e,label="G(jω)H(jω)",w_min=1e-2)
      ax_n2.set_xlim(-3,3); ax_n2.set_ylim(-4,4)
      ax_n2.set_title("G=K(s+3)(s+5)/[(s−2)(s−4)] K=1",fontsize=8.5)
      ax_n2.legend(fontsize=7)
      plt.suptitle("Exemplo 2 — P=2: necessário N=−2 para estabilidade",fontsize=9,fontweight="bold")
      plt.tight_layout()
      show_fig(fig5b, 0.88)
      gVal = float(np.polyval(num3_e,0))/float(np.polyval(den3_e,0))
      st.info(f"$G(0)H(0)={gVal:.4f}$ · $P=2$ → para $Z=0$: $N$ deve ser $-2$")
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 6
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("6. Desvio em Polos sobre o Eixo Imaginário")
      
      st.markdown(r"""
      ### 6.1 Problema
      
      Se $G(s)H(s)$ tem polos **sobre o eixo imaginário** ($1/s^n$, $s=\pm j\omega_0$),
      o contorno de Nyquist passaria por esses polos, fazendo $\lvert G(j\omega)H(j\omega)\rvert\to\infty$ — o
      diagrama não pode ser traçado.
      
      ### 6.2 Solução: desvio de raio $\varepsilon \to 0$
      
      | Convenção | Sentido do desvio | Polo fica | $P$ inclui esse polo? |
      |---|---|---|---|
      | **Padrão** (Nise, Ogata, Dorf) | Desvio pelo SPD | **Fora** do contorno | **Não** |
      | Alternativa | Desvio pelo SPE | **Dentro** do contorno | **Sim** (+1 a $P$) |
      
      > As duas convenções são **equivalentes**: a mudança em $P$ é compensada pela mudança em $N$.
      > O importante é ser **consistente**.
      
      ### 6.3 Efeito no diagrama (convenção padrão)
      
      Para integrador $G(s)=K/s$, o desvio $s=\varepsilon e^{j\theta}$ com $\theta:-90°\to+90°$:
      
      $$G(\varepsilon e^{j\theta})\approx\frac{K}{\varepsilon}e^{-j\theta}\xrightarrow{\varepsilon\to 0}\infty\angle(-\theta)$$
      
      | Polo no eixo Im | Arco resultante no plano GH | Ângulo varrido |
      |---|---|---|
      | Simples $s=0$ ($1/s$) | Semicírculo $R\to\infty$, sentido horário | 180° |
      | Duplo $s=0$ ($1/s^2$) | Círculo completo $R\to\infty$, sentido horário | 360° |
      | Par $s=\pm j\omega_0$ | Dois semicírculos $R\to\infty$ | 180° cada |
      """)
      
      fig6a, axes6a = plt.subplots(1,3,figsize=(12.5,4.6))
      subtitles6=["Sem desvio (inválido)",
                  "Desvio: polo excluído\n(variante padrão)",
                  "Desvio: polo incluído\n(variante alternativa)"]
      for ai, ax in enumerate(axes6a):
          ax.axhline(0,color="k",lw=0.8); ax.axvline(0,color="k",lw=0.8)
          ax.set_xlim(-1.2,1.2); ax.set_ylim(-4.5,4.5)
          ax.set_aspect("equal"); ax.axis("off")
          ax.text(-1.1,4.2,"plano s",fontsize=7.5,style="italic",color="gray")
          ax.text(-0.1,4.1,r"$j\omega$",fontsize=10)
          ax.text(1.0,0.1,r"$\sigma$",fontsize=10)
          for py6 in [2.0,-2.0]: polo_x(ax,0,py6,ms=10)
          theta_m6=np.linspace(np.pi/2,-np.pi/2,200)
          ax.plot(0.95*np.cos(theta_m6),0.95*np.sin(theta_m6),"b-",lw=2.2)
          ax.annotate("",
              xy=(0.95*np.cos(-0.6),0.95*np.sin(-0.6)),
              xytext=(0.95*np.cos(-0.5),0.95*np.sin(-0.5)),
              arrowprops=dict(arrowstyle="->",color="blue",lw=2.0))
          if ai == 0:
              for ys6 in [(0.1,1.85),(2.15,3.8),(-0.1,-1.85),(-2.15,-3.8)]:
                  ax.plot([0,0],list(ys6),"b-",lw=2.2)
              ax.text(-1.0,-4.2,"inválido: mapeamento ao infinito",fontsize=7,color="crimson")
          else:
              for ys6 in [(0.1,1.75),(2.25,3.8),(-0.1,-1.75),(-2.25,-3.8)]:
                  ax.plot([0,0],list(ys6),"b-",lw=2.2)
              eps6=0.25
              for py_dev in [2.0,-2.0]:
                  side6=1 if ai==1 else -1
                  th_d6=(np.linspace(np.pi,0,100) if py_dev>0
                         else np.linspace(-np.pi,0,100))
                  ax.plot(side6*eps6*np.cos(th_d6),py_dev+eps6*np.sin(th_d6),"b-",lw=2.2)
              col6="seagreen" if ai==1 else "#1f77b4"
              ax.text(-1.0,-4.2,"polo excluído" if ai==1 else "polo incluído",
                      fontsize=8,color=col6)
          ax.set_title(subtitles6[ai],fontsize=8.5)
      fig6a.suptitle("Desvio em polos no eixo imaginário",fontsize=9)
      plt.tight_layout()
      show_fig(fig6a, 0.90)
      
      st.markdown(r"""
      ### 6.4 Exemplo com integrador — $G(s)H(s)=1/[s(s+3)(s+5)]$
      
      Para $\omega_{pc}=\sqrt{15}$ rad/s:
      
      $$G(j\sqrt{15})H(j\sqrt{15}) = \frac{1}{j\sqrt{15}\cdot(j\sqrt{15}+3)\cdot(j\sqrt{15}+5)} = \frac{1}{-120} \approx -0{,}0083$$
      
      $N=0$, $Z=N+P=0$ → **ESTÁVEL**. Margem de ganho: $G_m=120\approx41{,}6$ dB.
      """)
      
      num_int6=[1]; den_int6=np.polymul([1,0],np.polymul([1,3],[1,5]))
      fig6b, axes6b = plt.subplots(1,2,figsize=(9.5,4.5))
      ax_si=axes6b[0]; ax_ni=axes6b[1]
      plano_s_base(ax_si,xlim=(-7,2),ylim=(-5,5),titulo="Plano s — polo em s=0 (integrador)")
      for p in np.roots(den_int6): polo_x(ax_si,p.real,p.imag)
      eps_arc6=np.linspace(-np.pi/2,np.pi/2,100)
      ax_si.plot(0.15*np.cos(eps_arc6[::-1]),0.15*np.sin(eps_arc6[::-1]),"b--",lw=1.5,alpha=0.7)
      
      Re_i6,Im_i6,w_i6=nyquist_data(num_int6,den_int6,w_min=1e-2,w_max=1e3)
      ax_ni.plot(Re_i6, Im_i6,"b-",lw=2.2,label=r"$\omega>0$")
      ax_ni.plot(Re_i6,-Im_i6,"b--",lw=2.2,alpha=0.55,label=r"$\omega<0$")
      ax_ni.plot(-1,0,"r*",ms=14,zorder=6,label="(−1,0)")
      R_inf6=max(abs(Re_i6[0]),abs(Im_i6[0]))*0.9
      th_arc6=np.linspace(np.pi/2,-np.pi/2,100)
      ax_ni.plot(R_inf6*np.cos(th_arc6),R_inf6*np.sin(th_arc6),
          "b:",lw=1.5,alpha=0.6,label="Arco infinito (desvio s=0)")
      sign_ii6=np.sign(Im_i6)
      cross_i6=[k for k in range(1,len(Im_i6))
                if sign_ii6[k-1]*sign_ii6[k]<0 and Re_i6[k]<0]
      if cross_i6:
          re_pc_i6=Re_i6[cross_i6[0]]
          ax_ni.plot(re_pc_i6,0,"s",color="#ff7f0e",ms=10,zorder=5,
              label=f"ωpc=√15: Re={re_pc_i6:.5f}")
      ax_ni.set_xlim(-0.06,0.08); ax_ni.set_ylim(-0.07,0.07)
      ax_ni.set_title("G(jω)H(jω)=1/[jω(jω+3)(jω+5)]",fontsize=8.5)
      ax_ni.legend(fontsize=7)
      ax_ni.axhline(0,color="k",lw=0.5,ls="--",alpha=0.4)
      ax_ni.axvline(0,color="k",lw=0.5,ls="--",alpha=0.4)
      ax_ni.spines[["right","top"]].set_visible(False)
      plt.suptitle("Integrador — P=0, N=0 → Z=0 (ESTÁVEL)",fontsize=9,fontweight="bold")
      plt.tight_layout()
      show_fig(fig6b, 0.88)
      if cross_i6:
          st.info(f"Cruzamento numérico: $Re[G(j\\sqrt{{15}})] = {re_pc_i6:.6f}$ · "
                  f"$\\sqrt{{15}}={np.sqrt(15):.5f}$ rad/s · $G_m=1/{abs(re_pc_i6):.4f}={1/abs(re_pc_i6):.1f}\\approx41$ dB")
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 7
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("7. Margens de Ganho e Fase pelo Critério de Nyquist")
      
      st.markdown(r"""
      ### 7.1 Margem de fase $\phi_m$
      
      Na frequência $\omega_{gc}$ onde $\lvert G(j\omega_{gc})H(j\omega_{gc})\rvert=1$:
      
      $$\boxed{\phi_m = 180° + \angle G(j\omega_{gc})H(j\omega_{gc})}$$
      
      $\phi_m$ é o atraso adicional que levaria o cruzamento do círculo unitário ao ponto $(-1, j0)$.
      
      ### 7.2 Margem de ganho $G_m$
      
      Na frequência $\omega_{pc}$ onde $\angle G(j\omega_{pc})H(j\omega_{pc})=-180°$:
      
      $$\boxed{G_m = \frac{1}{\lvert G(j\omega_{pc})H(j\omega_{pc})\rvert}, \qquad G_m\big|_{\text{dB}} = -20\log_{10}\lvert G(j\omega_{pc})H(j\omega_{pc})\rvert}$$
      
      | $G_m$ | Posição do cruzamento em $\omega_{pc}$ | Estabilidade ($P=0$) |
      |---|---|---|
      | $G_m>1\;(>0\,\text{dB})$ | Entre $-1$ e a origem | ✅ Estável |
      | $G_m=1\;(0\,\text{dB})$ | Exatamente em $(-1,j0)$ | Marginal |
      | $G_m<1\;(<0\,\text{dB})$ | Além de $-1$ | ❌ Instável |
      
      **Regra prática:** $30°\leq\phi_m\leq60°$ e $G_m\geq6$ dB.
      
      > **Cuidado:** margens são condições suficientes de estabilidade **apenas quando $P=0$**.
      """)
      
      # Fig 7.1 — interpretação geométrica
      num_geo7=[10]; den_geo7=[1,6,11,6]
      w_geo7=np.logspace(-1,2,5000)
      _,H_geo7=signal.freqs(num_geo7,den_geo7,worN=w_geo7)
      Re_geo7=H_geo7.real; Im_geo7=H_geo7.imag; mag_geo7=np.abs(H_geo7)
      idx_gc7=np.argmin(np.abs(mag_geo7-1.0))
      H_gc7=H_geo7[idx_gc7]; w_gc7=w_geo7[idx_gc7]
      pm7=180+np.degrees(np.angle(H_gc7))
      sign_im7=np.sign(Im_geo7)
      cross7=[i for i in range(1,len(Im_geo7))
              if sign_im7[i-1]*sign_im7[i]<0 and Re_geo7[i]<0]
      re_pc7=Re_geo7[cross7[0]] if cross7 else None
      gm_db7=-20*np.log10(abs(re_pc7)) if re_pc7 else None
      
      fig7a, ax7a = plt.subplots(figsize=(6.5,5.5))
      ax7a.plot(Re_geo7,Im_geo7,"b-",lw=2.2,label=r"$G(j\omega)H(j\omega),\;\omega>0$")
      ax7a.plot(Re_geo7,-Im_geo7,"b--",lw=2.2,alpha=0.4,label=r"$\omega<0$ (conj.)")
      th7=np.linspace(0,2*np.pi,300)
      ax7a.plot(np.cos(th7),np.sin(th7),color="gray",lw=0.9,ls=":",label=r"Círculo unitário $\|GH\|=1$")
      ax7a.plot(-1,0,"r*",ms=16,zorder=7,label="(−1,0) ponto crítico")
      ax7a.axhline(0,color="k",lw=0.6,ls="--",alpha=0.3)
      ax7a.axvline(0,color="k",lw=0.6,ls="--",alpha=0.3)
      ax7a.plot(H_gc7.real,H_gc7.imag,"go",ms=11,zorder=6,
          label=rf"$\omega_{{gc}}={w_gc7:.2f}$: $\phi_m={pm7:.1f}°$")
      ax7a.annotate("",xy=(H_gc7.real,H_gc7.imag),xytext=(0,0),
          arrowprops=dict(arrowstyle="->,head_width=0.07",color="green",lw=1.8))
      arc7=np.linspace(np.pi,np.angle(H_gc7),60); r_arc7=0.42
      ax7a.plot(r_arc7*np.cos(arc7),r_arc7*np.sin(arc7),"g-",lw=1.6)
      mid_ang7=(np.pi+np.angle(H_gc7))/2
      ax7a.text(r_arc7*np.cos(mid_ang7)-0.08,r_arc7*np.sin(mid_ang7)+0.07,
          rf"$\phi_m={pm7:.1f}°$",fontsize=9,color="green",ha="right")
      if re_pc7 is not None:
          ax7a.plot(re_pc7,0,"rs",ms=11,zorder=6,
              label=rf"$\omega_{{pc}}$: $G_m={gm_db7:.1f}$ dB")
          ax7a.annotate("",xy=(-1,0),xytext=(re_pc7,0),
              arrowprops=dict(arrowstyle="<->",color="red",lw=1.8))
          ax7a.text((re_pc7-1)/2,-0.11,
              rf"$G_m={gm_db7:.1f}$ dB",fontsize=9,color="red",ha="center")
      ax7a.set_xlabel("Re",fontsize=9); ax7a.set_ylabel("Im",fontsize=9)
      ax7a.set_title(r"$\phi_m$ e $G_m$ — $G(s)H(s)=10/[(s+1)(s+2)(s+3)]$",fontsize=9)
      ax7a.legend(fontsize=7.5,loc="upper right")
      ax7a.set_xlim(-2.0,1.8); ax7a.set_ylim(-1.5,1.5)
      ax7a.set_aspect("equal"); ax7a.spines[["right","top"]].set_visible(False)
      plt.tight_layout()
      show_fig(fig7a, 0.52)
      if gm_db7:
          st.info(f"$\\phi_m={pm7:.1f}°$ · $G_m={gm_db7:.1f}$ dB · $\\omega_{{gc}}={w_gc7:.3f}$ rad/s")
      
      # ── 7.4 Comparação Nyquist × Bode × LGR ─────────────────────────────────────
      st.markdown("### 7.4 🎛️ Comparação Nyquist × Bode × LGR")
      st.caption("Insira $G(s)H(s)=K\\cdot N(s)/D(s)$ e veja os três diagramas sincronizados")
      
      st.markdown("""
      | Exemplo | Numerador | Denominador |
      |:---|:---|:---|
      | $10/[s(s+1)(s+2)]$ | `10` | `1, 3, 2, 0` |
      | $1/[s(s+1)(s+3)(s+5)]` | `1` | `1, 9, 23, 15, 0` |
      | $6(s+2)/[s(s+1)(s+4)]` | `6, 12` | `1, 5, 4, 0` |
      """)
      
      c74a, c74b = st.columns([1, 3])
      with c74a:
          num74_str = st.text_input("Numerador $N(s)$", value="10", key="num74")
          den74_str = st.text_input("Denominador $D(s)$", value="1, 3, 2, 0", key="den74")
          K74_sl   = st.slider("Ganho $K$", 0.1, 5.0, 1.0, 0.1, key="K74")
          wmin74   = st.slider("$\\log_{10}(\\omega_{min})$", -3.0, 0.0, -2.0, 0.5, key="wmin74")
          wmax74   = st.slider("$\\log_{10}(\\omega_{max})$", 0.5, 4.0, 2.0, 0.5, key="wmax74")
          K_lgr74  = st.slider("$k_{max}$ LGR ($10^x$)", 0.5, 4.0, 2.0, 0.5, key="klgr74")
      
      with c74b:
          try:
              num74=[K74_sl*float(x.strip()) for x in num74_str.split(",") if x.strip()]
              den74=[float(x.strip()) for x in den74_str.split(",") if x.strip()]
              assert len(den74)>len(num74), "Grau Den deve ser > Num"
              n_p74=len(den74)-1
              w74=np.logspace(max(wmin74,-2), min(wmax74,3), 6000)
              _,H74=signal.freqs(num74,den74,worN=w74)
              mag74_db=20*np.log10(np.abs(H74)+1e-15)
              fase74=np.degrees(np.unwrap(np.angle(H74)))
              Re74=H74.real; Im74=H74.imag
      
              # Marcos de fase
              milestones74=[]
              for mult in range(-1,-9,-1):
                  pt=mult*90; diff=fase74-pt
                  crossings=np.where(np.diff(np.sign(diff)))[0]
                  if len(crossings):
                      i0=crossings[0]
                      den_i=diff[i0+1]-diff[i0]
                      frac=0.0 if abs(den_i)<1e-15 else max(0,min(1,-diff[i0]/den_i))
                      w0=float(w74[i0]+(w74[i0+1]-w74[i0])*frac)
                      _,Hw=signal.freqs(num74,den74,worN=[w0]); Hw=Hw[0]
                      milestones74.append({"pt":pt,"w":w0,
                          "mag":float(20*np.log10(abs(Hw)+1e-15)),
                          "Re":float(Hw.real),"Im":float(Hw.imag),
                          "col":PHASE_COLORS.get(pt,"#333")})
      
              # Margens
              sign_m74=np.sign(mag74_db)
              cross_0db74=np.where(np.diff(sign_m74))[0]
              w_gc74=float(w74[cross_0db74[0]]) if len(cross_0db74) else None
              pm74=None
              if w_gc74:
                  _,H_gc74=signal.freqs(num74,den74,worN=[w_gc74])
                  pm74=180+float(np.degrees(np.angle(H_gc74[0])))
              m_180_74=next((m for m in milestones74 if m["pt"]==-180),None)
              gm_db74=float(-m_180_74["mag"]) if m_180_74 else None
      
              # LGR
              k_lgr_max74=10**K_lgr74
              ramos74,k_arr74=calc_lgr(num74,den74,k_max=k_lgr_max74)
              poles_ma74=np.roots(den74)
              zeros_ma74=np.roots(num74) if len(num74)>1 else np.array([])
      
              def find_k_lgr(target_w):
                  best_ki=None; best_d=float("inf")
                  for ki in range(len(k_arr74)):
                      for ri in range(n_p74):
                          d=abs(abs(ramos74[ri][ki].imag)-target_w)
                          if d<best_d: best_d=d; best_ki=ki
                  return best_ki
      
              # Figura 2×2
              fig74=make_subplots(rows=2,cols=2,
                  subplot_titles=("Magnitude (dB)","Diagrama de Nyquist",
                                  "Fase (°)","LGR"),
                  column_widths=[0.50,0.50], row_heights=[0.55,0.45],
                  horizontal_spacing=0.10, vertical_spacing=0.14)
      
              fig74.add_trace(go.Scatter(x=w74,y=mag74_db,mode="lines",
                  line=dict(color="#1f77b4",width=2.0),name="Mag",
                  hovertemplate="ω=%{x:.3f}<br>%{y:.2f} dB<extra></extra>"),row=1,col=1)
              fig74.add_hline(y=0,line_color="gray",line_dash="dash",line_width=0.8,row=1,col=1)
              fig74.add_trace(go.Scatter(x=w74,y=fase74,mode="lines",
                  line=dict(color="#1f77b4",width=2.0),name="Fase",showlegend=False,
                  hovertemplate="ω=%{x:.3f}<br>%{y:.1f}°<extra></extra>"),row=2,col=1)
              fig74.add_hline(y=-180,line_color="gray",line_dash="dash",line_width=0.8,row=2,col=1)
              fig74.add_trace(go.Scatter(x=Re74,y=Im74,mode="lines",
                  line=dict(color="#2ca02c",width=2.0),name="Nyquist ω>0"),row=1,col=2)
              fig74.add_trace(go.Scatter(x=Re74,y=-Im74,mode="lines",
                  line=dict(color="#2ca02c",width=2.0,dash="dot"),
                  name="ω<0",showlegend=False),row=1,col=2)
              fig74.add_trace(go.Scatter(x=[-1],y=[0],mode="markers",
                  marker=dict(symbol="star",size=14,color="red"),
                  name="(−1,0)",showlegend=False),row=1,col=2)
              th_u74=np.linspace(0,2*np.pi,200)
              fig74.add_trace(go.Scatter(x=np.cos(th_u74),y=np.sin(th_u74),mode="lines",
                  line=dict(color="lightgray",width=0.8,dash="dot"),showlegend=False),row=1,col=2)
              for ri,ramo in enumerate(ramos74):
                  col_r=CORES_LGR[ri%len(CORES_LGR)]
                  fig74.add_trace(go.Scatter(
                      x=[r.real for r in ramo],y=[r.imag for r in ramo],
                      mode="lines",line=dict(color=col_r,width=1.8),
                      name=f"Ramo {ri+1}"),row=2,col=2)
              fig74.add_trace(go.Scatter(
                  x=[p.real for p in poles_ma74],y=[p.imag for p in poles_ma74],
                  mode="markers",marker=dict(symbol="x",size=12,color="#d62728",
                                              line=dict(width=2.5,color="#d62728")),
                  name="Polos MA"),row=2,col=2)
              if len(zeros_ma74)>0:
                  fig74.add_trace(go.Scatter(
                      x=[z.real for z in zeros_ma74],y=[z.imag for z in zeros_ma74],
                      mode="markers",marker=dict(symbol="circle-open",size=11,color="#333",
                          line=dict(width=2,color="#333")),name="Zeros MA"),row=2,col=2)
              fig74.add_vline(x=0,line_color="black",line_width=0.7,row=2,col=2)
      
              for m74 in milestones74:
                  col_m=m74["col"]; w0=m74["w"]; pt=m74["pt"]; sl=f"{pt}°"
                  fig74.add_vline(x=w0,line_color=col_m,line_dash="dot",line_width=1.2,row=1,col=1)
                  fig74.add_trace(go.Scatter(x=[w0],y=[m74["mag"]],mode="markers",
                      marker=dict(size=8,color=col_m),showlegend=False,
                      hovertemplate=f"Fase={pt}° ω={w0:.3f}<br>{m74['mag']:.1f} dB<extra></extra>"),row=1,col=1)
                  fig74.add_vline(x=w0,line_color=col_m,line_dash="dot",line_width=1.2,row=2,col=1)
                  fig74.add_trace(go.Scatter(x=[w0],y=[float(pt)],mode="markers",
                      marker=dict(size=8,color=col_m,symbol="diamond"),showlegend=False),row=2,col=1)
                  fig74.add_trace(go.Scatter(x=[m74["Re"]],y=[m74["Im"]],mode="markers",
                      marker=dict(size=9,color=col_m,symbol="circle",
                                  line=dict(color="white",width=1.5)),
                      showlegend=False,
                      hovertemplate=f"Fase={pt}°<br>Re={m74['Re']:.4f}<br>Im={m74['Im']:.4f}<extra></extra>"),row=1,col=2)
                  fig74.add_trace(go.Scatter(x=[m74["Re"]],y=[-m74["Im"]],mode="markers",
                      marker=dict(size=7,color=col_m,symbol="circle-open"),showlegend=False),row=1,col=2)
                  k_idx74=find_k_lgr(w0)
                  if k_idx74 is not None:
                      for ri74 in range(n_p74):
                          r74=ramos74[ri74][k_idx74]
                          fig74.add_trace(go.Scatter(x=[r74.real],y=[r74.imag],mode="markers",
                              marker=dict(size=9,color=col_m,symbol="diamond",
                                          line=dict(color="white",width=1)),
                              showlegend=False,
                              hovertemplate=f"k={k_arr74[k_idx74]:.3f}<br>σ={r74.real:.3f}<br>jω={r74.imag:.3f}<extra>{pt}°</extra>"),row=2,col=2)
      
              if w_gc74:
                  _,Hgc74=signal.freqs(num74,den74,worN=[w_gc74])
                  fig74.add_trace(go.Scatter(x=[w_gc74],y=[0],mode="markers",
                      marker=dict(size=11,color="seagreen",symbol="square"),
                      name=f"φm={pm74:.1f}°"),row=1,col=1)
                  fig74.add_trace(go.Scatter(x=[Hgc74[0].real],y=[Hgc74[0].imag],mode="markers",
                      marker=dict(size=11,color="seagreen",symbol="square"),showlegend=False),row=1,col=2)
              if gm_db74 and m_180_74:
                  fig74.add_trace(go.Scatter(x=[m_180_74["w"]],y=[m_180_74["mag"]],mode="markers",
                      marker=dict(size=11,color="#d62728",symbol="square"),
                      name=f"Gm={gm_db74:.1f} dB"),row=1,col=1)
      
              fig74.update_xaxes(type="log",title_text="ω (rad/s)",row=1,col=1)
              fig74.update_xaxes(type="log",title_text="ω (rad/s)",row=2,col=1)
              fig74.update_yaxes(title_text="Magnitude (dB)",row=1,col=1)
              fig74.update_yaxes(title_text="Fase (°)",row=2,col=1)
              fig74.update_xaxes(title_text="Re",zeroline=True,zerolinecolor="gray",autorange=True,row=1,col=2)
              fig74.update_yaxes(title_text="Im",zeroline=True,zerolinecolor="gray",autorange=True,row=1,col=2)
              fig74.update_xaxes(title_text="σ (Re s)",zeroline=True,zerolinecolor="black",autorange=True,row=2,col=2)
              fig74.update_yaxes(title_text="jω (Im s)",zeroline=True,zerolinecolor="black",autorange=True,row=2,col=2)
      
              P_e74=sum(1 for p in poles_ma74 if p.real>1e-9)
              pm_str74=f"φm={pm74:.1f}°" if pm74 else "sem cruzamento 0dB"
              gm_str74=f"Gm={gm_db74:.1f}dB" if gm_db74 else "sem cruzamento −180°"
              fig74.update_layout(
                  title=dict(text=f"N=[{num74_str}]/D=[{den74_str}]  K={K74_sl:.1f}  {pm_str74}  {gm_str74}  P={P_e74}",
                             font=dict(size=10)),
                  height=640, margin=dict(l=60,r=20,t=65,b=80),
                  template="plotly_white",
                  legend=dict(orientation="h",x=0.5,y=-0.10,xanchor="center",yanchor="top",font=dict(size=8)))
              st.plotly_chart(fig74, use_container_width=True)
              info74=[]
              if pm74: info74.append(f"$\\phi_m={pm74:.2f}°$ em $\\omega_{{gc}}={w_gc74:.4f}$ rad/s")
              if gm_db74: info74.append(f"$G_m={gm_db74:.2f}$ dB")
              info74.append(f"$P={P_e74}$")
              if info74: st.info("  ·  ".join(info74))
          except AssertionError as e:
              st.error(str(e))
          except Exception as ex:
              st.error(f"Erro: {ex}")
      
      st.divider()
      
      
      # ═══════════════════════════════════════════════════════════════════════════════
      # SEÇÃO 8 — EXPLORADOR GERAL
      # ═══════════════════════════════════════════════════════════════════════════════
      st.header("8. Explorador Interativo de Nyquist")
      
      st.markdown("""
      Insira $G(s)H(s)=K\\cdot N(s)/D(s)$, ajuste $K$ e $\\omega_{max}$. Marcações automáticas:
      
      | Marcador | Indicador |
      |:---|:---|
      | 🟢 Círculo | Cruzamento do círculo unitário → Margem de fase $\\phi_m$ |
      | 🟥 Quadrado | Cruzamento do eixo real negativo → Margem de ganho $G_m$ |
      | ⭐ Estrela | Ponto crítico $(-1, j0)$ |
      
      | $G(s)H(s)$ | Numerador | Denominador |
      |:---|:---|:---|
      | $500/[(s+1)(s+3)(s+10)]$ | `500` | `1, 14, 53, 30` |
      | $1/[s(s+1)(s+2)]$ | `1` | `1, 3, 2, 0` |
      | $K(s+3)(s+5)/[(s-2)(s-4)]$ | `1, 8, 15` | `1, -6, 8` |
      | $1/[s(s+3)(s+5)]$ | `1` | `1, 8, 15, 0` |
      """)
      
      c8a, c8b = st.columns([1, 2])
      with c8a:
          num8_str = st.text_input("Numerador $N(s)$", value="500", key="num8ny")
          den8_str = st.text_input("Denominador $D(s)$", value="1, 14, 53, 30", key="den8ny")
          K8_sl    = st.slider("Ganho $K$", 0.1, 20.0, 1.0, 0.1, key="K8ny")
          wmax8    = st.slider("$\\log_{10}(\\omega_{max})$", 1.0, 5.0, 3.0, 0.5, key="wmax8ny")
      
      with c8b:
          try:
              num8=[K8_sl*float(x.strip()) for x in num8_str.split(",") if x.strip()]
              den8=[float(x.strip()) for x in den8_str.split(",") if x.strip()]
              assert len(den8)>len(num8), "Grau Den deve ser > Num."
      
              w8=np.logspace(-2, wmax8, 6000)
              _,H8=signal.freqs(num8,den8,worN=w8)
              Re8=H8.real; Im8=H8.imag; mag8=np.abs(H8)
      
              # Margem de fase
              idx_gc8=np.argmin(np.abs(mag8-1.0))
              H_gc8=H8[idx_gc8]; pm8=180+np.degrees(np.angle(H_gc8)); w_gc8=float(w8[idx_gc8])
      
              # Margem de ganho
              sign_im8=np.sign(Im8)
              cross_neg8=[i for i in range(1,len(Im8))
                          if sign_im8[i-1]*sign_im8[i]<0 and Re8[i]<0]
              re_pc8=Re8[cross_neg8[0]] if cross_neg8 else None
              gm8=float(-20*np.log10(abs(re_pc8))) if re_pc8 else float("inf")
      
              # P e N
              P8=sum(1 for p in np.roots(den8) if p.real>1e-9)
              N8=count_N(Re8,Im8)
              Z8=N8+P8; est8="ESTÁVEL" if Z8==0 else f"INSTÁVEL (Z={Z8})"
      
              # Figura
              th8=np.linspace(0,2*np.pi,300)
              fig_e8=go.Figure()
              fig_e8.add_trace(go.Scatter(x=Re8,y=Im8,mode="lines",
                  line=dict(color="#1f77b4",width=2.5),name="GH(jω), ω>0",
                  hovertemplate="Re=%{x:.4f}<br>Im=%{y:.4f}<extra></extra>"))
              fig_e8.add_trace(go.Scatter(x=Re8,y=-Im8,mode="lines",
                  line=dict(color="#1f77b4",width=2.5,dash="dot"),name="ω<0 (conj.)",showlegend=True))
              fig_e8.add_trace(go.Scatter(x=np.cos(th8),y=np.sin(th8),mode="lines",
                  line=dict(color="lightgray",width=0.8,dash="dot"),name="Círculo unitário"))
              fig_e8.add_trace(go.Scatter(x=[-1],y=[0],mode="markers",
                  marker=dict(symbol="star",size=16,color="red"),name="(−1,0)"))
              fig_e8.add_trace(go.Scatter(x=[H_gc8.real],y=[H_gc8.imag],
                  mode="markers+text",
                  marker=dict(size=12,color="seagreen",symbol="circle"),
                  text=[f"φm={pm8:.1f}°"],
                  textposition="top right" if H_gc8.imag>=0 else "bottom right",
                  name=f"φm={pm8:.1f}°"))
              if re_pc8 is not None:
                  fig_e8.add_trace(go.Scatter(x=[re_pc8],y=[0],
                      mode="markers+text",
                      marker=dict(size=12,color="crimson",symbol="square"),
                      text=[f"Gm={gm8:.1f}dB"],textposition="top right",
                      name=f"Gm={gm8:.1f}dB"))
              fig_e8.add_hline(y=0,line_color="lightgray",line_width=0.8)
              fig_e8.add_vline(x=0,line_color="lightgray",line_width=0.8)
      
              est8_icon="🟢" if Z8==0 else "🔴"
              title8=f"Nyquist N=[{num8_str}]/D=[{den8_str}]  K={K8_sl:.1f}  P={P8}  N={N8}  Z={Z8}  {est8_icon} {est8}"
              fig_e8.update_layout(
                  title=dict(text=title8,font=dict(size=11)),
                  yaxis=dict(title="Im",zeroline=True,zerolinecolor="gray",
                             zerolinewidth=1,scaleanchor="x",scaleratio=1),
                  xaxis=dict(title="Re",zeroline=True,zerolinecolor="gray",zerolinewidth=1),
                  height=500,margin=dict(l=60,r=20,t=65,b=80),
                  template="plotly_white",
                  legend=dict(orientation="h",x=0.5,y=-0.12,
                              xanchor="center",yanchor="top",font=dict(size=9)))
              st.plotly_chart(fig_e8,use_container_width=True)
      
              info8=[]
              info8.append(f"**{est8_icon} {est8}**")
              info8.append(f"$P={P8}$ · $N={N8}$ · $Z={Z8}$")
              info8.append(f"$\\phi_m={pm8:.2f}°$")
              if re_pc8: info8.append(f"$G_m={gm8:.2f}$ dB")
              st.info("\n\n".join(info8))
          except AssertionError as e:
              st.error(str(e))
          except Exception as ex:
              st.error(f"Erro: {ex}")
      
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
          "🔁 Critério de Nyquist &nbsp;·&nbsp; 🎛️ SINTONIA — Sistemas de Controle<br>"
          "👤 Marcus V A Fernandes &nbsp;·&nbsp; 🏛️ Diretoria de Indústria &nbsp;·&nbsp; IFRN-CNAT"
          " &nbsp;·&nbsp; 🏷️ v1.0 &nbsp;·&nbsp; 📅 2026"
          "</div>",
          unsafe_allow_html=True,
      )
