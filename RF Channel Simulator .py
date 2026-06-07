# =============================================================================
#  RF Channel Simulator  —  v1.0
#  Autor  : Alex (refactored & unified)
#  Módulos: Rayleigh Fading · OFDM Tx · Antenna Pattern · WiFi Heatmap
#  Deps   : pip install numpy matplotlib
# =============================================================================

import tkinter as tk
from tkinter import ttk, font as tkfont
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

# ─────────────────────────────────────────────────────────────────────────────
#  PALETA / CONSTANTES VISUALES
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK    = "#0d0f14"
BG_PANEL   = "#13161e"
BG_CARD    = "#1a1e2a"
ACCENT     = "#00d4ff"
ACCENT2    = "#ff6b35"
ACCENT3    = "#7fff6b"
TEXT_PRI   = "#e8eaf0"
TEXT_SEC   = "#6b7280"
BORDER     = "#252a38"

MPL_STYLE = {
    "figure.facecolor":  BG_PANEL,
    "axes.facecolor":    BG_CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_SEC,
    "axes.titlecolor":   TEXT_PRI,
    "xtick.color":       TEXT_SEC,
    "ytick.color":       TEXT_SEC,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        TEXT_PRI,
    "lines.linewidth":   1.8,
}
plt.rcParams.update(MPL_STYLE)


# =============================================================================
#  MÓDULO 1 — RAYLEIGH FADING  (Jake's sum-of-sinusoids)
# =============================================================================
class RayleighModule(ttk.Frame):
    """Simula desvanecimiento Rayleigh con modelo de Jake simplificado."""

    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame")
        self._build_controls()
        self._build_canvas()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_controls(self):
        ctrl = ttk.Frame(self, style="Card.TFrame")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(18, 6))

        ttk.Label(ctrl, text="RAYLEIGH FADING  /  MODELO DE JAKE",
                  style="ModTitle.TLabel").pack(side=tk.LEFT)

        row = ttk.Frame(ctrl, style="Card.TFrame")
        row.pack(side=tk.RIGHT)

        self._add_param(row, "Velocidad (m/s)", "v", "15", 0)
        self._add_param(row, "fc  (GHz)",       "fc","2.4", 1)
        self._add_param(row, "Sinusoides M",    "M", "16",  2)

        ttk.Button(row, text="▶  Simular", style="Accent.TButton",
                   command=self._simulate).grid(row=0, column=6,
                   rowspan=2, padx=(18, 0), sticky="ns")

    def _add_param(self, parent, label, key, default, col):
        ttk.Label(parent, text=label, style="ParamLabel.TLabel"
                  ).grid(row=0, column=col*2, padx=(12, 4), sticky="w")
        entry = ttk.Entry(parent, width=7, style="Param.TEntry")
        entry.insert(0, default)
        entry.grid(row=0, column=col*2+1, padx=(0, 4))
        setattr(self, f"_e_{key}", entry)

    def _build_canvas(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 3.6))
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=14, pady=(0, 14))
        self._draw_placeholder()

    # ── SIMULACIÓN ───────────────────────────────────────────────────────────
    def _simulate(self):
        N  = 2000
        v  = float(self._e_v.get())
        fc = float(self._e_fc.get()) * 1e9
        M  = max(4, int(self._e_M.get()))
        c  = 3e8
        fd = v * fc / c

        t = np.linspace(0, 1, N)
        s = np.zeros(N, dtype=complex)
        for n in range(1, M + 1):
            phi = np.random.rand() * 2 * np.pi
            s  += np.exp(1j * (2 * np.pi * fd * np.cos(2 * np.pi * n / M) * t + phi))
        s   = (s - np.mean(s)) / np.std(s)
        ray = np.abs(s)
        ray_db = 20 * np.log10(ray + 1e-12)

        ax1, ax2 = self.axes
        ax1.clear(); ax2.clear()

        ax1.plot(t, ray, color=ACCENT, alpha=0.9)
        ax1.set_title(f"Envolvente Rayleigh  |  fd = {fd:.1f} Hz")
        ax1.set_xlabel("Tiempo (s)")
        ax1.set_ylabel("Amplitud")
        ax1.grid(True)

        ax2.hist(ray_db, bins=60, color=ACCENT2, edgecolor="none", alpha=0.85,
                 density=True)
        ax2.set_title("PDF de la envolvente (dB)")
        ax2.set_xlabel("Nivel (dB)")
        ax2.set_ylabel("Densidad")
        ax2.grid(True)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def _draw_placeholder(self):
        for ax in self.axes:
            ax.text(0.5, 0.5, "Ajusta parámetros y pulsa  ▶  Simular",
                    ha="center", va="center", color=TEXT_SEC,
                    transform=ax.transAxes, fontsize=10)
        self.canvas.draw()


# =============================================================================
#  MÓDULO 2 — OFDM TX
# =============================================================================
class OFDMModule(ttk.Frame):
    """Generación de señal OFDM con prefijo cíclico y espectro."""

    CONST_MAP = {
        "BPSK":  {0: 1+0j, 1: -1+0j},
        "QPSK":  {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j},
        "16-QAM": None,   # generado dinámicamente
    }

    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame")
        self._build_controls()
        self._build_canvas()

    def _build_controls(self):
        ctrl = ttk.Frame(self, style="Card.TFrame")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(18, 6))

        ttk.Label(ctrl, text="OFDM  TX  —  ANÁLISIS DE SEÑAL",
                  style="ModTitle.TLabel").pack(side=tk.LEFT)

        row = ttk.Frame(ctrl, style="Card.TFrame")
        row.pack(side=tk.RIGHT)

        ttk.Label(row, text="Subportadoras N", style="ParamLabel.TLabel"
                  ).grid(row=0, column=0, padx=(0, 4))
        self._e_N = ttk.Entry(row, width=6, style="Param.TEntry")
        self._e_N.insert(0, "64")
        self._e_N.grid(row=0, column=1, padx=(0, 14))

        ttk.Label(row, text="Constelación", style="ParamLabel.TLabel"
                  ).grid(row=0, column=2, padx=(0, 4))
        self._cb_mod = ttk.Combobox(row, values=["BPSK", "QPSK", "16-QAM"],
                                     state="readonly", width=8)
        self._cb_mod.set("QPSK")
        self._cb_mod.grid(row=0, column=3, padx=(0, 14))

        ttk.Label(row, text="CP ratio", style="ParamLabel.TLabel"
                  ).grid(row=0, column=4, padx=(0, 4))
        self._e_cp = ttk.Entry(row, width=5, style="Param.TEntry")
        self._e_cp.insert(0, "0.25")
        self._e_cp.grid(row=0, column=5, padx=(0, 14))

        ttk.Button(row, text="▶  Generar", style="Accent.TButton",
                   command=self._generate).grid(row=0, column=6, sticky="ns")

    def _build_canvas(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 4.4))
        self.fig.tight_layout(pad=2.8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=14, pady=(0, 14))
        self._draw_placeholder()

    def _generate(self):
        N      = int(self._e_N.get())
        mod    = self._cb_mod.get()
        cp_r   = float(self._e_cp.get())
        cp_len = int(N * cp_r)

        # Símbolos
        if mod == "BPSK":
            bits = np.random.randint(0, 2, N)
            symbols = np.array([1.0 if b == 0 else -1.0 for b in bits], dtype=complex)
        elif mod == "QPSK":
            bits = np.random.randint(0, 4, N)
            m    = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
            symbols = np.array([m[b] / np.sqrt(2) for b in bits])
        else:   # 16-QAM
            pts  = np.array([-3, -1, 1, 3])
            I, Q = np.random.choice(pts, N), np.random.choice(pts, N)
            symbols = (I + 1j * Q) / np.sqrt(10)

        tx    = np.fft.ifft(symbols)
        tx_cp = np.concatenate([tx[-cp_len:], tx])
        t     = np.arange(len(tx_cp))
        freqs = np.fft.fftfreq(1024)
        spec  = np.abs(np.fft.fftshift(np.fft.fft(tx, 1024)))

        (ax1, ax2), (ax3, ax4) = self.axes
        for ax in self.axes.flat: ax.clear()

        # Forma de onda — I
        ax1.plot(t, np.real(tx_cp), color=ACCENT, linewidth=1.2)
        ax1.axvspan(0, cp_len, color=ACCENT2, alpha=0.18, label="Prefijo cíclico")
        ax1.set_title("Dominio tiempo  (componente I)")
        ax1.set_xlabel("Muestra"); ax1.set_ylabel("Amplitud"); ax1.grid(True)
        ax1.legend(fontsize=8)

        # Forma de onda — Q
        ax2.plot(t, np.imag(tx_cp), color=ACCENT3, linewidth=1.2)
        ax2.axvspan(0, cp_len, color=ACCENT2, alpha=0.18)
        ax2.set_title("Dominio tiempo  (componente Q)")
        ax2.set_xlabel("Muestra"); ax2.set_ylabel("Amplitud"); ax2.grid(True)

        # Espectro
        ax3.plot(np.fft.fftshift(freqs) * N, spec, color=ACCENT2)
        ax3.set_title("Espectro de potencia")
        ax3.set_xlabel("Subportadora"); ax3.set_ylabel("|X(f)|"); ax3.grid(True)

        # Constelación
        ax4.scatter(np.real(symbols), np.imag(symbols),
                    s=8, color=ACCENT, alpha=0.5)
        ax4.axhline(0, color=BORDER, lw=0.8)
        ax4.axvline(0, color=BORDER, lw=0.8)
        ax4.set_title(f"Constelación  ({mod})")
        ax4.set_xlabel("I"); ax4.set_ylabel("Q"); ax4.grid(True)
        ax4.set_aspect("equal")

        self.fig.tight_layout(pad=2.8)
        self.canvas.draw()

    def _draw_placeholder(self):
        for ax in self.axes.flat:
            ax.text(0.5, 0.5, "Configura y pulsa  ▶  Generar",
                    ha="center", va="center", color=TEXT_SEC,
                    transform=ax.transAxes, fontsize=10)
        self.canvas.draw()


# =============================================================================
#  MÓDULO 3 — PATRÓN DE ANTENA 3D  +  INTERFERENCIA CO-CANAL
# =============================================================================
class AntennaModule(ttk.Frame):
    """Patrón de antena direccional 3D con interferencia co-canal."""

    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame")
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        top = ttk.Frame(self, style="Card.TFrame")
        top.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(18, 6))

        ttk.Label(top, text="PATRÓN DE ANTENA 3D  —  INTERFERENCIA CO-CANAL",
                  style="ModTitle.TLabel").pack(side=tk.LEFT)

        # Controles
        ctrl = ttk.Frame(self, style="Card.TFrame")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        ttk.Label(ctrl, text="Dirección ant.1 (rad)", style="ParamLabel.TLabel"
                  ).grid(row=0, column=0, padx=(0, 4))
        self._var_dir = tk.DoubleVar(value=0.0)
        s1 = ttk.Scale(ctrl, from_=-np.pi, to=np.pi, orient=tk.HORIZONTAL,
                       variable=self._var_dir, length=200,
                       command=lambda _: self._update_plot())
        s1.grid(row=0, column=1, padx=(0, 20))

        ttk.Label(ctrl, text="Offset X ant.2 (m)", style="ParamLabel.TLabel"
                  ).grid(row=0, column=2, padx=(0, 4))
        self._var_off = tk.DoubleVar(value=10.0)
        s2 = ttk.Scale(ctrl, from_=0, to=30, orient=tk.HORIZONTAL,
                       variable=self._var_off, length=200,
                       command=lambda _: self._update_plot())
        s2.grid(row=0, column=3, padx=(0, 20))

        ttk.Label(ctrl, text="Ganancia (factor)", style="ParamLabel.TLabel"
                  ).grid(row=0, column=4, padx=(0, 4))
        self._var_gain = tk.DoubleVar(value=100.0)
        s3 = ttk.Scale(ctrl, from_=20, to=200, orient=tk.HORIZONTAL,
                       variable=self._var_gain, length=140,
                       command=lambda _: self._update_plot())
        s3.grid(row=0, column=5)

        # Canvas + reporte
        content = ttk.Frame(self, style="Card.TFrame")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self.fig = plt.figure(figsize=(7, 4.4))
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rep_frame = ttk.Frame(content, style="Card.TFrame")
        rep_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        ttk.Label(rep_frame, text="REPORTE", style="ParamLabel.TLabel"
                  ).pack(anchor="w")
        self._report = tk.Text(rep_frame, width=32, height=14,
                               bg=BG_DARK, fg=TEXT_PRI,
                               font=("Consolas", 9), relief="flat",
                               insertbackground=ACCENT)
        self._report.pack(fill=tk.BOTH, expand=True)

    def _antenna_pattern(self, X, Y, cx, cy, direction, power=20,
                          gain=100, spread=0.1):
        dist_sq = (X - cx)**2 + (Y - cy)**2
        angle   = np.arctan2(Y - cy, X - cx)
        att     = 10 * np.log10(np.maximum(1, dist_sq))
        ang_g   = gain * np.exp(-spread * (np.abs(angle - direction)**2))
        return power - att + ang_g

    def _update_plot(self, _=None):
        self.ax3d.clear()
        x = np.linspace(-30, 30, 50)
        y = np.linspace(-30, 30, 50)
        X, Y = np.meshgrid(x, y)

        d1   = self._var_dir.get()
        off  = self._var_off.get()
        gain = self._var_gain.get()

        Z1 = self._antenna_pattern(X, Y, 0, 0, d1, gain=gain)
        Z2 = self._antenna_pattern(X, Y, off, 0, d1 + np.pi, gain=gain)
        Zt = 10 * np.log10(10**(Z1/10) + 10**(Z2/10))

        self.ax3d.plot_surface(X, Y, Zt, cmap="plasma", alpha=0.85,
                                linewidth=0, antialiased=True)
        self.ax3d.scatter(0, 0, np.max(Zt)+2,
                          color=ACCENT2, marker="^", s=120, label="Ant-1")
        self.ax3d.scatter(off, 0, np.max(Zt)+2,
                          color=ACCENT,  marker="v", s=120, label="Ant-2")
        self.ax3d.set_title("Interferencia Co-canal")
        self.ax3d.set_xlabel("X (m)")
        self.ax3d.set_ylabel("Y (m)")
        self.ax3d.set_zlabel("Potencia (dBm)")
        self.ax3d.legend(fontsize=8)
        self.canvas.draw()

        # Métricas rápidas
        sir = float(np.max(Z1) - np.max(Z2))
        self._report.delete("1.0", tk.END)
        self._report.insert(tk.END,
            f"─── MÉTRICAS ───────────────\n"
            f"Dir. Ant-1 : {np.degrees(d1):+.1f}°\n"
            f"Offset Ant-2 : {off:.1f} m\n"
            f"Factor ganancia: {gain:.0f}\n\n"
            f"Pot. máx Ant-1 : {np.max(Z1):.1f} dBm\n"
            f"Pot. máx Ant-2 : {np.max(Z2):.1f} dBm\n"
            f"SIR estimado   : {sir:.2f} dB\n\n"
            f"─── NOTAS ──────────────────\n"
            f"Las crestas indican lóbulos\n"
            f"de alta ganancia.\n"
            f"La interferencia es mayor\n"
            f"donde se solapan los lóbulos\n"
            f"de ambas antenas.\n\n"
            f"SIR > 10 dB  →  aceptable\n"
            f"SIR > 20 dB  →  bueno\n"
        )


# =============================================================================
#  MÓDULO 4 — MAPA DE CALOR WiFi
# =============================================================================
class HeatmapModule(ttk.Frame):
    """Mapa de propagación WiFi con modelo log-distance."""

    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame")
        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self, style="Card.TFrame")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(18, 6))

        ttk.Label(ctrl, text="MAPA DE COBERTURA WiFi  —  MODELO LOG-DISTANCE",
                  style="ModTitle.TLabel").pack(side=tk.LEFT)

        row = ttk.Frame(ctrl, style="Card.TFrame")
        row.pack(side=tk.RIGHT)

        params = [
            ("Tamaño grid", "size", "80"),
            ("Exp. pérdida n", "n", "3"),
            ("Potencia Tx (dBm)", "pt", "20"),
        ]
        for i, (lbl, key, default) in enumerate(params):
            ttk.Label(row, text=lbl, style="ParamLabel.TLabel"
                      ).grid(row=0, column=i*2, padx=(12, 4))
            e = ttk.Entry(row, width=6, style="Param.TEntry")
            e.insert(0, default)
            e.grid(row=0, column=i*2+1, padx=(0, 4))
            setattr(self, f"_e_{key}", e)

        ttk.Label(row, text="# Routers", style="ParamLabel.TLabel"
                  ).grid(row=0, column=6, padx=(12, 4))
        self._cb_routers = ttk.Combobox(row, values=["1", "2", "3", "4"],
                                         state="readonly", width=4)
        self._cb_routers.set("1")
        self._cb_routers.grid(row=0, column=7, padx=(0, 14))

        ttk.Button(row, text="▶  Generar", style="Accent.TButton",
                   command=self._generate).grid(row=0, column=8, sticky="ns")

        # Canvas
        self.fig, (self.ax_map, self.ax_cdf) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.tight_layout(pad=2.8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=14, pady=(0, 14))
        self._draw_placeholder()

    def _generate(self):
        size = int(self._e_size.get())
        n    = float(self._e_n.get())
        Pt   = float(self._e_pt.get())
        nr   = int(self._cb_routers.get())

        X, Y = np.meshgrid(np.arange(size), np.arange(size))

        # Posiciones de routers (distribuidos uniformemente)
        routers = []
        step = size // (nr + 1)
        for k in range(1, nr + 1):
            routers.append((step * k, size // 2 + np.random.randint(-5, 6)))

        # Potencia por modelo log-distance  P = Pt - 10·n·log10(d)
        # Combinar con "diversity combining" (máximo de todos los routers)
        P_combined = np.full((size, size), -200.0)
        for rx, ry in routers:
            dist = np.sqrt((X - rx)**2 + (Y - ry)**2)
            P    = Pt - 10 * n * np.log10(dist + 1)
            P_combined = np.maximum(P_combined, P)

        self.ax_map.clear()
        self.ax_cdf.clear()

        im = self.ax_map.imshow(P_combined, origin="lower", cmap="inferno",
                                 vmin=Pt - 60, vmax=Pt)
        plt.colorbar(im, ax=self.ax_map, label="Potencia (dBm)")
        for rx, ry in routers:
            self.ax_map.plot(rx, ry, "w^", ms=10, label="Router")
        self.ax_map.set_title(f"Cobertura WiFi  |  n={n}  |  {nr} router(s)")
        self.ax_map.set_xlabel("X (celdas)")
        self.ax_map.set_ylabel("Y (celdas)")

        # CDF de cobertura
        flat = P_combined.flatten()
        flat_sorted = np.sort(flat)
        cdf = np.arange(len(flat_sorted)) / len(flat_sorted)
        self.ax_cdf.plot(flat_sorted, cdf, color=ACCENT)
        self.ax_cdf.axvline(-70, color=ACCENT2, linestyle="--",
                             label="-70 dBm (mín WiFi)")
        cov_pct = np.mean(flat >= -70) * 100
        self.ax_cdf.set_title(f"CDF de señal  |  Cobertura ≥ −70 dBm: {cov_pct:.1f}%")
        self.ax_cdf.set_xlabel("Potencia (dBm)")
        self.ax_cdf.set_ylabel("CDF")
        self.ax_cdf.legend(fontsize=8)
        self.ax_cdf.grid(True)

        self.fig.tight_layout(pad=2.8)
        self.canvas.draw()

    def _draw_placeholder(self):
        for ax in (self.ax_map, self.ax_cdf):
            ax.text(0.5, 0.5, "Configura y pulsa  ▶  Generar",
                    ha="center", va="center", color=TEXT_SEC,
                    transform=ax.transAxes, fontsize=10)
        self.canvas.draw()


# =============================================================================
#  VENTANA PRINCIPAL
# =============================================================================
class RFChannelSimulator(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("RF Channel Simulator  —  v1.0")
        self.configure(bg=BG_DARK)
        self.geometry("1200x760")
        self.minsize(1000, 660)
        self._setup_styles()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

    # ── ESTILOS ──────────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".",               background=BG_DARK,   foreground=TEXT_PRI,
                         font=("Segoe UI", 10))
        style.configure("Card.TFrame",     background=BG_PANEL)
        style.configure("ModTitle.TLabel", background=BG_PANEL,  foreground=ACCENT,
                         font=("Segoe UI Semibold", 11))
        style.configure("ParamLabel.TLabel", background=BG_PANEL,
                         foreground=TEXT_SEC, font=("Segoe UI", 9))
        style.configure("Param.TEntry",    fieldbackground=BG_CARD, foreground=TEXT_PRI,
                         insertcolor=ACCENT, font=("Consolas", 10))
        style.configure("Accent.TButton",  background=ACCENT,  foreground=BG_DARK,
                         font=("Segoe UI Semibold", 10), padding=(14, 6))
        style.map("Accent.TButton",
                  background=[("active", "#00a8cc"), ("pressed", "#007fa0")])

        style.configure("TNotebook",         background=BG_DARK,   tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",     background=BG_CARD,   foreground=TEXT_SEC,
                         font=("Segoe UI", 10), padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", ACCENT)])

    # ── CABECERA ─────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=54)
        hdr.pack(side=tk.TOP, fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="RF CHANNEL SIMULATOR",
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Segoe UI Semibold", 15, "bold")).pack(side=tk.LEFT, padx=24)

        tk.Label(hdr, text="Rayleigh Fading  ·  OFDM Tx  ·  Antenna Pattern  ·  WiFi Coverage",
                 bg=BG_PANEL, fg=TEXT_SEC,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)

        tk.Label(hdr, text="v1.0  |  Python / NumPy / Matplotlib",
                 bg=BG_PANEL, fg=TEXT_SEC,
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=24)

    # ── TABS ─────────────────────────────────────────────────────────────────
    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        tabs = [
            ("📡  Rayleigh Fading",  RayleighModule),
            ("〰  OFDM Tx",          OFDMModule),
            ("🔺  Patrón Antena",    AntennaModule),
            ("🌡  Cobertura WiFi",   HeatmapModule),
        ]
        for name, Cls in tabs:
            frame = Cls(nb)
            nb.add(frame, text=name)

    # ── BARRA DE ESTADO ──────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_CARD, height=24)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar,
                 text="RF Channel Simulator  ·  Telecomunicaciones  |  Todos los módulos listos",
                 bg=BG_CARD, fg=TEXT_SEC,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=12)
        tk.Label(bar,
                 text="numpy · matplotlib · tkinter",
                 bg=BG_CARD, fg=TEXT_SEC,
                 font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=12)


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = RFChannelSimulator()
    app.mainloop()