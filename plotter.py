import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import queue


class Plotter:
    def __init__(self, shared_queue, engine_instance):
        self.data_queue = shared_queue
        self.engine = engine_instance

        # 1. Setup Canvas (2 Baris: Atas untuk Monitor, Bawah untuk Analisis)
        self.fig, self.axes = plt.subplots(2, 1)

        # 2. Inisialisasi Objek Grafis (Artists)
        (self.line_ppg,) = self.axes[0].plot([], [])
        (self.line_cycle,) = self.axes[1].plot([], [])
        (self.ppg_markers,) = self.axes[0].plot([], [], "ko")
        (self.dot_points,) = self.axes[1].plot([], [], "go")
        (self.plot_w1,) = self.axes[1].plot([], [], "g--", alpha=0.6)
        (self.plot_w2,) = self.axes[1].plot([], [], "g--", alpha=0.6)

        self._setup_axes()

        # List untuk optimasi blitting (gambar hanya bagian yang berubah)
        self.all_artists = [
            self.line_ppg,
            self.ppg_markers,
            self.line_cycle,
            self.dot_points,
            self.plot_w1,
            self.plot_w2,
        ]

    def _setup_axes(self):
        """Konfigurasi style visual tiap axis"""
        titles = ["PPG Global Monitor", "Best Cycle Analysis"]
        for i, ax in enumerate(self.axes):
            ax.set_title(titles[i])
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_facecolor("#f0f0f0")
        plt.tight_layout()

    def update_graph(self, frame):
        """Looping utama penggambaran grafik (dipanggil per interval)"""

        # 1. Kuras Queue agar tidak menumpuk di memori
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        # 2. Ambil data terbaru dari Engine (Read-Once)
        res = self.engine.get_last_result()

        if not res:
            return self.all_artists

        # 3. Update Grafik Atas (Selalu update sinyal mentah)
        self.line_ppg.set_data(res["t_ppg"], res["ppg_ir"])
        self._auto_scale_axis(self.axes[0], res["t_ppg"], res["ppg_ir"])

        # 4. Logika Update Grafik Bawah (Hanya jika sinyal VALI
        if res["status"] == "VALID" and res["best_cycle"]:
            bc = res["best_cycle"]
            w = bc["w_out"]

            # Update titik puncak & garis deteksi siklus
            self.ppg_markers.set_data(
                bc["global_markers"]["x"], bc["global_markers"]["y"]
            )
            self.line_cycle.set_data(bc["time"], bc["signal"])
            self.dot_points.set_data(bc["z_points"]["x"], bc["z_points"]["y"])

            # Gambar garis lebar pulsa W1 & W2
            self.plot_w1.set_data(
                [bc["time"][w["idxLeftW1"]], bc["time"][w["idxRightW1"]]],
                [w["valW1"], w["valW1"]],
            )
            self.plot_w2.set_data(
                [bc["time"][w["idxLeftW2"]], bc["time"][w["idxRightW2"]]],
                [w["valW2"], w["valW2"]],
            )

            self._auto_scale_axis(self.axes[1], bc["time"], bc["signal"])
            self.axes[1].set_title(f"Best Cycle (Skew: {bc['skew']:.2f})")

            # Tampilkan tabel hasil di terminal
            v = res["vitals_display"]
            self.engine._print_table(v)

        else:
            # Bersihkan layar bawah jika jari lepas atau sinyal berisik
            self._clear_sub_plots()

        self.fig.canvas.draw_idle()
        return self.all_artists

    def _clear_sub_plots(self):
        """Menghapus data visual pada plot analisis bawah"""
        for artist in [
            self.ppg_markers,
            self.line_cycle,
            self.dot_points,
            self.plot_w1,
            self.plot_w2,
        ]:
            artist.set_data([], [])

    def _auto_scale_axis(self, ax, x, y):
        """Menyesuaikan batas sumbu X dan Y secara otomatis"""
        if len(y) > 0:
            ymin, ymax = np.min(y), np.max(y)
            margin = (ymax - ymin) * 0.2 if ymax != ymin else 1.0
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(ymin - margin, ymax + margin)

    def show(self):
        """Menampilkan window grafik dan memulai animasi"""
        self.ani = FuncAnimation(
            self.fig, self.update_graph, blit=True, interval=50, cache_frame_data=False
        )
        plt.show()
