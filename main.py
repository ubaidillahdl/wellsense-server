import threading
import queue
import sys
from listener import TCPHandler
from plotter import Plotter
from vital_engine import VitalEngine

if __name__ == "__main__":
    # 1. Tanya Pengguna di Terminal
    # pilihan = input("\n[?] Jalankan dengan Grafik? (y/n): ").lower().strip()
    pilihan = "n"

    # 2. Inisialisasi Resource Utama
    shared_queue = queue.Queue()  # Penampung data untuk dikirim ke Plotter
    engine = VitalEngine()  # Mesin pengolah filter & ANN
    HOST, PORT = "0.0.0.0", 5005  # Konfigurasi server TCP

    # 3. Sinkronisasi Mode (Mencegah Double Print Tabel)
    # Variabel is_gui dititipkan ke TCPHandler agar ia tahu kapan harus 'diam'
    TCPHandler.is_gui = pilihan == "y"

    if TCPHandler.is_gui:
        # --- MODE GUI (Grafik Aktif) ---
        print(f"[*] MODE: GUI | Server: {HOST}:{PORT}")

        plotter = Plotter(shared_queue, engine)

        # Threading: Listener dipindah ke belakang (background)
        # agar tidak membekukan (freeze) jendela grafik Matplotlib
        t1 = threading.Thread(
            target=TCPHandler.start_listener,
            args=(HOST, PORT, shared_queue, engine),
            daemon=True,  # Mati otomatis saat jendela ditutup
        )
        t1.start()

        # Matplotlib wajib berjalan di Main Thread
        plotter.show()

    else:
        # --- MODE HEADLESS (Terminal Only) ---
        print(f"[*] MODE: HEADLESS | Server: {HOST}:{PORT}")

        # Di mode ini, Listener mengambil alih Main Thread (Running forever)
        try:
            TCPHandler.start_listener(HOST, PORT, shared_queue, engine)
        except KeyboardInterrupt:
            # Menangani penutupan paksa via Ctrl+C agar lebih rapi
            print("\n[*] Memberhentikan sistem...")
            sys.exit(0)
