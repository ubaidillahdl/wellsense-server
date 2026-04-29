import socket
import socketserver
import numpy as np


class TCPHandler(socketserver.BaseRequestHandler):
    """
    Handler untuk memproses koneksi TCP dari SIM800C.
    Bertugas menerima data, meminta Engine mengolahnya, dan mengirim feedback.
    """

    data_queue = None
    engine_ref = None
    is_gui = False

    def handle(self):
        """Menangani setiap paket data yang masuk dari SIM800C/Arduino"""
        print(f"\n[*] Koneksi Baru Masuk\t: {self.client_address}")
        try:
            # 1. Terima data mentah (Buffer 16KB untuk jaga-jaga paket besar)
            # self.request.settimeout(5.0)
            self.raw_data = self.request.recv(16384).decode("utf-8", errors="ignore")
            if not self.raw_data:
                return

            # --- HEARTBEAT CHECK ---
            if self.raw_data.strip() == "PING":
                self.request.sendall(b"PONG")
                print("[*] Heartbeat\t\t: PONG dikirim")
                return  # selesai, tidak perlu proses data

            # --- TAMBAHAN: PEMISAHAN TOKEN DAN DATA ---
            # data_parts[0] adalah TOKEN, data_parts[1] adalah SINYAL
            data_parts = self.raw_data.strip().split("|")
            token_perangkat = data_parts[0]
            signal_payload = data_parts[1]

            # 2. Parsing format "IR:RED,IR:RED," menjadi list angka
            clean_data = signal_payload.strip().rstrip(",")
            raw_pairs = clean_data.split(",")

            list_ir, list_red = [], []
            for pair in raw_pairs:
                if ":" in pair:
                    try:
                        p_ir, p_red = pair.split(":")
                        list_ir.append(int(p_ir))
                        list_red.append(int(p_red))
                    except ValueError:
                        continue

            # 3. Validasi: Pastikan list tidak kosong sebelum diproses
            if list_ir and list_red:
                print(f"[*] {len(list_ir)} Data Diterima\t: {token_perangkat}")
                print(f"[*] Standar Deviasi IR\t: {np.std(list_ir):.0f}")
                print(f"[*] Standar Deviasi Red\t: {np.std(list_red):.0f}")

                if self.engine_ref:
                    # Masukkan data ke mesin hitung ANN
                    res = self.engine_ref.process_package(
                        list_red, list_ir, token_perangkat
                    )
                    feedback = self.engine_ref.get_feedback()

                    # A. Feedback ke Arduino/SIM800C
                    if feedback:
                        self.request.sendall(feedback.encode("utf-8"))
                        print(f"[*] Data Baru Terkirim\t: {feedback.strip()}")

                    # B. Logika Pesan Status (Independent: Selalu muncul)
                    if res and res["status"] != "VALID":
                        msg = "Keadaan Sinyal\t: Tidak Stabil"
                        print(f"\r[!] {msg}\n", end="", flush=True)

                    # C. Logika Print Tabel (Hanya jika Headless / Mode Terminal)
                    if not TCPHandler.is_gui and res["status"] == "VALID":
                        # Memanggil fungsi tabel dari engine
                        self.engine_ref._print_table(res["vitals_display"])

                # 4. Kirim data ke Queue untuk divisualisasikan oleh Plotter
                if self.data_queue is not None:
                    self.data_queue.put({"red": list_red, "ir": list_ir})

        except socket.timeout:
            print("[!] Koneksi Baru Masuk\t: Timeout")

        except Exception as e:
            # import traceback

            print(f"[!] Listener Error\t: {e}")
            # traceback.print_exc()

    @staticmethod
    def start_listener(host, port, shared_queue, engine_ref=None):
        """Memulai server TCP dalam mode Threading (Bisa handle banyak koneksi)"""
        TCPHandler.data_queue = shared_queue
        TCPHandler.engine_ref = engine_ref

        # Mengizinkan port digunakan kembali tanpa menunggu OS timeout (Penting saat restart)
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.ThreadingTCPServer((host, port), TCPHandler) as server:
            server.serve_forever()
