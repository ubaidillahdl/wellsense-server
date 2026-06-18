import socket
import time
import random
import sqlite3
import json
import sys
from datetime import datetime

# ============================================================
# KONFIGURASI
# ============================================================
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005
PANJANG_BUFFER = 150
DELAY_ANTAR_KIRIM = 10  # detik
LIMIT_DATA = 20

# ============================================================
# DATABASE PATH
# ============================================================
DB_PATH = "D:/laragon/www/wellsense-monitor/database/database.sqlite"


# ============================================================
# FUNGSI AMBIL DATA
# ============================================================
def get_tokens():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT token_perangkat FROM perangkat")
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"[!] Gagal ambil token: {e}")
        return []


def get_latest_data(limit=LIMIT_DATA):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = """
            SELECT id, raw_ir, raw_red, created_at
            FROM data_kesehatan
            ORDER BY id DESC
            LIMIT ?
        """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()
        rows.reverse()
        return rows
    except Exception as e:
        print(f"[!] Gagal ambil data: {e}")
        return []


def parse_signal(json_str):
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        return []
    except:
        return []


def kirim_data(token, ir_data, red_data):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect((SERVER_IP, SERVER_PORT))

        pesan = f"{token}|"
        for i in range(len(ir_data)):
            pesan += f"{ir_data[i]}:{red_data[i]},"

        pesan += "\n"

        client.send(pesan.encode("utf-8"))
        print(f"[*] Data dikirim untuk token: {token} ({len(ir_data)} poin)")

        feedback = client.recv(1024).decode("utf-8")
        print(f"[*] Feedback: {feedback}")

        client.close()
        return True

    except socket.timeout:
        print("[!] Timeout: Server tidak merespon")
        return False
    except ConnectionRefusedError:
        print("[!] Koneksi ditolak: Pastikan server berjalan")
        return False
    except Exception as e:
        print(f"[!] Error: {e}")
        return False


def main():
    print("=" * 60)
    print("SIMULASI KIRIM DATA KE SERVER (DARI SQLITE)")
    print("=" * 60)

    # ============================================================
    # CARA 1: BACA ARGUMEN DARI COMMAND LINE
    # ============================================================
    # Contoh penggunaan:
    #   python simulate_client.py single WS-866501012348821
    #   python simulate_client.py multiple
    #   python simulate_client.py random
    # ============================================================

    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower()
    else:
        mode = "single"  # default

    # ============================================================
    # TENTUKAN TOKEN
    # ============================================================
    all_tokens = get_tokens()
    if not all_tokens:
        print("[!] Tidak ada token ditemukan di database!")
        return

    if mode == "single" and len(sys.argv) >= 3:
        # Mode 1: TOKEN TERTENTU (dari argumen)
        token_list = [sys.argv[2]]
        print("[*] Mode: SINGLE TOKEN (manual)")
        print(f"[*] Token: {token_list[0]}")

    elif mode == "single":
        # Mode 2: TOKEN PERTAMA SAJA
        token_list = [all_tokens[0]]
        print("[*] Mode: SINGLE TOKEN (pertama dari database)")
        print(f"[*] Token: {token_list[0]}")

    elif mode == "random":
        # Mode 3: RANDOM (acak dari semua token)
        token_list = all_tokens
        print(f"[*] Mode: RANDOM TOKEN (acak dari {len(token_list)} token)")

    else:
        # Mode 4: MULTIPLE (semua token bergantian)
        token_list = all_tokens
        print(f"[*] Mode: MULTIPLE TOKEN (bergantian dari {len(token_list)} token)")

    print("-" * 60)

    # ============================================================
    # AMBIL DATA
    # ============================================================
    data_rows = get_latest_data(LIMIT_DATA)
    if not data_rows:
        print("[!] Tidak ada data ditemukan di database!")
        return

    print(f"[*] Data tersedia: {len(data_rows)} record")
    print("[*] Mengirim data dari yang paling lama ke terbaru...\n")

    idx_data = 0
    idx_token = 0

    while True:
        row = data_rows[idx_data]
        ir_data = parse_signal(row["raw_ir"])
        red_data = parse_signal(row["raw_red"])

        if not ir_data or not red_data:
            print(f"[!] Data ID {row['id']} kosong, skip...")
        else:
            # Pilih token
            if mode == "random":
                token = random.choice(token_list)
            else:
                token = token_list[idx_token % len(token_list)]

            print("-" * 60)
            print(f"[*] Record ID   : {row['id']}")
            print(f"[*] Created at  : {row['created_at']}")
            print(f"[*] Token       : {token}")
            print(f"[*] Jumlah data : {len(ir_data)} poin")
            print("-" * 60)

            kirim_data(token, ir_data, red_data)

        # Update indeks
        idx_data = (idx_data + 1) % len(data_rows)
        if mode != "random":
            idx_token = (idx_token + 1) % len(token_list)

        # Refresh data
        if idx_data == 0:
            print("\n>>> SEMUA DATA TERKIRIM, AMBIL DATA TERBARU LAGI! <<<")
            data_rows = get_latest_data(LIMIT_DATA)
            if not data_rows:
                print("[!] Tidak ada data baru, keluar...")
                break
            data_rows.reverse()
            print(f"[*] Data baru tersedia: {len(data_rows)} record")

        # Tunggu
        print(f"\n[*] Menunggu {DELAY_ANTAR_KIRIM} detik...")
        time.sleep(DELAY_ANTAR_KIRIM)


if __name__ == "__main__":
    main()
