import sqlite3
import json
import requests
from datetime import datetime


class WellSenseDB:
    def __init__(self):
        # Sesuaikan path dengan lokasi database.sqlite Laravel di VPS nanti
        self.db_path = "D:/laragon/www/wellsense/database/database.sqlite"

    def get_connection(self):
        """SQLite tidak butuh maintain koneksi seperti MySQL,
        lebih aman buka-tutup setiap transaksi."""
        return sqlite3.connect(self.db_path)

    def triger_dashboard(self, token):
        """Mengirim sinyal ke Laravel API untuk update dashboard"""
        url = "http://192.168.0.105/api/v1/send-health-data"
        payload = {"token_perangkat": str(token)}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                print("[+] Pemicu Dashboard\t: Terkirim")
            else:
                print(f"[!] Web server merespon status {response.status_code}")
        except Exception as e:
            print(f"[!] Web server error (Koneksi): {e}")

    def get_device_info(self, token):
        try:
            conn = self.get_connection()
            # Row_factory membuat hasil query bisa dipanggil pakai nama kolom (seperti dictionary)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT id, pengguna_id FROM perangkat WHERE token_perangkat = ?"
            cursor.execute(query, (token,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Exception as e:
            print(f"[!] Lookup Error: {e}")
            return None

    def save_health_data(
        self, device_info, token_perangkat, data_vitals, signals, features
    ):
        """Menyimpan hasil olahan ANN ke database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            query = """
                INSERT INTO data_kesehatan 
                (
                    pengguna_id, perangkat_id, 
                    raw_ir, raw_red, 
                    filtered_ir, filtered_red, 
                    features,
                    hr, spo2, sbp, dbp, hb, 
                    created_at, updated_at
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

            # Kita konversi list signal ke JSON string agar masuk ke longText
            values = (
                device_info["pengguna_id"],
                device_info["id"],
                json.dumps(signals.get("raw_ir", [])),
                json.dumps(signals.get("raw_red", [])),
                json.dumps(signals.get("filtered_ir", [])),
                json.dumps(signals.get("filtered_red", [])),
                json.dumps(features),
                data_vitals.get("hr"),
                data_vitals.get("spo2"),
                data_vitals.get("sbp"),
                data_vitals.get("dbp"),
                data_vitals.get("hb"),
                waktu_sekarang,
                waktu_sekarang,
            )

            cursor.execute(query, values)
            conn.commit()
            conn.close()
            print("[+] Simpan Ke Database\t: Berhasil")

            # --- EKSEKUSI PATH 2 ---
            # Dipanggil otomatis tepat setelah data masuk database
            self.triger_dashboard(token_perangkat)

        except Exception as e:
            print(f"[!] Simpan Ke Database\t: {e}")
