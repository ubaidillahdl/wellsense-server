import numpy as np
import statistics
import json
import os
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy import signal


class Processor:
    def __init__(self):
        # --- Konfigurasi Filter Digital (Bandpass 0.5 - 3.5 Hz) ---
        self.fs = 50
        nyq = 0.5 * self.fs
        low = 0.5 / nyq
        high = 3.5 / nyq

        # Inisialisasi koefisien filter Butterworth Orde 2
        self.b_bt, self.a_bt = signal.butter(2, [low, high], btype="bandpass")
        self.k_ref = None

        # --- Inisialisasi Model AI (ANN) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_to_load = ["sbp", "dbp", "hb"]
        self.loaded_models = {}

        for name in models_to_load:
            file_name = f"model_{name}.json"
            model_path = os.path.join(current_dir, "model", file_name)

            try:
                # Muat data parameter dari hasil ekspor MATLAB
                with open(model_path, "r") as f:
                    data_json = json.load(f)

                # Masukkan parameter ke class ANNModel yang general
                self.loaded_models[name] = ANNModel(data_json)
            except Exception as e:
                print(e)
                exit()

        # Shortcut akses model (untuk mempermudah pemanggilan)
        self.model_sbp = self.loaded_models.get("sbp")
        self.model_dbp = self.loaded_models.get("dbp")
        self.model_hb = self.loaded_models.get("hb")

        # --- Inisialisasi untuk masing-masing parameter ---
        # Kita buat satu per satu agar buffer-nya tidak bercampur
        self.hr_filter = MedStabilizer(window_size=5)
        self.spo2_filter = MedStabilizer(window_size=5)
        self.sbp_filter = MedStabilizer(
            window_size=5
        )  # Sistolik biasanya lebih liar, window boleh lebih besar
        self.dbp_filter = MedStabilizer(
            window_size=5
        )  # Sistolik biasanya lebih liar, window boleh lebih besar
        self.hb_filter = MedStabilizer(window_size=5)

    def dc_remover(self, buffer):
        """Tahap 1: Hilangkan offset DC, luruskan kemiringan, dan balik sinyal"""
        data = np.array(buffer, dtype=float)
        dc_val = np.mean(data)

        # Detrend linear untuk meluruskan baseline yang miring
        detrended = signal.detrend(data, type="linear")

        # Invert (* -1) karena sensor MAX3010x puncaknya menghadap ke bawah
        return detrended * -1, dc_val

    def moving_average(self, buffer, window_size=8):
        """Tahap 2: Pembersihan awal noise tajam (spikes)"""
        # Window 14 di 50Hz = 0.2 detik perataan (aman untuk puncak sistolik)
        window = np.ones(window_size) / window_size
        return np.convolve(buffer, window, mode="same")

    def butter_filter(self, buffer):
        """Tahap 3: Filter utama 'Ganas' dengan filtfilt (Zero-Phase)"""
        # filtfilt menjalankan filter maju-mundur agar posisi puncak tidak bergeser (delay 0)
        return signal.filtfilt(self.b_bt, self.a_bt, buffer)

    def apply_spline(self, data_50hz, target_hz=400):
        """Tahap 4: Interpolasi agar visual kurva semulus monitor rumah sakit"""
        # Buat sumbu waktu lama dan baru
        x_old = np.linspace(0, len(data_50hz), len(data_50hz))

        # Hitung jumlah sampel baru (Target 200Hz = 4x lebih rapat dari 50Hz)
        num_samples_new = len(data_50hz) * (target_hz // self.fs)
        x_new = np.linspace(0, len(data_50hz) - 1, num_samples_new)

        # Proses Cubic Spline (Kurva melengkung antar titik)
        cs = CubicSpline(x_old, data_50hz)
        return cs(x_new)

    def _extract_valid_cycles(self, vpg):
        """
        Mengekstraksi segmen detak jantung yang stabil dan kontinu.
        Syarat: Irama konsisten (Temporal) & Antar detak menyambung (Continuity).
        """
        # 1. Cari puncak pada VPG (turunan pertama) sebagai acuan awal detak
        peaks, _ = find_peaks(vpg, distance=self.fs / 2, height=np.mean(vpg))
        cycles = []

        for p in peaks:
            # --- Mencari Titik-Titik Kunci (Zero Crossing pada VPG) ---

            # z1 (Onset): Titik awal saat VPG naik ke positif
            i = p
            while i > 0 and vpg[i] > 0:
                i -= 1
            if i == 0 and vpg[i] > 0:
                continue  # Buang jika terpotong di awal
            z1 = i

            # z2 (PPG Peak): Puncak gelombang asli (saat VPG turun ke nol)
            j = p
            while j < len(vpg) - 1 and vpg[j] > 0:
                j += 1
            if j == len(vpg) - 1 and vpg[j] > 0:
                continue
            z2 = j

            # z3 (Offset): Titik akhir siklus (saat VPG kembali ke positif)
            found_z3 = False
            while j < len(vpg) - 1 and vpg[j] <= 0:
                j += 1
                if vpg[j] >= 0:
                    found_z3 = True
                    break
            z3 = j

            # Filter Morfologi: Urutan indeks harus benar & durasi minimal > 0.3 detik
            if found_z3 and (z1 < p < z2 < z3) and (z3 - z1 > 120):
                cycles.append({"z1": z1, "z2": z2, "z3": z3, "peak_vpg": p})

        # 2. Validasi Kontinuitas: Memastikan detak menyambung (z3_n == z1_n+1)
        continuous_cycles = []
        if len(cycles) >= 2:
            for k in range(len(cycles) - 1):
                # Toleransi sampling 2-3 angka
                if abs(cycles[k]["z3"] - cycles[k + 1]["z1"]) <= 3:
                    if not continuous_cycles or continuous_cycles[-1] != cycles[k]:
                        continuous_cycles.append(cycles[k])
                    continuous_cycles.append(cycles[k + 1])

        # 3. Validasi Temporal: Cek stabilitas irama (Coefficient of Variation)
        # Jika irama paket 3 detik ini stabil, kembalikan siklusnya
        if self.validate_temporal_consistency(continuous_cycles):
            return continuous_cycles
        else:
            # Jika noise/irama berantakan, kembalikan list kosong
            return []

    def validate_temporal_consistency(self, valid_cycles):
        """
        Memvalidasi konsistensi durasi antar siklus.
        Hanya fokus pada domain waktu (sampel/indeks).
        """
        # Minimal 2-4 siklus untuk pengecekan pola interval
        if len(valid_cycles) < 2 or len(valid_cycles) > 4:
            return False

        z1_intervals = []
        z2_intervals = []
        z3_intervals = []
        cycle_durations = []
        has_good_ratio = False  # Flag pengunci: minimal harus ada 1 siklus miring kanan

        for i in range(len(valid_cycles)):
            # --- 1. FILTER MORFOLOGI (Symmetry Check) ---
            len_sys = valid_cycles[i]["z2"] - valid_cycles[i]["z1"]  # Waktu naik
            len_dias = valid_cycles[i]["z3"] - valid_cycles[i]["z2"]  # Waktu turun

            # Syarat: Sisi turun (diastolik) harus lebih panjang dari sisi naik (systolik)
            if len_sys > 0 and (len_dias / len_sys) > 1.1:
                has_good_ratio = True  # Paket data dianggap punya potensi siklus valid

            # --- 2. KOLEKSI DATA DURASI ---
            # Durasi total satu detak (Onset ke Offset berikutnya)
            cycle_durations.append(valid_cycles[i]["z3"] - valid_cycles[i]["z1"])

            # Interval antar titik (Peak-to-Peak, Onset-to-Onset, dsb)
            if i > 0:
                z1_intervals.append(valid_cycles[i]["z1"] - valid_cycles[i - 1]["z1"])
                z2_intervals.append(valid_cycles[i]["z2"] - valid_cycles[i - 1]["z2"])
                z3_intervals.append(valid_cycles[i]["z3"] - valid_cycles[i - 1]["z3"])

        # --- 3. REJECT JIKA MORFOLOGI CACAT ---
        # Jika dari 3-4 detak tidak ada satupun yang miring kanan, reject total
        if not has_good_ratio:
            return False

        # --- 4. HITUNG VARIABILITAS (Coefficient of Variation) ---
        # CV = Std / Mean. Digunakan untuk mendeteksi ketidakteraturan (misal: aritmia atau noise)
        cv_z1 = np.std(z1_intervals) / np.mean(z1_intervals)
        cv_z2 = np.std(z2_intervals) / np.mean(z2_intervals)
        cv_z3 = np.std(z3_intervals) / np.mean(z3_intervals)
        cv_duration = np.std(cycle_durations) / np.mean(cycle_durations)

        # Threshold Stabilitas (15%). Jika variasi interval > 15%, dianggap sinyal goyang/tidak stabil
        tresh = 0.15
        is_stable = (
            (cv_z1 < tresh)
            and (cv_z2 < tresh)
            and (cv_z3 < tresh)
            and (cv_duration < tresh)
        )

        return is_stable

    def get_best_cycle_by_sqi(self, ppg_raw, cycles):
        """Pilih satu siklus terbaik dengan prioritas Morfologi (Ratio) & Kualitas Sinyal (SQI)"""
        if not cycles:
            return None

        results = []
        IDEAL_SKEW = 0.5  # Target kemiringan (positif = miring kanan)
        IDEAL_KURT = 3.0  # Target keruncingan (normal distribution)
        IDEAL_RATIO = 1.6  # Target Diastolik ~1.6x Systolik

        # Pengali bobot: Semakin besar, semakin wajib rasionya mendekati 1.6
        WEIGHT_RATIO = 5.0

        for cyc in cycles:
            segment = ppg_raw[cyc["z1"] : cyc["z3"]]
            if len(segment) < 20:
                continue  # Skip jika segmen terlalu pendek

            # --- Hitung Rasio Morfologi (Symmetry) ---
            len_sys = cyc["z2"] - cyc["z1"]
            len_dias = cyc["z3"] - cyc["z2"]
            ratio = len_dias / len_sys if len_sys > 0 else 0

            # --- Hitung Statistik SQI ---
            s = skew(segment)
            k = kurtosis(segment)

            # --- Perhitungan Skor (Penalty System) ---
            if ratio < 1.1:
                # REJECT: Jika ratio < 1.1 (miring kiri/tegak), beri skor sampah
                score = 1000.0
            else:
                # Prioritas 1: Selisih Ratio (dikali bobot)
                diff_ratio = abs(ratio - IDEAL_RATIO)
                # Prioritas 2: Selisih Skewness & Kurtosis
                diff_skew = abs(s - IDEAL_SKEW)
                diff_kurt = abs(k - IDEAL_KURT)

                # Gabungkan: Nilai terkecil = Pemenang
                score = (WEIGHT_RATIO * diff_ratio) + diff_skew + diff_kurt

            # Simpan fitur hasil ekstraksi ke objek cycle
            new_data = cyc.copy()
            new_data.update({"skew": s, "kurt": k, "ratio": ratio, "score": score})
            results.append(new_data)

        if not results:
            return None

        # Eksekusi: Ambil siklus dengan deviasi terendah terhadap kriteria ideal
        best = min(results, key=lambda x: x["score"])
        return best

    def straighten_cycle(self, cycle_segment):
        """Meluruskan kemiringan baseline pada satu siklus"""
        n = len(cycle_segment)

        # Ambil nilai awal dan akhir
        y_start = cycle_segment[0]
        y_end = cycle_segment[-1]

        # Buat garis lurus (linear trend) dari titik awal ke akhir
        trend = np.linspace(y_start, y_end, n)

        # Kurangi sinyal asli dengan trend tersebut
        # Lalu angkat agar mulai dari nol (offset)
        straight_cycle = cycle_segment - trend

        return straight_cycle

    def extract_w1_w2(self, cycles_ppg, t_ppg_segment, fs):
        """
        Konversi 1:1 dari Matlab extractW1W2.
        cycles_ppg: segmen satu siklus (straighten_cycle)
        t_ppg_segment: sumbu waktu segmen tersebut (t_cycle)
        """
        n = len(cycles_ppg)
        if n < 2:
            return None

        # 1. Hitung Amplitudo
        max_val = np.max(cycles_ppg)
        min_val = np.min(cycles_ppg)
        max_idx = np.argmax(cycles_ppg)
        amplitude = max_val - min_val

        val_w1 = min_val + 0.5 * amplitude
        val_w2 = min_val + 0.66 * amplitude

        # 2. Titik W1 (Cari indeks potong kiri & kanan)
        # idxLeftW1: find(cyclesPPG(1:maxIdx) >= valW1, 1, 'first')
        left_w1_search = np.where(cycles_ppg[: max_idx + 1] >= val_w1)[0]
        idx_left_w1 = left_w1_search[0] if len(left_w1_search) > 0 else 0

        # idxRightW1: maxIdx - 1 + find(cyclesPPG(maxIdx:end) <= valW1, 1, 'first')
        right_w1_search = np.where(cycles_ppg[max_idx:] <= val_w1)[0]
        idx_right_w1 = (
            (right_w1_search[0] + max_idx) if len(right_w1_search) > 0 else n - 1
        )

        # 3. Titik W2
        left_w2_search = np.where(cycles_ppg[: max_idx + 1] >= val_w2)[0]
        idx_left_w2 = left_w2_search[0] if len(left_w2_search) > 0 else 0

        right_w2_search = np.where(cycles_ppg[max_idx:] <= val_w2)[0]
        idx_right_w2 = (
            (right_w2_search[0] + max_idx) if len(right_w2_search) > 0 else n - 1
        )

        # 4. Hitung Waktu (Time Tracking)
        # Karena kita sudah punya t_ppg_segment (t_cycle), kita ambil langsung
        t_left_w1 = t_ppg_segment[idx_left_w1]
        t_right_w1 = t_ppg_segment[idx_right_w1]
        t_left_w2 = t_ppg_segment[idx_left_w2]
        t_right_w2 = t_ppg_segment[idx_right_w2]

        # Durasi dan Lebar
        cycle_duration = t_ppg_segment[-1] - t_ppg_segment[0]
        width_w1 = (idx_right_w1 - idx_left_w1) / fs
        width_w2 = (idx_right_w2 - idx_left_w2) / fs

        # --- OUTPUT (Sesuai Struktur Matlab) ---
        out = {}
        out["tLeftW1"] = t_left_w1
        out["tRightW1"] = t_right_w1
        out["tLeftW2"] = t_left_w2
        out["tRightW2"] = t_right_w2
        out["idxLeftW1"] = idx_left_w1
        out["idxLeftW2"] = idx_left_w2
        out["idxRightW1"] = idx_right_w1
        out["idxRightW2"] = idx_right_w2
        out["valW1"] = val_w1
        out["valW2"] = val_w2

        out["cycleDuration"] = cycle_duration
        out["widthW1"] = width_w1
        out["widthW2"] = width_w2
        out["w1W2Ratio"] = width_w1 / width_w2 if width_w2 > 0 else 0
        out["w1centerTime"] = (idx_left_w1 + idx_right_w1) / (2 * fs)  # Center point
        out["relWidthW1"] = width_w1 / cycle_duration if cycle_duration > 0 else 0
        out["relWidthW2"] = width_w2 / cycle_duration if cycle_duration > 0 else 0
        out["heartRate"] = 60 / cycle_duration if cycle_duration > 0 else 0
        out["sysTime"] = t_ppg_segment[max_idx] - t_ppg_segment[0]
        out["diasTime"] = t_ppg_segment[-1] - t_ppg_segment[max_idx]

        return out

    def calculate_spo2(self, ac_ir, dc_ir, ac_red, dc_red):
        """
        Menghitung SpO2 secara instan (stateless) tanpa memori k_ref.
        Fokus pada kejujuran pembacaan sensor per frame.
        """
        try:
            # 1. Validasi Dasar
            if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0:
                return 0

            # 2. Hitung Indeks Modulasi (Rasio AC/DC)
            # Menghilangkan faktor ketebalan jaringan
            m_ir = ac_ir / dc_ir
            m_red = ac_red / dc_red

            # 3. Hitung Rasio R (R-Curve)
            # R = (AC_red / DC_red) / (AC_ir / DC_ir)
            R = m_red / m_ir
            # print(f"DEBUG SpO2 -> R: {R:.4f} | m_red: {m_red:.4f} | m_ir: {m_ir:.4f}")

            # 4. Rumus SpO2 (Empiris Standar)
            # Kita gunakan koefisien standar industri: 110 - 25*R
            # Jika hasil ingin lebih melengkung/akurat, gunakan kuadratik:
            # spo2 = 104 - 17*R - 9*(R**2)
            # spo2 = 104 - (28 * R)
            spo2 = (-45.060 * (R**2)) + (30.354 * R) + 94.845

            # 5. Boundary Control
            # Memastikan tidak ada angka 'ajaib' di luar batas fisiologis
            if spo2 > 100:
                spo2 = 100.0
            if spo2 < 70:
                spo2 = 70.0  # Batas bawah oximeter pada umumnya

            return round(float(spo2), 2)

        except ZeroDivisionError:
            return 0
        except Exception:
            return 0

    def reset_all_filters(self):
        """Fungsi untuk membersihkan semua buffer sekaligus"""
        self.hr_filter.reset()
        self.spo2_filter.reset()
        self.sbp_filter.reset()
        self.dbp_filter.reset()
        self.hb_filter.reset()


class ANNModel:
    def __init__(self, data_json):
        # 1. Bobot & Bias (Pastikan tipe data float)
        self.W1 = np.array(data_json["W1"], dtype=float)
        self.b1 = np.array(data_json["b1"], dtype=float).reshape(-1, 1)
        self.W2 = np.array(data_json["W2"], dtype=float).reshape(1, -1)
        self.b2 = np.array(data_json["b2"], dtype=float).flatten()[
            0
        ]  # Ambil sebagai skalar

        # 2. Parameter Z-Score (Dari CSV Mentah)
        self.mean_in = np.array(data_json["mean_in"], dtype=float).reshape(-1, 1)
        self.std_in = np.array(data_json["std_in"], dtype=float).reshape(-1, 1)

        # 3. Parameter MapMinMax (Input)
        self.xmin_in = np.array(data_json["norm_in"]["xmin"], dtype=float).reshape(
            -1, 1
        )
        self.xmax_in = np.array(data_json["norm_in"]["xmax"], dtype=float).reshape(
            -1, 1
        )
        self.ymin_in = float(data_json["norm_in"]["ymin"])
        self.ymax_in = float(data_json["norm_in"]["ymax"])

        # 4. Parameter Denormalisasi (Output)
        self.xmin_out = float(data_json["norm_out"]["xmin"])
        self.xmax_out = float(data_json["norm_out"]["xmax"])
        self.ymin_out = float(data_json["norm_out"]["ymin"])
        self.ymax_out = float(data_json["norm_out"]["ymax"])

    def tansig(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1

    def calculate(self, features):
        # Konversi input ke kolom vektor (5x1)
        x_raw = np.array(features, dtype=float).reshape(-1, 1)

        # --- STEP 1: Z-SCORE ---
        # Mengubah data mentah (misal HR 80) ke skala Z-score
        x_manual = (x_raw - self.mean_in) / self.std_in

        # --- STEP 2: MAPMINMAX (Input) ---
        # Mengubah hasil Z-score ke rentang ymin_in s/d ymax_in (biasanya -1 ke 1)
        x_norm = (self.ymax_in - self.ymin_in) * (x_manual - self.xmin_in) / (
            self.xmax_in - self.xmin_in
        ) + self.ymin_in

        # --- STEP 3: ARSITEKTUR JARINGAN ---
        # Hidden Layer
        z1 = np.dot(self.W1, x_norm) + self.b1
        a1 = self.tansig(z1)

        # Output Layer (Linear/Purelin)
        a2 = np.dot(self.W2, a1) + self.b2

        # --- STEP 4: REVERSE MAPMINMAX (Output) ---
        # Mengembalikan angka -1 s/d 1 ke skala asli (misal SBP 120)
        prediction = (a2 - self.ymin_out) * (self.xmax_out - self.xmin_out) / (
            self.ymax_out - self.ymin_out
        ) + self.xmin_out

        return round(float(prediction), 4)


class MedStabilizer:
    def __init__(self, window_size=5):
        self.buffer = []
        self.window_size = window_size

    def get_stable_value(self, new_value):
        # 1. Masukkan data baru
        self.buffer.append(new_value)

        # 2. Jaga ukuran jendela (window) agar tetap (misal: 5 data terakhir)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # 3. Hitung Median (Ini otomatis mengurutkan dan ambil tengah)
        # Menggunakan library statistics bawaan Python
        return statistics.median(self.buffer)

    def reset(self):
        """Mengosongkan memori filter"""
        self.buffer = []
        # print("[System] Filter buffer cleared.")
