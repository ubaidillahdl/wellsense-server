import numpy as np
from processor import Processor
from database import WellSenseDB


class VitalEngine:
    def __init__(self):
        self.processor = Processor()
        self.fs = 400
        self.feedback_str = ""
        self.last_result = None
        self.new_data_available = False  # Penanda untuk Arduino (Feedback)
        self.graph_ready = False  # Penanda untuk Plotter (Grafik)
        self.db = WellSenseDB()

    def process_package(self, red_data, ir_data, token_perangkat=None):
        """Alur: Pre-processing -> Gatekeeper -> ANN Analysis"""
        if not ir_data or not token_perangkat:
            print("[!] Data atau Token kosong!")
            return None

        # --- 0. VALIDASI TOKEN (PINTU UTAMA) ---
        # Cari tahu siapa pemilik token ini di database Laravel
        device_info = self.db.get_device_info(token_perangkat)

        if not device_info:
            print(f"[!] Akses Ditolak: Token {token_perangkat} tidak terdaftar!")
            return None

        # --- 1. PRE-PROCESSING (Sinyal Mentah) ---
        ac_ir, v_dc_ir = self.processor.dc_remover(ir_data)
        filtered_ir = self.processor.butter_filter(self.processor.moving_average(ac_ir))
        ppg_ir = self.processor.apply_spline(filtered_ir, target_hz=self.fs)

        ac_red, v_dc_red = self.processor.dc_remover(red_data)
        filtered_red = self.processor.butter_filter(
            self.processor.moving_average(ac_red)
        )
        ppg_red = self.processor.apply_spline(filtered_red, target_hz=self.fs)

        t_ppg = np.linspace(0, len(ppg_ir) / self.fs, len(ppg_ir))
        std_val = np.std(ir_data)

        # Inisialisasi output dasar
        output = {
            "status": "IDLE",
            "ppg_ir": ppg_ir,
            "t_ppg": t_ppg,
            "best_cycle": None,
            "vitals_display": None,
        }

        # Simpan state awal agar plotter tetap jalan
        self.last_result = output
        self.graph_ready = True

        self.feedback_str = f"*0;0;0;0;0;{int(round(std_val))}#\n"
        self.new_data_available = True

        # --- 2. GATEKEEPER (Cek Sensor) ---
        if std_val > 150:
            output["status"] = "NOISY"
            self.last_result = output
            return output

        # --- 3. ANALISIS SIKLUS & ANN ---
        v_ac_ir = np.max(ppg_ir) - np.min(ppg_ir)
        v_ac_red = np.max(ppg_red) - np.min(ppg_red)
        vpg = np.gradient(ppg_ir, 1 / self.fs)
        valid_cycles = self.processor._extract_valid_cycles(vpg)

        if valid_cycles:
            best_cycle = self.processor.get_best_cycle_by_sqi(ppg_ir, valid_cycles)
            if best_cycle:
                start, end = best_cycle["z1"], best_cycle["z3"]
                cycle = self.processor.straighten_cycle(ppg_ir[start:end])
                t_cycle = np.arange(len(cycle)) / self.fs
                w_out = self.processor.extract_w1_w2(cycle, t_cycle, self.fs)

                if w_out:
                    # Hitung parameter vital
                    spo2 = self.processor.calculate_spo2(
                        v_ac_ir, v_dc_ir, v_ac_red, v_dc_red
                    )
                    sbp = self.processor.model_sbp.calculate(
                        [
                            w_out["sysTime"],
                            w_out["w1W2Ratio"],
                            w_out["heartRate"],
                            w_out["diasTime"],
                            w_out["cycleDuration"],
                        ]
                    )
                    dbp = self.processor.model_dbp.calculate(
                        [
                            w_out["w1W2Ratio"],
                            w_out["sysTime"],
                            w_out["relWidthW1"],
                            w_out["heartRate"],
                            w_out["diasTime"],
                        ]
                    )
                    hb = self.processor.model_hb.calculate(
                        [
                            w_out["sysTime"],
                            w_out["w1W2Ratio"],
                            w_out["diasTime"],
                            w_out["relWidthW2"],
                            w_out["widthW2"],
                        ]
                    )

                    # Filter angka agar stabil
                    s_hr = self.processor.hr_filter.get_stable_value(w_out["heartRate"])
                    s_spo2 = self.processor.spo2_filter.get_stable_value(spo2)
                    s_sbp = self.processor.sbp_filter.get_stable_value(sbp)
                    s_dbp = self.processor.dbp_filter.get_stable_value(dbp)
                    s_hb = self.processor.hb_filter.get_stable_value(hb) * 10  # g/L

                    # --- INTEGRASI KE DATABASE MILIK ABANG ---
                    # Bungkus ke dictionary sesuai kebutuhan save_health_data
                    vitals_dict = {
                        "hr": s_hr,
                        "spo2": s_spo2,
                        "sbp": s_sbp,
                        "dbp": s_dbp,
                        "hb": s_hb,
                    }

                    # Ubah array sinyal ke string agar muat di LONGTEXT
                    signals_dict = {
                        # Kita bungkus dulu dengan np.array() baru panggil .tolist()
                        "raw_ir": np.array(ir_data).tolist(),
                        "raw_red": np.array(red_data).tolist(),
                        "filtered_ir": filtered_ir.tolist(),
                        "filtered_red": filtered_red.tolist(),
                    }

                    # Features untuk pemetaan titik di Dashboard Web
                    features_dict = {
                        "z_points": [
                            int(best_cycle["z1"]),
                            int(best_cycle["z2"]),
                            int(best_cycle["z3"]),
                        ],
                        "w_metadata": {k: float(v) for k, v in w_out.items()},
                        "quality_metrics": {
                            "score": float(best_cycle.get("score", 0)),
                            "skew": float(best_cycle.get("skew", 0)),
                            "kurt": float(best_cycle.get("kurt", 0)),
                            "ratio": float(best_cycle.get("ratio", 0)),
                        },
                    }

                    self.db.save_health_data(
                        device_info,
                        token_perangkat,
                        vitals_dict,
                        signals_dict,
                        features_dict,
                    )

                    # Siapkan feedback untuk Arduino via Listener
                    self.feedback_str = f"*{int(round(s_hr))};{int(round(s_spo2))};{int(round(s_sbp))};{int(round(s_dbp))};{int(round(s_hb))};{int(round(std_val))}#\n"
                    self.new_data_available = True

                    # Update output lengkap untuk Plotter
                    output["status"] = "VALID"
                    output["vitals_display"] = [s_hr, s_spo2, s_sbp, s_dbp, s_hb]
                    output["best_cycle"] = {
                        "signal": cycle,
                        "time": t_cycle,
                        "w_out": w_out,
                        "z_points": {
                            "x": [
                                t_cycle[best_cycle["z1"] - start],
                                t_cycle[best_cycle["z2"] - start],
                                t_cycle[best_cycle["z3"] - start - 1],
                            ],
                            "y": [
                                cycle[best_cycle["z1"] - start],
                                cycle[best_cycle["z2"] - start],
                                cycle[best_cycle["z3"] - start - 1],
                            ],
                        },
                        "global_markers": {
                            "x": [
                                t_ppg[best_cycle["z1"]],
                                t_ppg[best_cycle["z2"]],
                                t_ppg[best_cycle["z3"]],
                            ],
                            "y": [
                                ppg_ir[best_cycle["z1"]],
                                ppg_ir[best_cycle["z2"]],
                                ppg_ir[best_cycle["z3"]],
                            ],
                        },
                        "skew": best_cycle["skew"],
                    }

        # Simpan hasil akhir & nyalakan lampu hijau buat Plotter
        self.last_result = output
        self.graph_ready = True
        return output

    def get_feedback(self):
        """Diambil oleh Listener untuk feedback ke hardware"""
        if self.new_data_available:
            self.new_data_available = False
            return self.feedback_str
        return None

    def get_last_result(self):
        """Diambil oleh Plotter untuk update visualisasi"""
        if self.graph_ready:
            self.graph_ready = False
            return self.last_result
        return None

    def _print_table(self, v):
        """Mencetak data vital ke terminal dengan format tabel ASCII"""
        print("\n┌───────────────────────────────────────────┐")
        print("│      ANALISA VITAL SIGN REAL-TIME         │")
        print("├─────────────────────┬─────────────────────┤")
        print(f"│ HEART RATE          │  {v[0]:>8.2f} bpm       │")
        print(f"│ SPO2 (OXYGEN)       │  {v[1]:>8.2f} %         │")
        print("├─────────────────────┼─────────────────────┤")
        print(f"│ SYSTOLIC PRESSURE   │  {v[2]:>8.4f} mmHg      │")
        print(f"│ DIASTOLIC PRESSURE  │  {v[3]:>8.4f} mmHg      │")
        print("├─────────────────────┼─────────────────────┤")
        print(f"│ HEMOGLOBIN (Hb)     │  {v[4]:>8.4f} g/L       │")
        print("└─────────────────────┴─────────────────────┘")
