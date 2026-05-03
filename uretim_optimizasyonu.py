"""
=============================================================================
  FABRİKA ÜRETİM ÇİZELGESİ OPTİMİZASYONU  —  v3.0  (Sensör Entegrasyonu)
  Google OR-Tools CP-SAT  |  dataset.json  |  Arıza Kısıtları
  Fizibilite Yönetimi     |  FastAPI Hazır JSON Çıktısı
=============================================================================

  Bağımlılıklar  :  pip install ortools pandas fastapi uvicorn

  Doğrudan çalıştırma:
      python uretim_optimizasyonu.py

  FastAPI entegrasyonu:
      from uretim_optimizasyonu import (
          optimize_schedule, oku_fiyatlar_csv, oku_dataset_json,
          EPIAS_CSV_YOL, SENSOR_MOCK_DATA
      )
=============================================================================
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import pandas as pd
from ortools.sat.python import cp_model


# ============================================================================
# DOSYA YOLLARI
# ============================================================================
EPIAS_CSV_YOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Piyasa_Takas_Fiyati-02042026-02052026.csv")
DATASET_JSON_YOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dataset.json")

# ============================================================================
# SENSÖR SİMÜLASYONU  (Jüri Şov — Predictive Maintenance Mock)
# ============================================================================
# Gerçek sensör API'si geldiğinde bu dict yerine API çağrısı yapılır.
SENSOR_MOCK_DATA: dict[str, dict] = {
    "CEL-001": {
        "Failure_Within_7_Days": False,  # Geçici olarak çalışır duruma alındı (Jüri sunumu için True yapılabilir)
        "Remaining_Useful_Life_days": 45,
        "sensor_notu": "Titreşim seviyeleri normal aralıkta.",
    },
    "OTO-003": {
        "Failure_Within_7_Days": True,  # Jüri sunumu için arızalı senaryosu aktif
        "Remaining_Useful_Life_days": 2,
        "sensor_notu": "Dinamometre yatak sıcaklığı kritik seviyede.",
    },
}

# RUL eşik değeri (gün) — bu değerin altındaki makineler devre dışı
RUL_ESIK_GUN = 5


# ============================================================================
# DATASET.JSON OKUYUCU
# ============================================================================

def oku_dataset_json(yol: str = DATASET_JSON_YOL) -> list[dict]:
    """
    dataset.json dosyasını okur ve her makineyi iş olarak döndürür.
    Sensör verisini (mock) ekler.

    Döndürülen her kayıt:
      id, makine_adi, sektor, guc_kw, sure_saat, karakteristik, aciklama,
      Failure_Within_7_Days, Remaining_Useful_Life_days, sensor_notu,
      aktif (bool), hedef_bitis_saati (None = serbest)
    """
    with open(yol, encoding="utf-8") as f:
        ham = json.load(f)

    makineler = []
    for m in ham:
        mid = m["id"]
        sensor = SENSOR_MOCK_DATA.get(mid, {})

        failure = sensor.get("Failure_Within_7_Days", False)
        rul = sensor.get("Remaining_Useful_Life_days", 999)
        snot = sensor.get("sensor_notu", "")

        # Arıza kısıtı: failure True veya RUL < eşik → devre dışı
        aktif = not (failure or rul < RUL_ESIK_GUN)

        makineler.append({
            "id": mid,
            "makine_adi": m["makine_adi"],
            "sektor": m.get("sektor", ""),
            "guc_kw": float(m["guc_kw"]),
            "sure_saat": int(m["sure_saat"]),
            "karakteristik": m.get("karakteristik", ""),
            "aciklama": m.get("aciklama", ""),
            "Failure_Within_7_Days": failure,
            "Remaining_Useful_Life_days": rul,
            "sensor_notu": snot,
            "aktif": aktif,
            "hedef_bitis_saati": None,  # serbest optimize
        })

    return makineler


# ============================================================================
# EPİAŞ / FİYAT CSV OKUYUCU
# ============================================================================

def oku_fiyatlar_csv(dosya_yolu: str) -> tuple[list[float], list[float]]:
    """
    EPİAŞ aylık PTF dosyasını okur (noktalı virgül ayracı ile).
    Türk formatındaki sayıları float'a, saatleri integer'a çevirir.
    Gruplama (groupby) yaparak 1 aylık verinin saatlik ortalamasını alır.
    Sonucu 1000'e bölerek TL/kWh cinsinden listeye çevirip döndürür.
    """
    # CSV'yi pandas ile oku
    df = pd.read_csv(dosya_yolu, sep=";", encoding="utf-8")
    
    # PTF (TL/MWh) sütunundaki veriler "1.000,00" formatında, str işlemlerle float'a çevir
    df["PTF (TL/MWh)"] = (
        df["PTF (TL/MWh)"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    
    # Saat sütunundan ilk 2 karakteri alıp integer'a çevir (0-23)
    df["Saat"] = df["Saat"].astype(str).str[:2].astype(int)
    
    # 1 aylık veriyi saatlere göre grupla ve ortalamasını al
    ozet = df.groupby("Saat")["PTF (TL/MWh)"].mean().reset_index().sort_values("Saat")
    
    # Sonuçları 1000'e bölerek TL/kWh cinsine çevir
    fiyatlar = (ozet["PTF (TL/MWh)"] / 1000.0).tolist()
    ges_uretimi = [0.0] * 24
    
    return fiyatlar, ges_uretimi


# ============================================================================
# YARDIMCI: BASELINE MALİYET
# ============================================================================

def _hesapla_maliyet(
    siparisler: list[dict],
    baslangic_map: dict[str, int],
    fiyatlar: list[float],
    ges: list[float],
) -> float:
    SAAT = len(fiyatlar)
    saatlik_guc = [0.0] * SAAT
    for s in siparisler:
        bas = baslangic_map[s["id"]]
        for t in range(bas, bas + s["sure_saat"]):
            if t < SAAT:
                saatlik_guc[t] += s["guc_kw"]
    toplam = 0.0
    for t in range(SAAT):
        net = max(0.0, saatlik_guc[t] - ges[t])
        toplam += net * fiyatlar[t]
    return toplam


def _naive_cizelge(siparisler: list[dict], SAAT: int = 24) -> dict[str, int]:
    cizelge: dict[str, int] = {}
    t = 0
    for s in siparisler:
        if t + s["sure_saat"] <= SAAT:
            cizelge[s["id"]] = t
            t += s["sure_saat"]
        else:
            cizelge[s["id"]] = max(0, SAAT - s["sure_saat"])
    return cizelge


# ============================================================================
# FİZİBİLİTE ANALİZİ
# ============================================================================

def _infeasible_analiz(
    siparisler: list[dict],
    fiyatlar: list[float],
    ges: list[float],
    max_guc: int,
) -> dict[str, Any]:
    SAAT = len(fiyatlar)
    sorunlar = []
    for s in siparisler:
        deadline = s.get("hedef_bitis_saati")
        sure = s["sure_saat"]
        iid = s["id"]
        if deadline is not None and sure > deadline:
            sorunlar.append({
                "is_id": iid,
                "sure_saat": sure,
                "hedef_bitis_saati": deadline,
                "min_bitis_saati_mumkun": sure,
                "aciklama": (
                    f"'{iid}' işi {sure} saat sürmektedir. "
                    f"Saat {deadline}:00'a kadar tamamlanamaz."
                ),
            })

    acil_isler = [s for s in siparisler if s.get("hedef_bitis_saati") is not None]
    saatlik_zorunlu_guc = [0.0] * SAAT
    for s in acil_isler:
        deadline = s["hedef_bitis_saati"]
        sure = s["sure_saat"]
        en_gec_bas = max(0, deadline - sure)
        for t in range(en_gec_bas, en_gec_bas + sure):
            if t < SAAT:
                saatlik_zorunlu_guc[t] += s["guc_kw"]

    kapasite_ihlalleri = []
    for t in range(SAAT):
        if saatlik_zorunlu_guc[t] > max_guc:
            kapasite_ihlalleri.append({
                "saat": t,
                "zorunlu_guc_kw": saatlik_zorunlu_guc[t],
                "max_guc_kw": max_guc,
                "asim_kw": saatlik_zorunlu_guc[t] - max_guc,
            })

    return {
        "deadline_ihlalleri": sorunlar,
        "kapasite_ihlalleri": kapasite_ihlalleri,
    }


# ============================================================================
# ANA OPTİMİZASYON FONKSİYONU
# ============================================================================

def optimize_schedule(
    siparisler: list[dict],
    fiyatlar: list[float],
    ges: list[float],
    max_guc_kw: int = 50000,
    hizlanma_orani: float = 1.0,
) -> dict[str, Any]:
    """
    Fabrika üretim çizelgesini OR-Tools CP-SAT ile optimize eder.
    Sadece aktif (arıza riski olmayan) makineler çizelgelenir.
    """
    # Dinamik Zaman Ufku Hesaplaması
    max_hedef = max([s.get("hedef_bitis_saati") or 24 for s in siparisler], default=24)
    zaman_ufku = max(48, int(max_hedef) + 24)
    SAAT = zaman_ufku

    # Gerekirse fiyatlar ve ges profillerini yeni zaman ufkuna göre uzat (tile)
    if len(fiyatlar) < SAAT:
        fiyatlar = (fiyatlar * (SAAT // 24 + 2))[:SAAT]
    if len(ges) < SAAT:
        ges = (ges * (SAAT // 24 + 2))[:SAAT]

    MAX_GUC = int(max_guc_kw)

    OLCEK = 1000
    fiyat_int = [round(f * OLCEK) for f in fiyatlar]
    ges_int = [round(g) for g in ges]

    # Baseline maliyet
    naive_map = _naive_cizelge(siparisler, SAAT)
    maliyet_once = _hesapla_maliyet(siparisler, naive_map, fiyatlar, ges)

    # CP-SAT Modeli
    model = cp_model.CpModel()

    bas_vars: dict[str, cp_model.IntVar] = {}
    for s in siparisler:
        iid = s["id"]
        sure = s["sure_saat"]
        ust = SAAT - sure
        if ust < 0:
            return {
                "durum": "HATA",
                "mesaj": (
                    f"'{iid}' işinin süresi ({sure} sa) 24 saati aşıyor! "
                    "Lütfen iş sürelerini kontrol edin."
                ),
            }
        bas_vars[iid] = model.NewIntVar(0, ust, f"bas_{iid}")

    # Deadline kısıtı
    for s in siparisler:
        deadline = s.get("hedef_bitis_saati")
        if deadline is not None:
            model.Add(bas_vars[s["id"]] + s["sure_saat"] <= deadline)

    # Fiziksel Makine Çakışma Kısıtı (NoOverlap)
    # Aynı makineye (OTO-001) atanan batch'lerin (B1, B2) aynı anda çalışmasını engeller
    machine_intervals = {}
    for s in siparisler:
        iid = s["id"]
        base_id = iid.split("_B")[0] # OTO-001_B1 -> OTO-001
        sure = s["sure_saat"]
        interval = model.NewIntervalVar(bas_vars[iid], sure, bas_vars[iid] + sure, f"interval_{iid}")
        
        if base_id not in machine_intervals:
            machine_intervals[base_id] = []
        machine_intervals[base_id].append(interval)

    for base_id, intervals in machine_intervals.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # ── Şebeke Yükü Kısıtı (Cumulative — çok daha verimli) ────────────────
    # hizlanma_orani 1.0 ise %60, 2.0 ise %95 olacak şekilde doğrusal artış
    katsayi = 0.60 + 0.35 * (hizlanma_orani - 1.0)
    en_buyuk_makine = max((float(s["guc_kw"]) for s in siparisler), default=0.0)
    MAX_SEBEKE_KW = max(int(max_guc_kw * katsayi), int(en_buyuk_makine))

    # Cumulative constraint: aynı anda çalışan makinelerin toplam gücü <= MAX_SEBEKE_KW
    # Interval'ler zaten NoOverlap bloğunda oluşturuldu, onları tekrar kullanıyoruz.
    demands = []
    cumul_intervals = []
    for s in siparisler:
        iid = s["id"]
        sure = s["sure_saat"]
        guc = int(s["guc_kw"])
        interval = model.NewIntervalVar(bas_vars[iid], sure, bas_vars[iid] + sure, f"cumul_{iid}")
        cumul_intervals.append(interval)
        demands.append(guc)

    model.AddCumulative(cumul_intervals, demands, MAX_SEBEKE_KW)

    # ── Ağırlıklı Amaç Fonksiyonu ──────────────────────────────────────
    Maliyet_Agirligi = int((2.0 - hizlanma_orani) * 100)
    Zaman_Agirligi = int((hizlanma_orani - 1.0) * 1000)

    maliyet_terimleri = []
    zaman_terimleri = []

    # Zaman terimleri (her işin bitiş saati)
    for s in siparisler:
        zaman_terimleri.append(bas_vars[s["id"]] + s["sure_saat"])

    # Maliyet terimleri — her iş için başlangıç saatindeki fiyat × güç × süre
    # Bu yaklaşım boolean'sız olduğu için çok daha hızlı
    for s in siparisler:
        iid = s["id"]
        sure = s["sure_saat"]
        guc = int(s["guc_kw"])
        bas = bas_vars[iid]

        # Her olası başlangıç saati için maliyet hesapla
        # Element constraint ile başlangıç saatine göre fiyatı seç
        fiyat_var = model.NewIntVar(0, max(fiyat_int), f"fiyat_{iid}")
        model.AddElement(bas, fiyat_int[:SAAT], fiyat_var)

        # Toplam maliyet = ortalama_fiyat * güç * süre (başlangıç fiyatıyla yaklaşık)
        maliyet_terimleri.append(fiyat_var * guc * sure)

    model.Minimize(Maliyet_Agirligi * sum(maliyet_terimleri) + Zaman_Agirligi * sum(zaman_terimleri))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False

    durum_kodu = solver.Solve(model)
    durum_str = solver.StatusName(durum_kodu)

    # INFEASIBLE
    if durum_kodu not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        analiz = _infeasible_analiz(siparisler, fiyatlar, ges, MAX_GUC)
        oneriler = []
        for ih in analiz["deadline_ihlalleri"]:
            oneriler.append(ih["aciklama"])
        for kap in analiz["kapasite_ihlalleri"]:
            oneriler.append(
                f"Saat {kap['saat']}:00'da zorunlu güç talebi "
                f"{kap['zorunlu_guc_kw']:.0f} kW, kapasite {kap['max_guc_kw']} kW'ı "
                f"{kap['asim_kw']:.0f} kW aşıyor."
            )
        return {
            "durum": "HATA",
            "solver_durumu": durum_str,
            "mesaj": "Kapasite/deadline kısıtları nedeniyle çizelge oluşturulamadı.",
            "oneriler": oneriler,
            "fizibilite_analizi": analiz,
        }

    # Sonuç
    optimal_bas = {s["id"]: solver.Value(bas_vars[s["id"]]) for s in siparisler}
    maliyet_sonra = _hesapla_maliyet(siparisler, optimal_bas, fiyatlar, ges)
    tasarruf = maliyet_once - maliyet_sonra
    tasarruf_yuzde = (tasarruf / maliyet_once * 100) if maliyet_once > 0 else 0.0

    cizelge = []
    for s in siparisler:
        iid = s["id"]
        bas = optimal_bas[iid]
        bitis = bas + s["sure_saat"]
        deadline = s.get("hedef_bitis_saati")
        cizelge.append({
            "id": iid,
            "makine_adi": s.get("makine_adi", iid),
            "sektor": s.get("sektor", ""),
            "sure_saat": s["sure_saat"],
            "guc_kw": s["guc_kw"],
            "baslangic_saati": bas,
            "bitis_saati": bitis,
            "calisma_araligi": f"{bas:02d}:00 – {bitis:02d}:00",
            "hedef_bitis_saati": deadline,
            "deadline_karsilandi": (bitis <= deadline) if deadline is not None else None,
        })

    return {
        "durum": durum_str,
        "onceki_maliyet": round(maliyet_once, 2),
        "optimize_maliyet": round(maliyet_sonra, 2),
        "tasarruf": round(tasarruf, 2),
        "tasarruf_yuzde": round(tasarruf_yuzde, 2),
        "cizelge": cizelge,
    }


# ============================================================================
# TERMİNAL RAPORU
# ============================================================================

def yazdir_rapor(sonuc: dict[str, Any], devre_disi: list[dict] | None = None) -> None:
    LINE = "=" * 72
    DASH = "-" * 72

    print(f"\n{LINE}")
    print("  FABRİKA ÜRETİM ÇİZELGESİ  —  OPTİMİZASYON RAPORU  v3.0")
    print(f"  Sensör Entegrasyonu + Arıza Kısıtları")
    print(LINE)

    if devre_disi:
        print(f"\n  ⚠️  BAKIM UYARISI — {len(devre_disi)} makine devre dışı:")
        print(DASH)
        for m in devre_disi:
            print(f"  🔴  {m['id']:10s}  {m['makine_adi']}")
            print(f"      RUL: {m['Remaining_Useful_Life_days']} gün | "
                  f"Arıza Riski: {m['Failure_Within_7_Days']}")
            if m.get("sensor_notu"):
                print(f"      Not: {m['sensor_notu']}")
        print(DASH)

    if sonuc.get("durum") == "HATA":
        print(f"  ❌  HATA : {sonuc.get('mesaj', '')}")
        oneriler = sonuc.get("oneriler", [])
        if oneriler:
            print(f"\n  📋  Tespit Edilen Sorunlar:")
            for i, o in enumerate(oneriler, 1):
                print(f"     {i}. {o}")
        print(LINE)
        return

    print(f"  Çözüm Durumu                : {sonuc['durum']}")
    print(DASH)
    print(f"  Optimizasyon Öncesi Maliyet : {sonuc['onceki_maliyet']:>12.2f} ₺")
    print(f"  Optimizasyon Sonrası Maliyet: {sonuc['optimize_maliyet']:>12.2f} ₺")
    print(f"  Kazanılan Tasarruf          : {sonuc['tasarruf']:>12.2f} ₺"
          f"  (%{sonuc['tasarruf_yuzde']:.1f})")
    print(LINE)
    print(f"  {'MAKİNE':<12s}  {'AD':<30s}  {'SÜRE':>5}  {'GÜÇ':>8}  {'ÇALIŞMA ARALIĞI':<18}")
    print(DASH)

    for row in sonuc["cizelge"]:
        print(
            f"  {row['id']:<12s}"
            f"  {row.get('makine_adi', ''):<30s}"
            f"  {row['sure_saat']:>3} sa"
            f"  {row['guc_kw']:>6.0f} kW"
            f"  {row['calisma_araligi']:<18}"
        )

    print(LINE)
    print()


# ============================================================================
# GİRİŞ NOKTASI
# ============================================================================

if __name__ == "__main__":
    # dataset.json'dan makineleri oku
    print("\n📂  dataset.json okunuyor …")
    tum_makineler = oku_dataset_json()
    print(f"   Toplam {len(tum_makineler)} makine bulundu.")

    # Aktif / devre dışı ayır
    aktif_makineler = [m for m in tum_makineler if m["aktif"]]
    devre_disi = [m for m in tum_makineler if not m["aktif"]]
    print(f"   ✅ Aktif: {len(aktif_makineler)}  |  🔴 Devre dışı: {len(devre_disi)}")

    # Fiyat verisini yükle
    if os.path.exists(EPIAS_CSV_YOL):
        print(f"\n📊  EPİAŞ aylık verisi bulundu: {os.path.basename(EPIAS_CSV_YOL)}")
        fiyatlar, ges = oku_fiyatlar_csv(EPIAS_CSV_YOL)
    else:
        print("\n⚠️  EPİAŞ CSV bulunamadı, fallback fiyatlar kullanılıyor.")
        fiyatlar = [0.25] * 24
        ges = [0.0] * 24

    # Dinamik MAX_GUC_KW: aktif makinelerin toplam gücü
    max_guc = sum(m["guc_kw"] for m in aktif_makineler)
    print(f"   Toplam aktif kapasite: {max_guc:.0f} kW")

    # Optimizasyon
    print("\n⚙️  OR-Tools optimizasyonu başlatılıyor …\n")
    sonuc = optimize_schedule(aktif_makineler, fiyatlar, ges, int(max_guc))

    yazdir_rapor(sonuc, devre_disi)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cizelge_sonuc.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"✅  Sonuçlar '{os.path.basename(out_path)}' dosyasına kaydedildi.\n")
