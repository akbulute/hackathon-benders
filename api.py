"""
=============================================================================
  FABRİKA ÜRETİM ÇİZELGESİ  —  FastAPI In-Memory REST Servisi  v4.0
=============================================================================

  Mimari
  ------
  • Uygulama başlarken dataset.json + EPİAŞ CSV yalnızca 1 kez okunur.
  • Sensör verisine göre arızalı makineler devre dışı bırakılır.
  • Tüm endpoint'ler bellekteki veriye erişir (disk I/O yapılmaz).
  • INFEASIBLE durumu HTTP 400 + açıklayıcı JSON ile raporlanır.

  Başlatma
  --------
      pip install ortools pandas fastapi uvicorn

      uvicorn api:app --reload --port 8000

  Swagger UI : http://localhost:8000/docs
  ReDoc      : http://localhost:8000/redoc
=============================================================================
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Optimizasyon motoru
from uretim_optimizasyonu import (
    EPIAS_CSV_YOL,
    oku_fiyatlar_csv,
    oku_dataset_json,
    optimize_schedule,
)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fabrika-api")


# ============================================================================
# IN-MEMORY STATE
# ============================================================================

class AppState:
    """Uygulama genelinde tek bir örnek (singleton) olarak çalışır."""

    def __init__(self) -> None:
        self.tum_makineler: list[dict] = []     # dataset.json'dan tüm makineler
        self.aktif_makineler: list[dict] = []   # arızasız, çizelgelenebilir
        self.devre_disi: list[dict] = []        # arıza riski olanlar
        self.fiyatlar: list[float] = []         # EPİAŞ fiyatları (24 saat)
        self.ges: list[float] = []              # güneş üretim verileri
        self.max_guc_kw: int = 0                # dinamik kapasite
        self.hazir: bool = False

    def yukle_veriler(self) -> None:
        """Startup'ta bir kez çağrılır — dataset.json + EPİAŞ CSV yükler."""
        import os

        # ── Makineleri yükle ──────────────────────────────────────────────
        self.tum_makineler = oku_dataset_json()
        self.aktif_makineler = [m for m in self.tum_makineler if m["aktif"]]
        self.devre_disi = [m for m in self.tum_makineler if not m["aktif"]]

        # ── Fiyatları yükle ───────────────────────────────────────────────
        if os.path.exists(EPIAS_CSV_YOL):
            self.fiyatlar, self.ges = oku_fiyatlar_csv(EPIAS_CSV_YOL)
            log.info("📊  EPİAŞ aylık CSV yüklendi: %s", os.path.basename(EPIAS_CSV_YOL))
        else:
            log.warning("⚠️  EPİAŞ CSV bulunamadı, fallback fiyatlar kullanılıyor.")
            self.fiyatlar = [0.25] * 24
            self.ges = [0.0] * 24

        # ── Dinamik kapasite ──────────────────────────────────────────────
        self.max_guc_kw = int(sum(m["guc_kw"] for m in self.aktif_makineler))

        self.hazir = True
        log.info(
            "Veriler yüklendi: %d makine (%d aktif, %d devre dışı), "
            "%d saatlik fiyat, kapasite: %d kW",
            len(self.tum_makineler),
            len(self.aktif_makineler),
            len(self.devre_disi),
            len(self.fiyatlar),
            self.max_guc_kw,
        )

    def guncelle_is(self, is_id: str, yeni_deadline: Optional[int]) -> bool:
        """Bellekteki ilgili işin hedef_bitis_saati'ni günceller."""
        for s in self.aktif_makineler:
            if s["id"] == is_id:
                onceki = s.get("hedef_bitis_saati")
                s["hedef_bitis_saati"] = yeni_deadline
                log.info("İş '%s' deadline: %s → %s", is_id, onceki, yeni_deadline)
                return True
        return False

    def optimize_et(self) -> dict[str, Any]:
        """Bellekteki mevcut veriyle optimizasyonu çalıştırır."""
        return optimize_schedule(
            siparisler=copy.deepcopy(self.aktif_makineler),
            fiyatlar=self.fiyatlar,
            ges=self.ges,
            max_guc_kw=self.max_guc_kw,
        )


# Global state
state = AppState()


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama ayağa kalkarken verileri yükle."""
    log.info("🚀  Fabrika API v4.0 başlatılıyor …")
    try:
        state.yukle_veriler()
        log.info("✅  Bellek hazır. Servis istekleri kabul ediyor.")
    except Exception as exc:
        log.error("❌  Veri yüklenemedi: %s", exc)

    yield

    log.info("🛑  Fabrika API kapatılıyor.")


# ============================================================================
# UYGULAMA
# ============================================================================

app = FastAPI(
    title="Fabrika Üretim Çizelgesi — Sensör Entegrasyonlu REST API",
    description=(
        "OR-Tools CP-SAT tabanlı enerji maliyeti optimizasyonu. "
        "dataset.json'dan makine verisi + arıza kısıtları. "
        "Tüm işlemler bellekte gerçekleşir."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELLER
# ============================================================================

class IsGuncelleIstek(BaseModel):
    model_config = {"json_schema_extra": {"example": {"is_id": "SVN-001", "yeni_hedef_bitis_saati": 15}}}

    is_id: str = Field(..., description="Güncellenecek makine/iş ID'si (ör: SVN-001)")
    yeni_hedef_bitis_saati: Optional[int] = Field(
        None, ge=1, le=23,
        description="Yeni deadline saati (1-23). None = kısıt kaldır.",
    )


class OptimizeIstek(BaseModel):
    model_config = {"json_schema_extra": {"example": {"sektor": "Automotive Industry", "deadline": "2026-05-03T15:30:00Z", "hizlanma_orani": 1.5, "adet": 2}}}

    sektor: str = Field(..., description="Siparişin atanacağı sektör")
    deadline: str = Field(..., description="Hedef bitiş tarihi/saati (ISO format)")
    hizlanma_orani: float = Field(1.0, ge=1.0, le=2.0, description="Hızlanma oranı (1x - 2x)")
    adet: int = Field(1, ge=1, le=10, description="Üretim adedi (batch sayısı)")


class HataYanit(BaseModel):
    durum: str
    mesaj: str
    oneriler: Optional[list[str]] = None
    fizibilite_analizi: Optional[dict] = None


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def _hazirlik_kontrol() -> None:
    if not state.hazir:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "durum": "HATA",
                "mesaj": "Uygulama henüz hazır değil. Veriler yüklenememiş olabilir.",
            },
        )


def _enerji_durumu(fiyat: float, ortalama: float) -> str:
    if fiyat < ortalama * 0.85:
        return "ucuz_saat"
    elif fiyat > ortalama * 1.25:
        return "pahali_saat"
    else:
        return "normal_saat"


def _frontend_formati_olustur(
    sonuc: dict[str, Any],
    fiyatlar: list[float],
    devre_disi: list[dict],
) -> dict[str, Any]:
    """
    optimize_schedule() çıktısını Frontend JSON yapısına dönüştürür.

    Çıktı:
    {
      "kpi_verileri":    { gunluk_maliyet_tl, tasarruf_orani_yuzde,
                           toplam_kapasite_kw, aktif_makine_sayisi,
                           bakim_uyarisi },
      "epias_grafigi":   { saatler, fiyatlar },
      "gantt_cizelgesi": [ { makine_adi, makine_id, sektor, isler: [...] } ]
    }
    """
    # ── Bakım uyarısı ─────────────────────────────────────────────────────
    bakim_uyarisi = []
    for m in devre_disi:
        bakim_uyarisi.append({
            "makine_id": m["id"],
            "makine_adi": m["makine_adi"],
            "Failure_Within_7_Days": m["Failure_Within_7_Days"],
            "Remaining_Useful_Life_days": m["Remaining_Useful_Life_days"],
            "sensor_notu": m.get("sensor_notu", ""),
        })

    # ── KPI ───────────────────────────────────────────────────────────────
    kpi = {
        "gunluk_maliyet_tl": round(sonuc["optimize_maliyet"], 2),
        "tasarruf_orani_yuzde": round(sonuc["tasarruf_yuzde"], 1),
        "toplam_kapasite_kw": sum(r["guc_kw"] for r in sonuc["cizelge"]),
        "aktif_makine_sayisi": len(sonuc["cizelge"]),
        "devre_disi_makine_sayisi": len(devre_disi),
        "bakim_uyarisi": bakim_uyarisi,
    }

    # ── EPİAŞ Grafiği & Enerji Yükü ───────────────────────────────────────
    max_saat = 24
    if sonuc.get("cizelge"):
        max_saat = max(max_saat, max(r["bitis_saati"] for r in sonuc["cizelge"]))

    enerji_yuku = [0.0] * max_saat
    for is_row in sonuc.get("cizelge", []):
        for h in range(is_row["baslangic_saati"], is_row["bitis_saati"]):
            if h < max_saat:
                enerji_yuku[h] += is_row.get("guc_kw", 0)

    genisletilmis_fiyatlar = (fiyatlar * (max_saat // 24 + 2))[:max_saat]

    epias = {
        "saatler": [f"{(h % 24):02d}:00" + (f" (+{h // 24}G)" if h >= 24 else "") for h in range(max_saat)],
        "fiyatlar": [round(f, 4) for f in genisletilmis_fiyatlar],
        "enerji_yuku": [round(y, 1) for y in enerji_yuku],
    }

    # ── Gantt Çizelgesi ───────────────────────────────────────────────────
    ortalama_fiyat = sum(fiyatlar) / len(fiyatlar) if fiyatlar else 1.0

    gantt = []
    for is_row in sonuc["cizelge"]:
        bas = is_row["baslangic_saati"]
        bit = is_row["bitis_saati"]
        saat_indeks = min(bas, 23)
        fiyat_bu_saat = fiyatlar[saat_indeks]

        gantt.append({
            "makine_id": is_row["id"],
            "makine_adi": is_row.get("makine_adi", is_row["id"]),
            "sektor": is_row.get("sektor", ""),
            "isler": [{
                "is_adi": is_row["id"],
                "baslama_saati": f"{bas:02d}:00",
                "bitis_saati": f"{bit:02d}:00",
                "sure_saat": is_row["sure_saat"],
                "guc_kw": is_row["guc_kw"],
                "enerji_durumu": _enerji_durumu(fiyat_bu_saat, ortalama_fiyat),
            }],
        })

    return {
        "kpi_verileri": kpi,
        "epias_grafigi": epias,
        "gantt_cizelgesi": gantt,
    }


def _sonuc_veya_hata(sonuc: dict[str, Any]) -> JSONResponse:
    if sonuc.get("durum") == "HATA":
        hata = {
            "durum": sonuc.get("durum", "HATA"),
            "mesaj": sonuc.get("mesaj", "Bilinmeyen hata."),
        }
        if sonuc.get("oneriler"):
            hata["oneriler"] = sonuc["oneriler"]
        if sonuc.get("fizibilite_analizi"):
            hata["fizibilite_analizi"] = sonuc["fizibilite_analizi"]
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=hata)

    frontend_json = _frontend_formati_olustur(sonuc, state.fiyatlar, state.devre_disi)
    return JSONResponse(status_code=status.HTTP_200_OK, content=frontend_json)


# ============================================================================
# ENDPOINT 1: GET /cizelge
# ============================================================================

@app.get(
    "/cizelge",
    summary="Optimal üretim çizelgesini getir (Frontend formatı)",
    response_description="kpi_verileri + epias_grafigi + gantt_cizelgesi",
    tags=["Çizelge"],
)
def get_cizelge() -> JSONResponse:
    """
    Bellekteki mevcut makine listesi ve fiyat verileriyle OR-Tools
    optimizasyonunu çalıştırır.

    Arızalı makineler otomatik olarak devre dışı bırakılır ve
    `kpi_verileri.bakim_uyarisi` alanında raporlanır.
    """
    _hazirlik_kontrol()

    log.info("GET /cizelge — %d aktif makine, %d devre dışı",
             len(state.aktif_makineler), len(state.devre_disi))
    sonuc = state.optimize_et()

    if sonuc.get("durum") == "HATA":
        log.warning("GET /cizelge — HATA: %s", sonuc.get("mesaj"))
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=sonuc)

    log.info("GET /cizelge — %s | maliyet: %.2f ₺ | tasarruf: %%%.1f",
             sonuc.get("durum"), sonuc.get("optimize_maliyet", 0),
             sonuc.get("tasarruf_yuzde", 0))

    frontend_json = _frontend_formati_olustur(sonuc, state.fiyatlar, state.devre_disi)
    return JSONResponse(status_code=status.HTTP_200_OK, content=frontend_json)


# ============================================================================
# ENDPOINT 2: POST /is_guncelle
# ============================================================================

@app.post(
    "/is_guncelle",
    summary="Bir makinenin deadline'ını güncelle ve yeniden optimize et",
    tags=["Çizelge"],
    responses={
        200: {"description": "Yeni optimal çizelge üretildi."},
        400: {"description": "INFEASIBLE.", "model": HataYanit},
        404: {"description": "Makine ID'si bulunamadı."},
    },
)
def post_is_guncelle(istek: IsGuncelleIstek) -> JSONResponse:
    """Bir makinenin hedef_bitis_saati'ni günceller ve yeniden optimize eder."""
    _hazirlik_kontrol()

    bulundu = state.guncelle_is(istek.is_id, istek.yeni_hedef_bitis_saati)
    if not bulundu:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "durum": "HATA",
                "mesaj": f"'{istek.is_id}' ID'li makine bulunamadı veya devre dışı.",
                "aktif_makineler": [s["id"] for s in state.aktif_makineler],
            },
        )

    log.info("POST /is_guncelle — '%s' deadline → %s",
             istek.is_id, istek.yeni_hedef_bitis_saati)

    sonuc = state.optimize_et()
    return _sonuc_veya_hata(sonuc)


# ============================================================================
# ENDPOINT 3: POST /optimize
# ============================================================================

@app.post(
    "/optimize",
    summary="Sektörel sipariş için optimizasyonu çalıştır",
    tags=["Çizelge"],
)
def post_optimize(istek: OptimizeIstek) -> JSONResponse:
    """Belirli bir sektör için sipariş oluşturur, adet kadar çoğaltır ve optimize eder."""
    _hazirlik_kontrol()

    global_makineler = []

    try:
        simdi = datetime.now()
        dt_str = istek.deadline.replace("Z", "")
        if "." in dt_str:
            dt_str = dt_str.split(".")[0]
        deadline_dt = datetime.fromisoformat(dt_str)
        
        # Başlangıç anına göre kaç saat sonra olduğunu hesapla
        saat_farki = int((deadline_dt - simdi).total_seconds() / 3600)
        hedef_saat = max(1, saat_farki) # En az 1 saat
        
        # Bu sektöre ait temel makineleri bul
        sektor_makineleri = [m for m in state.aktif_makineler if m.get("sektor") == istek.sektor]
        if not sektor_makineleri:
            raise HTTPException(
                status_code=404,
                detail={"durum": "HATA", "mesaj": f"'{istek.sektor}' sektöründe aktif makine bulunamadı."}
            )
            
        # Adet kadar kopyala (Batch)
        for i in range(1, istek.adet + 1):
            for m in sektor_makineleri:
                kopya = copy.deepcopy(m)
                kopya["id"] = f"{m['id']}_B{i}"
                kopya["makine_adi"] = f"{m['makine_adi']} (B{i})"
                kopya["hedef_bitis_saati"] = hedef_saat
                global_makineler.append(kopya)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"durum": "HATA", "mesaj": f"Geçersiz veri veya format: {e}"}
        )

    log.info("POST /optimize — Sektör: %s, Adet: %d, Hız: %s", istek.sektor, istek.adet, istek.hizlanma_orani)

    # Geçici verilerle optimizasyonu çalıştır
    sonuc = optimize_schedule(
        siparisler=global_makineler,
        fiyatlar=state.fiyatlar,
        ges=state.ges,
        max_guc_kw=state.max_guc_kw,
        hizlanma_orani=istek.hizlanma_orani,
    )

    return _sonuc_veya_hata(sonuc)


# ============================================================================
# ENDPOINT 3: GET /durum
# ============================================================================

@app.get("/durum", summary="Servis sağlık kontrolü", tags=["Sistem"])
def get_durum():
    """Servisin durumunu ve bellekteki veri özetini döner."""
    return {
        "servis": "Fabrika Çizelge API",
        "versiyon": "4.0.0",
        "hazir": state.hazir,
        "bellek": {
            "toplam_makine": len(state.tum_makineler),
            "aktif_makine": len(state.aktif_makineler),
            "devre_disi_makine": len(state.devre_disi),
            "fiyat_saati": len(state.fiyatlar),
            "max_guc_kw": state.max_guc_kw,
        },
        "aktif_makineler": [
            {
                "id": m["id"],
                "makine_adi": m["makine_adi"],
                "sektor": m["sektor"],
                "sure_saat": m["sure_saat"],
                "guc_kw": m["guc_kw"],
            }
            for m in state.aktif_makineler
        ],
        "devre_disi_makineler": [
            {
                "id": m["id"],
                "makine_adi": m["makine_adi"],
                "Failure_Within_7_Days": m["Failure_Within_7_Days"],
                "Remaining_Useful_Life_days": m["Remaining_Useful_Life_days"],
                "sensor_notu": m.get("sensor_notu", ""),
            }
            for m in state.devre_disi
        ],
    }


# ============================================================================
# ENDPOINT 4: GET /makineler
# ============================================================================

@app.get("/makineler", summary="Tüm makine listesini getir", tags=["Makineler"])
def get_makineler():
    """dataset.json'dan yüklenen tüm makineleri (aktif + devre dışı) döner."""
    _hazirlik_kontrol()
    return {
        "toplam": len(state.tum_makineler),
        "aktif_sayisi": len(state.aktif_makineler),
        "devre_disi_sayisi": len(state.devre_disi),
        "makineler": [
            {
                "id": m["id"],
                "makine_adi": m["makine_adi"],
                "sektor": m["sektor"],
                "guc_kw": m["guc_kw"],
                "sure_saat": m["sure_saat"],
                "karakteristik": m["karakteristik"],
                "aktif": m["aktif"],
                "Failure_Within_7_Days": m["Failure_Within_7_Days"],
                "Remaining_Useful_Life_days": m["Remaining_Useful_Life_days"],
            }
            for m in state.tum_makineler
        ],
    }


# ============================================================================
# DOĞRUDAN ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
