"""
Quran Audio Service — FastAPI v5.0
  - Concurrent ayah downloads via asyncio.gather + Semaphore (non-blocking)
  - everyayah.com CDN — no global ayah number lookup needed
  - 15 reciters (IDs match quran.com)
  - Progress events fire as each download completes (true parallel)
"""

import asyncio, io, json, tempfile, uuid
from pathlib import Path
from typing import Optional, Callable

import aiohttp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

# ── Config ──────────────────────────────────────────────────────────────
QURAN_BASE     = "https://api.quran.com/api/v4"
EVERYAYAH      = "https://everyayah.com/data"
TIMEOUT        = aiohttp.ClientTimeout(total=30, connect=8)
MAX_CONCURRENT = 10
MAX_AYAHS      = 300
AUDIO_DIR      = Path(tempfile.gettempdir()) / "quran_audio"
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Quran Audio Service", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://quran-frontend-clyc0edgm-noureddineachibanes-projects.vercel.app", "https://quran-frontend-git-vercel-r-7be73a-noureddineachibanes-projects.vercel.app/"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────────────
class AudioRequest(BaseModel):
    recitation_id: int           = Field(..., example=7)
    surah_number:  int           = Field(..., example=1)
    whole_surah:   bool          = Field(True)
    ayah_min:      Optional[int] = Field(None)
    ayah_max:      Optional[int] = Field(None)

class RecitationOut(BaseModel):
    id:              int
    reciter_name:    str
    style:           Optional[str] = None
    translated_name: Optional[str] = None

class SurahOut(BaseModel):
    id: int; name_arabic: str; name_simple: str
    translated_name: str; verses_count: int; revelation_place: str

# ── Reciter registry ─────────────────────────────────────────────────────
RECITERS = [
    {"id":  1, "name_ar": "عبد الباسط — مرتّل",        "name_en": "Abdul Basit Murattal",   "folder": "Abdul_Basit_Murattal_192kbps"},
    {"id":  2, "name_ar": "عبد الباسط — مجوّد",        "name_en": "Abdul Basit Mujawwad",   "folder": "Abdul_Basit_Mujawwad_128kbps"},
    {"id":  3, "name_ar": "عبد الرحمن السديس",          "name_en": "Abdurrahman As-Sudais",  "folder": "Abdurrahmaan_As-Sudais_192kbps"},
    {"id":  4, "name_ar": "أبو بكر الشاطري",            "name_en": "Abu Bakr Al-Shatri",     "folder": "Abu_Bakr_Ash-Shaatree_128kbps"},
    {"id":  5, "name_ar": "هاني الرفاعي",               "name_en": "Hani Ar-Rifai",          "folder": "Hani_Rifai_192kbps"},
    {"id":  6, "name_ar": "محمود خليل الحصري",          "name_en": "Mahmoud Al-Hussary",     "folder": "Husary_128kbps"},
    {"id":  7, "name_ar": "مشاري راشد العفاسي",         "name_en": "Mishary Alafasy",        "folder": "Alafasy_128kbps"},
    {"id":  8, "name_ar": "محمد صديق المنشاوي — مجوّد", "name_en": "Minshawi Mujawwad",     "folder": "Minshawy_Mujawwad_128kbps"},
    {"id":  9, "name_ar": "محمد صديق المنشاوي — مرتّل", "name_en": "Minshawi Murattal",     "folder": "Minshawy_Murattal_128kbps"},
    {"id": 10, "name_ar": "سعود الشريم",                "name_en": "Saud Al-Shuraym",        "folder": "Saud_Al-Shuraym_128kbps"},
    {"id": 11, "name_ar": "ماهر المعيقلي",              "name_en": "Maher Al-Mueaqly",       "folder": "Maher_AlMuaiqly_128kbps"},
    {"id": 12, "name_ar": "محمود خليل الحصري — معلم",   "name_en": "Husary Muallim",         "folder": "Husary_128kbps"},
    {"id": 13, "name_ar": "سعد الغامدي",                "name_en": "Saad Al-Ghamdi",         "folder": "Saad_Al-Ghamdi_128kbps"},
    {"id": 14, "name_ar": "ياسر الدوسري",               "name_en": "Yasser Ad-Dussary",      "folder": "Yasser_Ad-Dussary_128kbps"},
    {"id": 15, "name_ar": "ناصر القطامي",               "name_en": "Nasser Al-Qatami",       "folder": "Nasser_Alqatami_128kbps"},
]
_RMAP = {r["id"]: r for r in RECITERS}
FALLBACK_FOLDER = "Alafasy_128kbps"

# ── URL helpers ──────────────────────────────────────────────────────────
def _url(folder: str, surah: int, ayah: int) -> str:
    return f"{EVERYAYAH}/{folder}/{surah:03d}{ayah:03d}.mp3"

# ── Fetch one raw MP3 ────────────────────────────────────────────────────
async def _fetch_raw(session: aiohttp.ClientSession, url: str) -> tuple[bytes, bool]:
    try:
        async with session.get(url) as r:
            if r.status == 200:
                data = await r.read()
                if len(data) > 1000:
                    return data, True
        return b"", False
    except Exception:
        return b"", False

# ── Download one ayah — with semaphore + fallback + progress ────────────
async def _fetch_ayah(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    folder: str,
    surah: int, ayah: int,
    index: int, total: int,
    on_progress: Optional[Callable],
) -> bytes:
    async with sem:
        data, ok = await _fetch_raw(session, _url(folder, surah, ayah))
        if ok:
            status = "ok"
        elif folder != FALLBACK_FOLDER:
            data, ok = await _fetch_raw(session, _url(FALLBACK_FOLDER, surah, ayah))
            status = "fallback" if ok else "failed"
        else:
            status = "failed"

        if on_progress:
            await on_progress({"type": "progress", "ayah": ayah,
                               "index": index, "total": total, "status": status})
        return data if ok else b""

# ── Merge MP3 bytes ──────────────────────────────────────────────────────
def _merge(chunks: list[bytes]) -> bytes:
    if not PYDUB_OK:
        raise HTTPException(500, "pydub not installed — pip install pydub audioop-lts")
    combined = AudioSegment.empty()
    for raw in chunks:
        if raw:
            combined += AudioSegment.from_file(io.BytesIO(raw), format="mp3")
    if len(combined) == 0:
        raise HTTPException(502, "All ayah downloads failed")
    buf = io.BytesIO()
    combined.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()

# ── Main pipeline ────────────────────────────────────────────────────────
async def _pipeline(
    req: AudioRequest,
    on_progress: Optional[Callable] = None,
) -> tuple[bytes, str, int, int]:
    rec = _RMAP.get(req.recitation_id)
    if not rec:
        raise HTTPException(400, f"recitation_id {req.recitation_id} not found")
    folder = rec["folder"]

    # Resolve ayah range
    if req.whole_surah:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            r = await s.get(f"{QURAN_BASE}/chapters/{req.surah_number}")
            if r.status != 200:
                raise HTTPException(502, "Cannot fetch surah info from quran.com")
            ch = (await r.json())["chapter"]
        ayah_min, ayah_max = 1, ch["verses_count"]
    else:
        ayah_min = req.ayah_min or 1
        ayah_max = req.ayah_max or ayah_min

    if ayah_max - ayah_min + 1 > MAX_AYAHS:
        raise HTTPException(400, f"Max {MAX_AYAHS} ayahs per request")

    ayahs = list(range(ayah_min, ayah_max + 1))
    total = len(ayahs)

    if on_progress:
        await on_progress({"type": "start", "total": total})

    # Concurrent downloads — all fire at once, limited by semaphore
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT)
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=connector) as session:
        tasks = [
            _fetch_ayah(session, sem, folder, req.surah_number,
                        ayah, i + 1, total, on_progress)
            for i, ayah in enumerate(ayahs)
        ]
        chunks: list[bytes] = await asyncio.gather(*tasks)

    if on_progress:
        await on_progress({"type": "merging", "message": f"Merging {total} ayahs..."})

    merged = _merge(chunks)
    fname  = f"surah_{req.surah_number}_ayah_{ayah_min}-{ayah_max}.mp3"
    return merged, fname, ayah_min, ayah_max

# ── Routes ───────────────────────────────────────────────────────────────
@app.get("/recitations", response_model=list[RecitationOut])
async def get_recitations():
    return [RecitationOut(id=r["id"], reciter_name=r["name_ar"],
                          style=None, translated_name=r["name_en"]) for r in RECITERS]

@app.get("/surahs", response_model=list[SurahOut])
async def get_surahs():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
        r = await s.get(f"{QURAN_BASE}/chapters", params={"language": "en"})
        if r.status != 200:
            raise HTTPException(502, "Cannot fetch surahs")
        data = await r.json()
    return [
        SurahOut(
            id=ch["id"], name_arabic=ch.get("name_arabic", ""),
            name_simple=ch.get("name_simple", ""),
            translated_name=(ch.get("translated_name") or {}).get("name", ""),
            verses_count=ch.get("verses_count", 0),
            revelation_place=ch.get("revelation_place", ""),
        )
        for ch in data.get("chapters", [])
    ]

@app.post("/generate-audio")
async def generate_audio(req: AudioRequest):
    if not req.whole_surah:
        if not req.ayah_min or not req.ayah_max:
            raise HTTPException(400, "ayah_min and ayah_max required")
        if req.ayah_min < 1 or req.ayah_max < req.ayah_min:
            raise HTTPException(400, "Invalid ayah range")
    merged, fname, *_ = await _pipeline(req)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(merged); tmp.close()
    return FileResponse(tmp.name, media_type="audio/mpeg", filename=fname)

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = AUDIO_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found or expired")
    return FileResponse(str(path), media_type="audio/mpeg", filename=filename)

# ── WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/ws/generate-audio")
async def ws_generate(ws: WebSocket):
    await ws.accept()

    async def send(evt: dict):
        await ws.send_text(json.dumps(evt, ensure_ascii=False))

    try:
        raw = await ws.receive_text()
        try:
            req = AudioRequest(**json.loads(raw))
        except Exception as e:
            await send({"type": "error", "message": f"Bad request: {e}"}); return

        if not req.whole_surah:
            if not req.ayah_min or not req.ayah_max:
                await send({"type": "error", "message": "ayah_min/ayah_max required"}); return
            if req.ayah_min < 1 or req.ayah_max < req.ayah_min:
                await send({"type": "error", "message": "Invalid ayah range"}); return

        merged, fname, *_ = await _pipeline(req, on_progress=send)
        uid = uuid.uuid4().hex[:8]
        out = f"{uid}_{fname}"
        (AUDIO_DIR / out).write_bytes(merged)

        await send({"type": "done", "filename": fname,
                    "download_url": f"/audio/{out}",
                    "size_kb": round(len(merged) / 1024, 1)})

    except WebSocketDisconnect:
        pass
    except HTTPException as e:
        try: await send({"type": "error", "message": e.detail})
        except Exception: pass
    except Exception as e:
        try: await send({"type": "error", "message": str(e)})
        except Exception: pass
    finally:
        try: await ws.close()
        except Exception: pass
