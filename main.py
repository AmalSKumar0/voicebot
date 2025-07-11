from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import torch
import os
import uuid
import shutil
import subprocess
import tempfile
import traceback
import asyncio
from response import response as ai_response  # Import AI response function

# ---------- Configuration ----------
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load Whisper model once
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Loading WhisperModel on device: {device}")
model = WhisperModel("tiny", device=device, compute_type="float16" if device == "cuda" else "int8")

# ---------- FastAPI App ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_page():
    print("[DEBUG] Serving index.html")
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

# ---------- WebSocket Endpoint ----------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    last_transcript = ""

    try:
        print(f"[DEBUG] WebSocket connection started: {session_id}")

        while True:
            try:
                # Set timeout to prevent hanging connection
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            except asyncio.TimeoutError:
                print("[DEBUG] WebSocket timeout - closing connection")
                break

            chunk_file = os.path.join(TEMP_DIR, f"{session_id}_{uuid.uuid4()}.webm")
            wav_file = chunk_file.replace(".webm", ".wav")

            # Save the chunk to a new file
            with open(chunk_file, "wb") as f:
                f.write(data)

            print(f"[DEBUG] Received audio chunk: {chunk_file}")

            # Convert to WAV
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", chunk_file, "-ar", "16000", "-ac", "1", wav_file
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print("[ERROR] ffmpeg failed to convert audio")
                await websocket.send_json({"error": "Audio conversion failed"})
                continue

            # Transcribe
            segments, _ = model.transcribe(wav_file, beam_size=1)
            transcript = "".join([s.text for s in segments]).strip()
            print(f"[DEBUG] Transcribed Text: {transcript}")

            if transcript and transcript != last_transcript:
                await websocket.send_json({"text": transcript, "type": "user"})
                last_transcript = transcript

                # 🧠 Generate AI response
                try:
                    ai_reply = ai_response(session_id, transcript)
                    await websocket.send_json({"text": ai_reply, "type": "ai"})
                except Exception as e:
                    print(f"[ERROR] AI generation failed: {e}")
                    await websocket.send_json({"error": "AI response failed"})

            # Clean up
            for path in (chunk_file, wav_file):
                if os.path.exists(path):
                    os.remove(path)

    except WebSocketDisconnect:
        print(f"[DEBUG] WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"[ERROR] Exception in WebSocket: {e}")
        traceback.print_exc()
        await websocket.send_json({"error": str(e)})

# ---------- POST Upload Endpoint ----------
@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    raw_path = os.path.join(TEMP_DIR, f"{uid}_{audio.filename}")
    wav_path = os.path.join(TEMP_DIR, f"{uid}.wav")

    try:
        print(f"[DEBUG] Upload received: {audio.filename}")
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        print(f"[DEBUG] Saved to: {raw_path}")

        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"[DEBUG] Converted to WAV: {wav_path}")

        segments, _ = model.transcribe(
            wav_path,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters={"threshold": 0.5}
        )
        transcript = "".join([seg.text for seg in segments]).strip()
        print(f"[DEBUG] Final Transcript: {transcript}")

        return {"text": transcript}
    except Exception as e:
        print(f"[ERROR] Exception in POST /transcribe: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        for path in (raw_path, wav_path):
            if os.path.exists(path):
                os.remove(path)
        print(f"[DEBUG] Cleaned up files: {raw_path}, {wav_path}")