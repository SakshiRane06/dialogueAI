import sys
import os
import tempfile
from pathlib import Path

# Two-voice TTS using free Windows voices via pyttsx3.
# Exports MP3 if possible, else falls back to WAV.

def synthesize_windows_tts(lines, host_hint="Zira", expert_hint="David"):
    import pyttsx3
    engine = pyttsx3.init()  # SAPI5 on Windows
    voices = engine.getProperty('voices')

    def pick_voice(hint):
        for v in voices:
            name = getattr(v, 'name', '').lower()
            if hint.lower() in name:
                return v.id
        return voices[0].id if voices else None

    host_voice_id = pick_voice(host_hint)
    expert_voice_id = pick_voice(expert_hint)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', max(150, min(rate, 185)))

    temp_dir = tempfile.mkdtemp()
    wav_paths = []

    for idx, (role, text) in enumerate(lines):
        engine.setProperty('voice', host_voice_id if role == 'host' else expert_voice_id)
        wav_path = os.path.join(temp_dir, f"seg_{idx:03d}.wav")
        engine.save_to_file(text, wav_path)
        wav_paths.append(wav_path)

    engine.runAndWait()
    return wav_paths


def combine_audio(wav_paths, out_path_mp3, out_path_wav):
    try:
        from pydub import AudioSegment
        combined = AudioSegment.silent(duration=400)
        for path in wav_paths:
            seg = AudioSegment.from_wav(path)
            combined += seg + AudioSegment.silent(duration=220)
        # Try MP3 first
        try:
            combined.export(out_path_mp3, format="mp3")
            return out_path_mp3
        except Exception:
            # Fallback to WAV
            combined.export(out_path_wav, format="wav")
            return out_path_wav
    except Exception as e:
        raise RuntimeError(f"Audio combining failed: {e}")


def parse_dialogue(dialogue_text):
    lines = []
    for raw in dialogue_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        lower = s.lower()
        if lower.startswith(("🎙️ host:", "host:")):
            lines.append(("host", s.split(":", 1)[1].strip()))
        elif lower.startswith(("🧠 expert:", "expert:")):
            lines.append(("expert", s.split(":", 1)[1].strip()))
        else:
            role = "host" if len(lines) % 2 == 0 else "expert"
            lines.append((role, s))
    return lines


def main():
    if len(sys.argv) < 3:
        print("Usage: python make_podcast.py <dialogue.txt> <output_basename>")
        sys.exit(1)

    dialogue_path = Path(sys.argv[1])
    out_base = Path(sys.argv[2])
    text = dialogue_path.read_text(encoding="utf-8")

    lines = parse_dialogue(text)
    wavs = synthesize_windows_tts(lines)

    out_mp3 = str(out_base) if str(out_base).lower().endswith('.mp3') else str(out_base) + '.mp3'
    out_wav = str(out_base) if str(out_base).lower().endswith('.wav') else str(out_base) + '.wav'

    final_path = combine_audio(wavs, out_mp3, out_wav)
    print(f"✅ Audio saved to {final_path}")

if __name__ == "__main__":
    main()