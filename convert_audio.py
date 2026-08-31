import os
import subprocess
from pathlib import Path

AUDIO_SRC = "raw_data/ILCI_AGRICULTURE_SPEECH_CORPUS_INTERNS/ILCI_Agriculture_audio_files"
AUDIO_DST = "data/audio_wav"

os.makedirs(AUDIO_DST, exist_ok=True)

webm_files = list(Path(AUDIO_SRC).glob("*.webm"))
print(f"Found {len(webm_files)} .webm files")

for i, webm_path in enumerate(webm_files):
    wav_name = webm_path.stem + ".wav"
    wav_path = os.path.join(AUDIO_DST, wav_name)

    subprocess.run([
        "ffmpeg", "-i", str(webm_path),
        "-ar", "16000", "-ac", "1",
        "-y",
        wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if (i + 1) % 100 == 0:
        print(f"Converted {i + 1}/{len(webm_files)}")

print("Done.")
