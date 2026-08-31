import os
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

AUDIO_DIR = "/workspace/data/audio_wav"
TEXT_DIR = "/workspace/raw_data/ILCI_AGRICULTURE_SPEECH_CORPUS_INTERNS/ILCI_Agriculture_text_files"

def build_pairs():
    wav_files = sorted(Path(AUDIO_DIR).glob("*.wav"))
    pairs = []

    for wav_path in wav_files:
        txt_path = Path(TEXT_DIR) / (wav_path.stem + ".txt")
        if not txt_path.exists():
            print(f"WARNING: no matching transcript for {wav_path.name}, skipping")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        if not transcript:
            print(f"WARNING: empty transcript for {wav_path.name}, skipping")
            continue
        pairs.append({"audio": str(wav_path), "transcript": transcript})

    print(f"Built {len(pairs)} valid audio-transcript pairs (out of {len(wav_files)} wav files)")
    return pairs

def load_and_split():
    pairs = build_pairs()
    ds = Dataset.from_list(pairs)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # 80/10/10 split
    split1 = ds.train_test_split(test_size=0.20, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.50, seed=42)

    dataset = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })

    print({k: len(v) for k, v in dataset.items()})
    dataset.save_to_disk("/workspace/outputs/konkani_dataset")
    return dataset

if __name__ == "__main__":
    load_and_split()
