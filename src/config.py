import torch

def detect_hardware():
    if not torch.cuda.is_available():
        return {"device": "cpu", "vram_gb": 0, "use_8bit": False, "fp16": False}

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    return {
        "device": "cuda",
        "vram_gb": round(vram_gb, 1),
        "gpu_name": props.name,
        "use_8bit": True,   # always use 8-bit on a 6GB card
        "fp16": True,
    }

def get_training_config():
    hw = detect_hardware()
    print(f"Detected hardware: {hw}")

    if hw["device"] == "cpu":
        print("No GPU detected. This config is for pipeline verification only.")
        return {"batch_size": 1, "grad_accum": 16, "fp16": False, "use_8bit": False}

    if hw["vram_gb"] < 8:
        # Tuned for your RTX 3050 (6GB) specifically
        return {"batch_size": 2, "grad_accum": 8, "fp16": True, "use_8bit": True}
    elif hw["vram_gb"] < 16:
        return {"batch_size": 4, "grad_accum": 4, "fp16": True, "use_8bit": True}
    else:
        return {"batch_size": 8, "grad_accum": 2, "fp16": True, "use_8bit": True}

if __name__ == "__main__":
    print(get_training_config())

