"""
J.A.R.V.I.S. — Wake Word Nuclear Fix
======================================
This script:
  1. Checks if onnxruntime is actually installed
  2. Clears ALL cached openWakeWord models (they may be stuck in tflite format)
  3. Re-downloads models in ONNX format
  4. Runs a live 20-second wake word test with verbose output

Run:
    cd ~/jarvis
    python tests/fix_wake_word.py
"""

import importlib
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd


def step_1_check_onnxruntime():
    """Verify onnxruntime is installed."""
    print("\n" + "=" * 60)
    print("STEP 1: Checking onnxruntime installation")
    print("=" * 60)

    try:
        import onnxruntime as ort
        print(f"  ✅ onnxruntime version: {ort.__version__}")
        print(f"  ✅ Available providers: {ort.get_available_providers()}")
        return True
    except ImportError:
        print("  ❌ onnxruntime is NOT installed!")
        print("  Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
        # Verify after install
        import onnxruntime as ort
        print(f"  ✅ onnxruntime installed: {ort.__version__}")
        return True


def step_2_clear_cached_models():
    """Delete all cached openWakeWord models to force fresh download."""
    print("\n" + "=" * 60)
    print("STEP 2: Clearing cached openWakeWord models")
    print("=" * 60)

    # openWakeWord caches models in ~/.local/share/openwakeword/ 
    # or in the package directory itself
    possible_cache_dirs = [
        os.path.expanduser("~/.local/share/openwakeword"),
        os.path.expanduser("~/openwakeword_models"),
    ]

    # Also check the package's own model directory
    try:
        import openwakeword
        oww_dir = os.path.dirname(openwakeword.__file__)
        oww_resources = os.path.join(oww_dir, "resources")
        possible_cache_dirs.append(oww_resources)
        print(f"  openWakeWord package location: {oww_dir}")
    except Exception as e:
        print(f"  Warning: couldn't find openwakeword package dir: {e}")

    for cache_dir in possible_cache_dirs:
        if os.path.exists(cache_dir):
            # List what's there before deleting
            print(f"\n  Found cache: {cache_dir}")
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    size = os.path.getsize(fpath) / 1024
                    ext = os.path.splitext(f)[1]
                    print(f"    {f} ({size:.1f}KB) [{ext}]")

            # Only delete model files, not the directory structure
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    if f.endswith((".tflite", ".onnx")):
                        fpath = os.path.join(root, f)
                        os.remove(fpath)
                        print(f"    🗑️  Deleted: {f}")
        else:
            print(f"  (not found: {cache_dir})")

    print("\n  ✅ Cache cleared")


def step_3_download_onnx_models():
    """Download models fresh, explicitly requesting ONNX format."""
    print("\n" + "=" * 60)
    print("STEP 3: Downloading hey_jarvis model (ONNX format)")
    print("=" * 60)

    import openwakeword

    # Force download — specifying the model name
    print("  Downloading 'hey_jarvis' model...")
    openwakeword.utils.download_models(["hey_jarvis"])
    print("  ✅ Model downloaded")

    # Now verify we can load it with ONNX
    print("\n  Loading model with onnx inference framework...")
    try:
        from openwakeword.model import Model as OWWModel
        model = OWWModel(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx",
        )
        print("  ✅ Model loaded successfully with ONNX framework!")

        # Do a quick test prediction with silence
        test_audio = np.zeros(1280, dtype=np.int16)
        pred = model.predict(test_audio)
        print(f"  Test prediction (silence): {pred}")
        print(f"  ✅ Inference is running — got prediction output")

        return model

    except Exception as e:
        print(f"\n  ❌ Failed to load with 'onnx' framework: {e}")
        print("\n  Trying without specifying framework (let openWakeWord auto-detect)...")

        try:
            from openwakeword.model import Model as OWWModel
            model = OWWModel(wakeword_models=["hey_jarvis"])
            test_audio = np.zeros(1280, dtype=np.int16)
            pred = model.predict(test_audio)
            print(f"  Test prediction (silence): {pred}")
            print(f"  ✅ Model loaded with auto-detect framework")
            return model
        except Exception as e2:
            print(f"  ❌ Auto-detect also failed: {e2}")
            return None


def step_4_live_test(model):
    """20-second live wake word test with maximum verbosity."""
    print("\n" + "=" * 60)
    print("STEP 4: Live Wake Word Test (20 seconds)")
    print("=" * 60)
    print("\n🎙️  Say 'HEY JARVIS' clearly. Speak at normal volume,")
    print("   about 1 foot from the laptop. Try it 5-6 times.\n")
    print("   Every prediction is printed — even zeros.\n")

    input("Press Enter when ready...")

    sample_rate = 16000
    chunk_samples = 1280
    duration = 20
    num_chunks = int(duration * sample_rate / chunk_samples)

    max_conf = 0.0
    non_zero_count = 0

    for i in range(num_chunks):
        audio = sd.rec(
            frames=chunk_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocking=True,
        )
        audio_flat = audio.flatten()

        # Check audio is actually non-silent
        rms = float(np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2)))

        prediction = model.predict(audio_flat)

        for model_name, confidence in prediction.items():
            max_conf = max(max_conf, confidence)
            if confidence > 0.0:
                non_zero_count += 1

            # Print EVERY 25th chunk (roughly every 2 sec) regardless,
            # and every chunk where confidence > 0
            if i % 25 == 0 or confidence > 0.0:
                bar_len = int(min(confidence * 50, 50))
                bar = "█" * bar_len if bar_len > 0 else "·"
                print(
                    f"  chunk {i:4d} | rms={rms:7.1f} | "
                    f"conf={confidence:.6f} | {bar}"
                )

    print(f"\n📊 RESULTS:")
    print(f"   Total chunks processed:    {num_chunks}")
    print(f"   Non-zero confidence count: {non_zero_count}")
    print(f"   Max confidence:            {max_conf:.6f}")

    if max_conf == 0.0 and non_zero_count == 0:
        print("\n❌ DIAGNOSIS: Model inference is completely dead.")
        print("   The model is loaded but always returns 0.0.")
        print("   This is an openWakeWord + framework compatibility issue.")
        print("\n   🔧 RECOMMENDED FIX: Skip openWakeWord entirely.")
        print("   Use mlx-whisper as the wake word detector instead.")
        print("   (Always-on STT with keyword matching — more reliable on M1)")
    elif max_conf < 0.1:
        print(f"\n⚠️  Model is producing signal but very low ({max_conf:.4f}).")
        print(f"   Try threshold of {max(0.05, max_conf * 0.7):.3f}")
    else:
        print(f"\n✅ Model is working! Set threshold to {max_conf * 0.6:.2f}")


def main():
    print("🔧 J.A.R.V.I.S. — WAKE WORD NUCLEAR FIX")
    print("=" * 60)

    step_1_check_onnxruntime()
    step_2_clear_cached_models()
    model = step_3_download_onnx_models()

    if model is None:
        print("\n💀 Could not load wake word model at all.")
        print("   Recommendation: Switch to STT-based wake word detection.")
        return

    step_4_live_test(model)

    print("\n" + "=" * 60)
    print("🏁 DONE — Copy this entire output and share it with Claude.")
    print("=" * 60)


if __name__ == "__main__":
    main()