"""
J.A.R.V.I.S. Diagnostic — Audio & Wake Word Troubleshooter
============================================================
Run this INSTEAD of main.py to diagnose wake word issues.

Usage:
    cd ~/jarvis
    python -m tests.diagnose_audio

It runs 4 tests:
  Test 1: Which mic is sounddevice using?
  Test 2: Is the mic actually capturing audio? (shows live RMS levels)
  Test 3: What chunk format does openwakeword expect vs what we send?
  Test 4: Live wake word detection with verbose confidence logging.
"""

import sys
import time

import numpy as np
import sounddevice as sd


def test_1_check_audio_device():
    """Test 1: What audio device is sounddevice using?"""
    print("\n" + "=" * 60)
    print("TEST 1: Audio Device Check")
    print("=" * 60)

    print(f"\nDefault input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")

    print("\nAll available input devices:")
    print("-" * 40)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <<<< DEFAULT" if i == sd.default.device[0] else ""
            print(
                f"  [{i}] {dev['name']} "
                f"(inputs={dev['max_input_channels']}, "
                f"rate={dev['default_samplerate']}){marker}"
            )

    print("\n✅ If you see 'MacBook Air Microphone' with <<<< DEFAULT, you're good.")
    print("   If it's pointing to something else, we need to set the device manually.")


def test_2_check_mic_levels():
    """Test 2: Record 5 seconds and show live audio levels."""
    print("\n" + "=" * 60)
    print("TEST 2: Live Microphone Levels (5 seconds)")
    print("=" * 60)
    print("\n🎙️  Speak, clap, or make noise NOW. You should see the bars move.\n")

    sample_rate = 16000
    chunk_samples = 1280  # 80ms at 16kHz
    duration = 5  # seconds
    num_chunks = int(duration * sample_rate / chunk_samples)

    max_rms = 0
    min_rms = float("inf")

    for i in range(num_chunks):
        audio = sd.rec(
            frames=chunk_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocking=True,
        )
        audio_flat = audio.flatten()
        rms = float(np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2)))

        max_rms = max(max_rms, rms)
        min_rms = min(min_rms, rms)

        # Visual bar
        bar_length = int(min(rms / 100, 50))
        bar = "█" * bar_length
        print(f"  RMS: {rms:7.1f} | {bar}", end="\r")

    print(f"\n\n📊 Results over {duration} seconds:")
    print(f"   Min RMS (silence): {min_rms:.1f}")
    print(f"   Max RMS (loudest): {max_rms:.1f}")
    print(f"   Current silence_threshold in config: 500")

    if max_rms < 100:
        print("\n❌ PROBLEM: Max RMS is very low. Your mic may not be working.")
        print("   → Check System Settings > Sound > Input")
        print("   → Make sure input volume is turned UP")
    elif max_rms < 500:
        print(f"\n⚠️  Your loudest sound ({max_rms:.0f}) is BELOW the silence threshold (500).")
        print(f"   → Lower silence_threshold to {int(max_rms * 0.3):.0f} in jarvis_config.yaml")
    else:
        print("\n✅ Mic levels look healthy.")

    return max_rms


def test_3_check_oww_format():
    """Test 3: Verify openwakeword receives the right audio format."""
    print("\n" + "=" * 60)
    print("TEST 3: openWakeWord Format Check")
    print("=" * 60)

    import openwakeword
    from openwakeword.model import Model as OWWModel

    print("\nDownloading/loading wake word model...")
    openwakeword.utils.download_models(["hey_jarvis"])

    model = OWWModel(
        wakeword_models=["hey_jarvis"],
        inference_framework="tflite",
    )

    # Record one chunk exactly as main.py does
    sample_rate = 16000
    chunk_samples = 1280

    audio = sd.rec(
        frames=chunk_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocking=True,
    )
    audio_flat = audio.flatten()

    print(f"\n  Audio chunk shape:  {audio_flat.shape}")
    print(f"  Audio dtype:        {audio_flat.dtype}")
    print(f"  Audio min/max:      {audio_flat.min()} / {audio_flat.max()}")
    print(f"  Expected shape:     (1280,)")
    print(f"  Expected dtype:     int16")

    # Try prediction
    prediction = model.predict(audio_flat)
    print(f"\n  Prediction output:  {prediction}")
    print(f"  Model keys:         {list(prediction.keys())}")

    # Check if model key matches what we expect
    for key in prediction.keys():
        print(f"  Model '{key}' confidence: {prediction[key]:.6f}")

    print("\n✅ If you see model keys and confidence values (even 0.000), the format is correct.")
    print("   The issue is likely threshold tuning, not format.")

    return model


def test_4_live_wake_word(model=None):
    """Test 4: Live wake word detection with verbose confidence logging."""
    print("\n" + "=" * 60)
    print("TEST 4: Live Wake Word Detection (15 seconds)")
    print("=" * 60)
    print("\n🎙️  Say 'HEY JARVIS' clearly, multiple times. Watch the confidence values.\n")

    import openwakeword
    from openwakeword.model import Model as OWWModel

    if model is None:
        openwakeword.utils.download_models(["hey_jarvis"])
        model = OWWModel(
            wakeword_models=["hey_jarvis"],
            inference_framework="tflite",
        )

    sample_rate = 16000
    chunk_samples = 1280
    duration = 15
    num_chunks = int(duration * sample_rate / chunk_samples)

    max_confidence = 0.0
    trigger_count = 0
    threshold = 0.5  # Our config threshold

    for i in range(num_chunks):
        audio = sd.rec(
            frames=chunk_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocking=True,
        )
        audio_flat = audio.flatten()
        prediction = model.predict(audio_flat)

        for model_name, confidence in prediction.items():
            max_confidence = max(max_confidence, confidence)

            if confidence > 0.01:  # Show any non-zero confidence
                bar_len = int(min(confidence * 50, 50))
                bar = "█" * bar_len
                status = "🔥 TRIGGERED!" if confidence > threshold else ""
                print(
                    f"  [{model_name}] conf={confidence:.4f} "
                    f"|{bar}| {status}"
                )
                if confidence > threshold:
                    trigger_count += 1
                    model.reset()

    print(f"\n📊 Results over {duration} seconds:")
    print(f"   Max confidence seen: {max_confidence:.4f}")
    print(f"   Trigger count (>{threshold}): {trigger_count}")
    print(f"   Current threshold in config: {threshold}")

    if max_confidence < 0.1:
        print("\n❌ Wake word never reached even 0.1 confidence.")
        print("   Possible causes:")
        print("   1. Mic input is too quiet — check Test 2 results")
        print("   2. Too much background noise")
        print("   3. Try speaking louder and closer to the mic")
        print(f"\n   💡 Try lowering threshold to 0.3 in jarvis_config.yaml")
    elif max_confidence < threshold:
        print(f"\n⚠️  Wake word peaked at {max_confidence:.3f} but threshold is {threshold}.")
        print(f"   → Lower wake_word threshold to {max(0.2, max_confidence * 0.7):.2f} in jarvis_config.yaml")
    else:
        print("\n✅ Wake word detection is working!")


def main():
    print("🔧 J.A.R.V.I.S. AUDIO DIAGNOSTIC TOOL")
    print("=" * 60)

    test_1_check_audio_device()

    input("\nPress Enter to start Test 2 (mic level check)...")
    max_rms = test_2_check_mic_levels()

    input("\nPress Enter to start Test 3 (format check)...")
    model = test_3_check_oww_format()

    input("\nPress Enter to start Test 4 (live wake word — 15 seconds)...")
    test_4_live_wake_word(model)

    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nShare the full output with Claude to get your fix.")


if __name__ == "__main__":
    main()