"""
J.A.R.V.I.S. Phase 5 — Vision Tool
===================================
Screen OCR  → macOS native Vision framework (0 RAM, ~200ms)
Describe    → LLaVA-Phi3 via Ollama (~2.5GB, auto-swapped)
Webcam      → imagesnap + LLaVA-Phi3

Three actions:
  - ocr            → read text on screen (native macOS, free)
  - describe_screen → screenshot + LLaVA-Phi3 analysis
  - describe_webcam → webcam frame + LLaVA-Phi3 analysis

RAM strategy:
  Ollama auto-swaps models. When vision is needed, llava-phi3 loads
  (unloading phi3:mini). After vision query, we set keep_alive=30s
  so it unloads quickly and phi3:mini can reload for chat.
"""

import subprocess
import base64
import os
import re
import logging
import time
import requests

logger = logging.getLogger(__name__)

# ── Dependency checks ───────────────────────────────────────

# Native OCR via macOS Vision framework (PyObjC)
NATIVE_OCR_AVAILABLE = False
try:
    import Quartz
    from Foundation import NSURL
    import Vision
    NATIVE_OCR_AVAILABLE = True
    logger.info("✅ macOS Vision framework loaded — native OCR enabled (0 RAM)")
except ImportError:
    logger.warning(
        "⚠️  PyObjC not installed — OCR will fall back to LLaVA-Phi3 (slower). "
        "Install with: pip install pyobjc-framework-Vision pyobjc-framework-Quartz"
    )

# Webcam capture via imagesnap
IMAGESNAP_AVAILABLE = False
try:
    result = subprocess.run(["which", "imagesnap"], capture_output=True, text=True)
    IMAGESNAP_AVAILABLE = result.returncode == 0
    if IMAGESNAP_AVAILABLE:
        logger.info("✅ imagesnap found — webcam capture enabled")
    else:
        logger.warning("⚠️  imagesnap not found. Install: brew install imagesnap")
except Exception:
    logger.warning("⚠️  Could not check for imagesnap")

# Image resizing via Pillow (optional, for faster LLaVA inference)
PILLOW_AVAILABLE = False
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    logger.info("ℹ️  Pillow not installed — screenshots sent at full resolution")


# ── Constants ───────────────────────────────────────────────

SCREENSHOT_PATH = "/tmp/jarvis_vision_screen.png"
WEBCAM_PATH = "/tmp/jarvis_vision_webcam.jpg"
MAX_IMAGE_DIM = 1024  # Resize longest side to this before sending to LLaVA
VISION_KEEP_ALIVE = 30  # Seconds to keep llava-phi3 loaded after use


class VisionTool:
    """
    Handles all vision-related commands for J.A.R.V.I.S.
    
    Three capabilities:
      1. OCR (read text) — uses macOS native Vision framework, zero RAM
      2. Describe screen — screenshots + LLaVA-Phi3 via Ollama
      3. Describe webcam — imagesnap + LLaVA-Phi3 via Ollama
    """

    def __init__(self, config: dict):
        """
        Config keys used:
          - vision_model: Ollama model name (default: "llava-phi3")
          - ollama_base_url: Ollama API base (default: "http://localhost:11434")
          - vision_timeout: Max seconds to wait for LLaVA response (default: 90)
        """
        self.vision_model = config.get("vision_model", "llava-phi3")
        self.ollama_base = config.get("ollama_base_url", "http://localhost:11434")
        self.timeout = config.get("vision_timeout", 90)

        # Stores raw unformatted result for NLU reasoning
        # (OCR: full text before truncation, describe: full LLaVA response)
        self.last_raw_result = ""

        logger.info(
            f"🔭 Vision tool initialized — model: {self.vision_model}, "
            f"native_ocr: {NATIVE_OCR_AVAILABLE}, webcam: {IMAGESNAP_AVAILABLE}"
        )

    # ════════════════════════════════════════════════════════
    #  PUBLIC: execute() — called by the tool router
    # ════════════════════════════════════════════════════════

    def execute(self, action: str, params: dict) -> str:
        """
        Route to the correct vision action.
        
        Actions:
          - "ocr"             → read text from screen
          - "describe_screen" → describe what's on screen
          - "describe_webcam" → describe webcam view
        
        Returns a string response suitable for TTS.
        """
        try:
            if action == "ocr":
                return self._ocr_screen()
            elif action == "describe_screen":
                question = params.get("question", "Describe what you see on this screen.")
                return self._describe_screen(question)
            elif action == "describe_webcam":
                question = params.get("question", "Describe what you see.")
                return self._describe_webcam(question)
            else:
                return f"Unknown vision action: {action}"
        except FileNotFoundError as e:
            logger.error(f"❌ Vision file error: {e}")
            return f"Sorry, I couldn't capture the image. {str(e)}"
        except requests.exceptions.ConnectionError:
            logger.error("❌ Ollama not running")
            return "I can't reach Ollama. Make sure it's running with: ollama serve"
        except requests.exceptions.Timeout:
            logger.error("❌ Vision model timed out")
            return "The vision model took too long. It might still be loading — try again in a moment."
        except Exception as e:
            logger.error(f"❌ Vision error: {type(e).__name__}: {e}")
            return f"Vision error: {str(e)}"

    # ════════════════════════════════════════════════════════
    #  CAPTURE: screenshot + webcam
    # ════════════════════════════════════════════════════════

    def _take_screenshot(self) -> str:
        """
        Capture the screen using macOS screencapture.
        
        Flags:
          -x  = no shutter sound
          -C  = include cursor
        
        Returns the file path.
        """
        try:
            subprocess.run(
                ["screencapture", "-x", "-C", SCREENSHOT_PATH],
                check=True,
                timeout=5
            )
        except subprocess.TimeoutExpired:
            raise FileNotFoundError("Screenshot timed out")
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(f"screencapture failed: {e}")

        if not os.path.exists(SCREENSHOT_PATH):
            raise FileNotFoundError("Screenshot file not created")

        size_kb = os.path.getsize(SCREENSHOT_PATH) // 1024
        logger.info(f"📸 Screenshot captured: {size_kb}KB")
        return SCREENSHOT_PATH

    def _capture_webcam(self) -> str:
        """
        Capture a single webcam frame using imagesnap.
        
        -w 1.5 = 1.5s warm-up for camera auto-exposure.
        
        Requires: brew install imagesnap
        Also requires camera permission in System Preferences > Privacy > Camera.
        """
        if not IMAGESNAP_AVAILABLE:
            raise FileNotFoundError(
                "Webcam capture requires imagesnap. Install with: brew install imagesnap"
            )

        try:
            subprocess.run(
                ["imagesnap", "-w", "1.5", WEBCAM_PATH],
                check=True,
                timeout=10
            )
        except subprocess.TimeoutExpired:
            raise FileNotFoundError("Webcam capture timed out")
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(f"imagesnap failed: {e}")

        if not os.path.exists(WEBCAM_PATH):
            raise FileNotFoundError("Webcam file not created")

        size_kb = os.path.getsize(WEBCAM_PATH) // 1024
        logger.info(f"📷 Webcam frame captured: {size_kb}KB")
        return WEBCAM_PATH

    # ════════════════════════════════════════════════════════
    #  NATIVE OCR: macOS Vision framework (zero RAM)
    # ════════════════════════════════════════════════════════

    def _native_ocr(self, image_path: str) -> str:
        """
        Extract text from image using Apple's VNRecognizeTextRequest.
        
        This uses the macOS Vision framework via PyObjC.
        Zero additional RAM — it's built into the OS.
        Accuracy is excellent (same engine as macOS text selection in images).
        
        Returns extracted text as a string (one line per detected text block).
        """
        # Load image as CGImage via Quartz
        image_url = NSURL.fileURLWithPath_(image_path)
        image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
        if image_source is None:
            raise ValueError(f"Could not load image source: {image_path}")

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            raise ValueError(f"Could not create CGImage: {image_path}")

        # Create text recognition request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        # 0 = VNRequestTextRecognitionLevelAccurate (slower but better)
        # 1 = VNRequestTextRecognitionLevelFast
        request.setRecognitionLevel_(0)
        request.setUsesLanguageCorrection_(True)

        # Create handler and perform request
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        success = handler.performRequests_error_([request], None)

        # PyObjC may return (bool, error) tuple or just bool
        if isinstance(success, tuple):
            success, error = success
            if not success:
                raise RuntimeError(f"OCR request failed: {error}")
        elif not success:
            raise RuntimeError("OCR request failed")

        # Extract recognized text
        results = request.results()
        if not results or len(results) == 0:
            return ""

        lines = []
        for observation in results:
            candidates = observation.topCandidates_(1)
            if candidates and len(candidates) > 0:
                text = candidates[0].string()
                confidence = candidates[0].confidence()
                # Only include text with reasonable confidence
                if confidence > 0.3:
                    lines.append(text)

        text = "\n".join(lines)
        logger.info(f"📝 Native OCR: {len(lines)} lines, {len(text)} chars")
        return text

    # ════════════════════════════════════════════════════════
    #  LLAVA: send image to vision model via Ollama
    # ════════════════════════════════════════════════════════

    def _resize_image(self, image_path: str) -> str:
        """
        Resize image so longest side <= MAX_IMAGE_DIM.
        
        Reduces base64 payload size dramatically:
          - Full retina screenshot: ~5MB PNG → ~2MB base64
          - Resized to 1024px: ~200KB PNG → ~270KB base64
        
        This makes LLaVA inference 3-5x faster.
        Returns path to resized image (or original if Pillow unavailable).
        """
        if not PILLOW_AVAILABLE:
            return image_path

        try:
            img = Image.open(image_path)
            w, h = img.size

            # Skip if already small enough
            if max(w, h) <= MAX_IMAGE_DIM:
                return image_path

            # Calculate new dimensions preserving aspect ratio
            if w > h:
                new_w = MAX_IMAGE_DIM
                new_h = int(h * (MAX_IMAGE_DIM / w))
            else:
                new_h = MAX_IMAGE_DIM
                new_w = int(w * (MAX_IMAGE_DIM / h))

            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            resized_path = image_path.replace(".png", "_resized.png").replace(
                ".jpg", "_resized.jpg"
            )
            img_resized.save(resized_path, quality=85)

            orig_kb = os.path.getsize(image_path) // 1024
            new_kb = os.path.getsize(resized_path) // 1024
            logger.info(f"🔄 Resized: {w}x{h} ({orig_kb}KB) → {new_w}x{new_h} ({new_kb}KB)")

            return resized_path
        except Exception as e:
            logger.warning(f"⚠️  Resize failed, using original: {e}")
            return image_path

    def _encode_image(self, image_path: str) -> str:
        """Read image file and return base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _ask_vision_model(self, image_path: str, prompt: str) -> str:
        """
        Send image + prompt to LLaVA-Phi3 via Ollama's /api/chat endpoint.
        
        Uses /api/chat (not /api/generate) because it supports the messages
        format with images embedded in the user message.
        
        Sets keep_alive to a short duration so the vision model unloads
        quickly after use, freeing RAM for phi3:mini to reload.
        """
        # Resize for faster inference
        resized_path = self._resize_image(image_path)
        image_b64 = self._encode_image(resized_path)

        payload_kb = len(image_b64) // 1024
        logger.info(
            f"🧠 Sending to {self.vision_model} "
            f"(image: {payload_kb}KB base64, may take 10-30s on first load)..."
        )
        start = time.time()

        response = requests.post(
            f"{self.ollama_base}/api/chat",
            json={
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "temperature": 0.3,
                },
                # Unload vision model after 30s to free RAM for phi3:mini
                "keep_alive": f"{VISION_KEEP_ALIVE}s",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        result = data.get("message", {}).get("content", "").strip()

        elapsed = time.time() - start
        logger.info(f"🔭 Vision response in {elapsed:.1f}s ({len(result)} chars)")

        # Store raw response for NLU reasoning
        self.last_raw_result = result

        # Clean up resized temp file
        if resized_path != image_path and os.path.exists(resized_path):
            os.remove(resized_path)

        return result

    # ════════════════════════════════════════════════════════
    #  ACTIONS: the three public capabilities
    # ════════════════════════════════════════════════════════

    def _ocr_screen(self) -> str:
        """
        Action: "ocr" — Read text from the screen.
        
        Strategy:
          1. Take screenshot
          2. If PyObjC available → use native macOS OCR (instant, 0 RAM)
          3. Else → send to LLaVA-Phi3 with OCR prompt (slower)
        
        For TTS, we truncate long text and summarize.
        """
        path = self._take_screenshot()

        if NATIVE_OCR_AVAILABLE:
            logger.info("📝 Using native macOS Vision OCR (0 RAM)")
            text = self._native_ocr(path)

            if not text or text.strip() == "":
                self.last_raw_result = ""
                return "I couldn't detect any text on your screen."

            # Store FULL raw text for NLU reasoning (before TTS truncation)
            self.last_raw_result = text.strip()

            # For TTS: don't read 500 lines of code — summarize
            lines = text.strip().split("\n")
            num_lines = len(lines)

            if num_lines <= 10:
                # Short enough to read fully
                return f"I can see the following text on your screen: {text.strip()}"
            elif num_lines <= 30:
                # Read first few lines + count
                preview = "\n".join(lines[:8])
                return (
                    f"I can see {num_lines} lines of text on your screen. "
                    f"Here's the start: {preview}"
                )
            else:
                # Too long — just summarize
                preview = "\n".join(lines[:5])
                return (
                    f"There's a lot of text on your screen — {num_lines} lines total. "
                    f"It starts with: {preview}"
                )
        else:
            # Fallback to LLaVA
            logger.info("📝 Using LLaVA-Phi3 for OCR (native unavailable)")
            return self._ask_vision_model(
                path,
                "Read all the text visible on this screen. "
                "List the most important text you can see. Be concise."
            )

    def _describe_screen(self, question: str) -> str:
        """
        Action: "describe_screen" — Describe what's on screen.
        
        Always uses LLaVA-Phi3 (needs visual understanding, not just text).
        
        The user's original question is passed through so the model
        can answer specifically (e.g., "What app is open?" vs "What's on screen?").
        """
        path = self._take_screenshot()

        prompt = (
            "You are Jarvis, a helpful AI assistant. "
            "The user is showing you their screen. "
            f"Answer this question about what you see: {question}\n\n"
            "Rules:\n"
            "- Be concise: 2-3 sentences maximum\n"
            "- Be specific about what you see (app names, content, etc.)\n"
            "- Don't describe UI elements that aren't relevant to the question"
        )

        return self._ask_vision_model(path, prompt)

    def _describe_webcam(self, question: str) -> str:
        """
        Action: "describe_webcam" — Describe what the webcam sees.
        
        Uses imagesnap to capture one frame, then LLaVA-Phi3 to analyze.
        
        Note: First call may trigger macOS camera permission dialog.
        The user needs to grant Terminal/Python camera access in:
          System Preferences > Privacy & Security > Camera
        """
        path = self._capture_webcam()

        prompt = (
            "You are Jarvis, a friendly AI assistant looking through a webcam. "
            f"{question}\n\n"
            "Rules:\n"
            "- Be concise: 2-3 sentences maximum\n"
            "- Be friendly and conversational\n"
            "- Describe what you see naturally, like talking to a friend\n"
            "- If you see a person, describe what they're doing, not their appearance"
        )

        return self._ask_vision_model(path, prompt)

    # ════════════════════════════════════════════════════════
    #  CLEANUP
    # ════════════════════════════════════════════════════════

    def cleanup(self):
        """Remove temporary image files."""
        for path in [SCREENSHOT_PATH, WEBCAM_PATH]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"🧹 Cleaned up: {path}")
            except OSError:
                pass

        # Also clean resized variants
        for suffix in ["_resized.png", "_resized.jpg"]:
            for base in [SCREENSHOT_PATH, WEBCAM_PATH]:
                resized = base.replace(".png", suffix).replace(".jpg", suffix)
                try:
                    if os.path.exists(resized):
                        os.remove(resized)
                except OSError:
                    pass