"""
Depression Detection — Multimodal Fusion
==========================================
Runs face analysis (DeepFace) and voice analysis (NLP) in parallel,
then fuses both scores into a final depression assessment.

Files needed in same folder:
  - face.py        (face module)
  - voice_nlp.py   (NLP analysis — assess_depression)
  - voice_stt.py   (STT questionnaire — run_questionnaire)

Run:
  python fusion.py
"""

import threading
import time
import json
from dataclasses import dataclass, field
from typing import Optional

# ── Local Modules ─────────────────────────────────────────────
from face import FaceAnalyzer, FaceAnalysisResult, extract_fusion_payload
from voice_nlp import assess_depression
from voice_stt import run_questionnaire


# ─────────────────────────────────────────────────────────────
# Fusion Weights
# ─────────────────────────────────────────────────────────────
# Face captures involuntary micro-expressions → reliable signal
# Voice captures semantic content → stronger clinical indicator
# Weights must sum to 1.0

FACE_WEIGHT  = 0.35
VOICE_WEIGHT = 0.65


# ─────────────────────────────────────────────────────────────
# Score Conversion
# ─────────────────────────────────────────────────────────────

def face_score_to_depression(avg_face_score: float) -> float:
    """Convert face positivity score → depression indicator (0–1)."""
    return round(1.0 - avg_face_score, 4)


# ─────────────────────────────────────────────────────────────
# Depression Level
# ─────────────────────────────────────────────────────────────

def get_depression_level(fused_score: float) -> tuple:
    """Return (level, action) based on fused 0–1 score."""
    if fused_score < 0.30:
        return (
            "Normal",
            "No immediate concern. Continue monitoring."
        )
    elif fused_score < 0.55:
        return (
            "Moderate",
            "Consider speaking to a counselor or trusted person."
        )
    else:
        return (
            "Severe",
            "Professional consultation strongly recommended. Please reach out for help."
        )


# ─────────────────────────────────────────────────────────────
# Result Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    # Face
    avg_face_score: float            = 0.0
    face_depression_score: float     = 0.0
    face_dominant_emotion: str       = "neutral"
    face_emotion_distribution: dict  = field(default_factory=dict)
    face_detected_ratio: float       = 0.0
    face_snapshots: int              = 0

    # Voice
    voice_normalized_score: float    = 0.0
    voice_depression_level: str      = "Normal"
    voice_total_weighted: float      = 0.0
    voice_per_question: list         = field(default_factory=list)

    # Fusion
    fused_score: float               = 0.0
    final_depression_level: str      = "Normal"
    recommended_action: str          = ""
    face_weight_used: float          = FACE_WEIGHT
    voice_weight_used: float         = VOICE_WEIGHT


# ─────────────────────────────────────────────────────────────
# Parallel Runner
# ─────────────────────────────────────────────────────────────

class DepressionDetector:
    """
    Runs face analysis and voice questionnaire in parallel,
    then fuses results into a final depression score.
    """

    def __init__(self, session_id: str = "session_001"):
        self.session_id        = session_id
        self._face_result:     Optional[FaceAnalysisResult] = None
        self._voice_result:    Optional[dict]               = None
        self._voice_responses: Optional[list]               = None
        self._face_error:      Optional[str]                = None
        self._voice_error:     Optional[str]                = None
        self._face_analyzer:   Optional[FaceAnalyzer]       = None

    # ── Public API ────────────────────────────

    def run(self) -> FusionResult:
        """
        Strategy:
          1. Start face analysis in background thread (headless, no cv2 window)
          2. Run voice questionnaire on main thread (needs terminal input)
          3. After voice questionnaire ends → stop face analysis
          4. Run NLP on voice responses
          5. Fuse both results
        """
        print("\n" + "=" * 65)
        print("  DEPRESSION DETECTION — Multimodal Analysis")
        print("  Face (DeepFace) + Voice (NLP)")
        print("=" * 65)

        # ── Step 1: Create face analyzer and start in background ──
        print("\n[Step 1/3] Starting face analysis in background...")
        self._face_analyzer = FaceAnalyzer(
            session_id=self.session_id,
            snapshot_interval=2.0,
            detector_backend="opencv",
            save_face_crops=False,
        )

        face_thread = threading.Thread(
            target=self._run_face,
            args=(self._face_analyzer,),
            daemon=True
        )
        face_thread.start()

        # Give camera time to open and warm up before voice starts
        print("[Fusion] Waiting for camera to initialise (5s)...")
        time.sleep(5)

        # ── Step 2: Voice questionnaire (blocking, main thread) ───
        print("\n[Step 2/3] Starting voice questionnaire...")
        print("           Answer each question aloud when prompted.")
        print("           Face analysis is running in the background.\n")

        try:
            self._voice_responses = run_questionnaire()
        except Exception as e:
            self._voice_error = str(e)
            print(f"[Voice] Error: {e}")

        # ── Step 3: Stop face analysis ────────────────────────────
        print("\n[Step 3/3] Voice done. Stopping face analysis...")
        self._face_analyzer._running = False          # signal analysis loop to exit
        face_thread.join(timeout=8)                   # wait for thread to finish

        if self._face_error:
            print(f"[Face] Warning: {self._face_error}")

        # ── Step 4: Run NLP on voice responses ────────────────────
        if self._voice_responses:
            try:
                self._voice_result = assess_depression(self._voice_responses)
            except Exception as e:
                self._voice_error = str(e)
                print(f"[Voice NLP] Error: {e}")
        else:
            print("[Voice] No responses captured — voice score set to 0.")

        # ── Step 5: Fuse ──────────────────────────────────────────
        return self._fuse()

    # ── Face Thread ───────────────────────────

    def _run_face(self, analyzer: FaceAnalyzer) -> None:
        """Runs in background thread. Calls headless start(), then stop()."""
        try:
            analyzer.start()                           # blocks until _running = False
            self._face_result = analyzer.stop()        # release camera + build result
        except Exception as e:
            self._face_error = str(e)
            print(f"[Face Thread] Error: {e}")
            # Attempt to release camera even on error
            try:
                self._face_result = analyzer.stop()
            except Exception:
                pass

    # ── Fusion ────────────────────────────────

    def _fuse(self) -> FusionResult:
        result = FusionResult()

        # ── Face values ───────────────────────
        if self._face_result and self._face_result.snapshots:
            fr = self._face_result
            result.avg_face_score           = fr.average_face_score
            result.face_depression_score    = face_score_to_depression(fr.average_face_score)
            result.face_dominant_emotion    = fr.dominant_emotion_overall
            result.face_emotion_distribution = fr.emotion_distribution
            result.face_detected_ratio      = fr.face_detected_ratio
            result.face_snapshots           = len(fr.snapshots)
        else:
            print("[Fusion] No face data. Using voice score only (weight = 1.0).")
            result.face_depression_score = 0.5   # neutral fallback

        # ── Voice values ──────────────────────
        if self._voice_result:
            vr = self._voice_result
            result.voice_normalized_score = vr["normalized_score"]
            result.voice_depression_level = vr["depression_level"]
            result.voice_total_weighted   = vr["total_weighted_score"]
            result.voice_per_question     = vr["per_question"]
        else:
            print("[Fusion] No voice data. Using face score only (weight = 1.0).")
            result.voice_normalized_score = 0.5  # neutral fallback

        # ── Weighted Fusion ───────────────────
        face_w  = FACE_WEIGHT
        voice_w = VOICE_WEIGHT

        if not self._face_result or not self._face_result.snapshots:
            face_w, voice_w = 0.0, 1.0
        elif not self._voice_result:
            face_w, voice_w = 1.0, 0.0

        fused = (result.face_depression_score * face_w) + (result.voice_normalized_score * voice_w)
        fused = round(min(max(fused, 0.0), 1.0), 4)

        result.fused_score        = fused
        result.face_weight_used   = face_w
        result.voice_weight_used  = voice_w

        level, action = get_depression_level(fused)
        result.final_depression_level = level
        result.recommended_action     = action

        return result


# ─────────────────────────────────────────────────────────────
# Print Final Report
# ─────────────────────────────────────────────────────────────

def print_report(result: FusionResult) -> None:
    W   = 65
    sep = "=" * W

    print(f"\n{sep}")
    print("  FINAL FUSION REPORT")
    print(sep)

    # Face
    print("\n  ── FACE ANALYSIS ──────────────────────────────────")
    print(f"  Avg Face Score        : {result.avg_face_score:.4f}  (positivity)")
    print(f"  Face Depression Score : {result.face_depression_score:.4f}  (1 - face score)")
    print(f"  Dominant Emotion      : {result.face_dominant_emotion.upper()}")
    print(f"  Snapshots Captured    : {result.face_snapshots}")
    print(f"  Face Detected Ratio   : {result.face_detected_ratio:.1%}")
    if result.face_emotion_distribution:
        print("  Emotion Distribution  :")
        for emo, prob in sorted(result.face_emotion_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 25)
            print(f"    {emo:<12} {bar:<25} {prob:.3f}")

    # Voice
    print("\n  ── VOICE / NLP ANALYSIS ───────────────────────────")
    print(f"  Voice Depression Score: {result.voice_normalized_score:.4f}")
    print(f"  Voice Level           : {result.voice_depression_level}")
    print(f"  Total Weighted Score  : {result.voice_total_weighted:.3f}")

    # Fusion
    print("\n  ── FUSION ──────────────────────────────────────────")
    print(f"  Face Weight           : {result.face_weight_used:.0%}")
    print(f"  Voice Weight          : {result.voice_weight_used:.0%}")
    print(f"  Fused Score           : {result.fused_score:.4f}  ({result.fused_score*100:.1f}%)")

    bar_len = int(result.fused_score * 40)
    bar     = "█" * bar_len + "░" * (40 - bar_len)
    level_icon = {"Normal": "✅", "Moderate": "⚠️ ", "Severe": "🚨"}.get(result.final_depression_level, "")

    print(f"\n  [{bar}] {result.fused_score*100:.1f}%")
    print(f"\n  {level_icon} Depression Level     : {result.final_depression_level.upper()}")
    print(f"  Recommended Action    : {result.recommended_action}")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────
# Export JSON
# ─────────────────────────────────────────────────────────────

def export_json(result: FusionResult, path: str = "fusion_result.json") -> None:
    data = {
        "session_id": "session_001",
        "face": {
            "avg_face_score":        result.avg_face_score,
            "face_depression_score": result.face_depression_score,
            "dominant_emotion":      result.face_dominant_emotion,
            "emotion_distribution":  result.face_emotion_distribution,
            "face_detected_ratio":   result.face_detected_ratio,
            "snapshots_captured":    result.face_snapshots,
        },
        "voice": {
            "normalized_score":    result.voice_normalized_score,
            "depression_level":    result.voice_depression_level,
            "total_weighted_score": result.voice_total_weighted,
        },
        "fusion": {
            "fused_score":             result.fused_score,
            "face_weight":             result.face_weight_used,
            "voice_weight":            result.voice_weight_used,
            "final_depression_level":  result.final_depression_level,
            "recommended_action":      result.recommended_action,
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Fusion] Result saved to {path}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = DepressionDetector(session_id="session_001")
    result   = detector.run()
    print_report(result)
    export_json(result)