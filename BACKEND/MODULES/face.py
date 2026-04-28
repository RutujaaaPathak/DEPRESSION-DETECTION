"""
Face Analysis Model using DeepFace
====================================
- Opens default laptop camera automatically
- Detects face + analyzes emotions in real time
- Supports both standalone (with preview window) and headless (fusion) mode
- Press ENTER to stop and get average face score
"""

import cv2
import time
import base64
import threading
import numpy as np
from deepface import DeepFace
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Emotion → Score Mapping
# ─────────────────────────────────────────────

EMOTION_SCORES = {
    "happy":    1.0,
    "surprise": 0.6,
    "neutral":  0.5,
    "fear":     0.2,
    "sad":      0.1,
    "angry":    0.0,
    "disgust":  0.0,
}

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class EmotionSnapshot:
    timestamp: float
    dominant_emotion: str
    emotion_scores: dict
    normalized_emotions: dict
    face_score: float
    confidence: float
    frame_index: int
    snapshot_b64: Optional[str] = None


@dataclass
class FaceAnalysisResult:
    session_id: str
    snapshots: list = field(default_factory=list)
    average_face_score: float = 0.0
    dominant_emotion_overall: str = "neutral"
    emotion_distribution: dict = field(default_factory=dict)
    total_frames_analyzed: int = 0
    session_duration_seconds: float = 0.0
    face_detected_ratio: float = 0.0


# ─────────────────────────────────────────────
# Core Face Analyzer
# ─────────────────────────────────────────────

class FaceAnalyzer:
    def __init__(
        self,
        session_id: str = "session_001",
        snapshot_interval: float = 2.0,
        detector_backend: str = "opencv",
        save_face_crops: bool = True,
        frame_skip: int = 2,
    ):
        self.session_id        = session_id
        self.snapshot_interval = snapshot_interval
        self.detector_backend  = detector_backend
        self.save_face_crops   = save_face_crops
        self.frame_skip        = frame_skip

        self._cap: Optional[cv2.VideoCapture] = None
        self._running   = False
        self._lock      = threading.Lock()
        self._stop_event = threading.Event()

        self._snapshots:              list  = []
        self._frame_index:            int   = 0
        self._total_frames:           int   = 0
        self._face_detected_frames:   int   = 0
        self._last_snapshot_time:     float = 0.0
        self._start_time:             float = 0.0
        self._emotion_buffer: deque         = deque(maxlen=5)

        # Overlay state (updated by analysis thread, read by display thread)
        self._overlay_emotion = "Warming up..."
        self._overlay_score   = 0.0
        self._overlay_count   = 0
        self._overlay_avg     = 0.0
        self._overlay_color   = (180, 180, 180)

    # ─────────────────────────────────────────
    # PUBLIC: Standalone mode (with preview window)
    # ─────────────────────────────────────────

    def run(self) -> FaceAnalysisResult:
        """
        Opens camera with a live OpenCV window.
        Press ENTER (terminal) or Q (window) to stop.
        Use this when running face.py directly.
        """
        print("\n[FaceAnalyzer] Opening default camera...")
        self._open_camera()

        ret, test_frame = self._cap.read()
        if not ret or test_frame is None:
            raise RuntimeError("Camera opened but cannot read frames. Close Teams/Zoom/etc and retry.")

        h, w = test_frame.shape[:2]
        print(f"[FaceAnalyzer] Camera OK — {w}x{h}")
        print("[FaceAnalyzer] Loading DeepFace models (first run ~10s)...")
        self._warmup(test_frame)

        self._running             = True
        self._start_time          = time.time()
        self._last_snapshot_time  = self._start_time

        # ENTER key listener
        enter_thread = threading.Thread(target=self._wait_for_enter, daemon=True)
        enter_thread.start()

        # Analysis in background
        analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        analysis_thread.start()

        print("\n[FaceAnalyzer] Running. Press ENTER or Q to stop.\n")

        # Main thread: display window
        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            self._draw_overlay(frame)
            cv2.imshow("Face Analysis  —  Press Q or ENTER to stop", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 13:
                print("\n[FaceAnalyzer] Stopped.")
                self._running = False
                break

            if self._stop_event.is_set():
                break

        self._running = False
        self._cap.release()
        cv2.destroyAllWindows()
        analysis_thread.join(timeout=3)
        return self._build_result()

    # ─────────────────────────────────────────
    # PUBLIC: Headless mode (called from fusion.py background thread)
    # ─────────────────────────────────────────

    def start(self) -> None:
        """
        Headless start — no cv2 window.
        Blocks on the calling thread until self._running is set False externally.
        Used by DepressionDetector in fusion.py.
        """
        print("\n[FaceAnalyzer] Opening camera (headless mode)...")
        self._open_camera()

        ret, test_frame = self._cap.read()
        if not ret or test_frame is None:
            raise RuntimeError("Camera opened but cannot read frames. Close Teams/Zoom/etc and retry.")

        h, w = test_frame.shape[:2]
        print(f"[FaceAnalyzer] Camera OK — {w}x{h}")
        print("[FaceAnalyzer] Loading DeepFace models (first run ~10s)...")
        self._warmup(test_frame)

        self._running            = True
        self._start_time         = time.time()
        self._last_snapshot_time = self._start_time

        print("[FaceAnalyzer] Headless analysis started (running in background).\n")

        # Run analysis loop directly on this thread — blocks until _running = False
        self._analysis_loop()

    def stop(self) -> FaceAnalysisResult:
        """
        Stop the analyzer and return the collected result.
        Call this from the main thread after setting _running = False.
        """
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            print("[FaceAnalyzer] Camera released.")
        return self._build_result()

    # ─────────────────────────────────────────
    # PRIVATE: Camera open helper
    # ─────────────────────────────────────────

    def _open_camera(self) -> None:
        self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        if not self._cap.isOpened():
            # Fallback without CAP_DSHOW
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise RuntimeError(
                    "Could not open default camera. "
                    "Make sure no other app (Teams, Zoom) is using it."
                )

    # ─────────────────────────────────────────
    # PRIVATE: Enter key listener
    # ─────────────────────────────────────────

    def _wait_for_enter(self) -> None:
        input()
        print("\n[FaceAnalyzer] ENTER pressed — stopping...")
        self._running = False
        self._stop_event.set()

    # ─────────────────────────────────────────
    # PRIVATE: Warmup
    # ─────────────────────────────────────────

    def _warmup(self, frame: np.ndarray) -> None:
        try:
            DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )
            print("[FaceAnalyzer] Models loaded OK.")
        except Exception as e:
            print(f"[FaceAnalyzer] Warmup note: {e}")

    # ─────────────────────────────────────────
    # PRIVATE: Analysis loop (runs on background thread)
    # ─────────────────────────────────────────

    def _analysis_loop(self) -> None:
        while self._running:
            now = time.time()
            if (now - self._last_snapshot_time) < self.snapshot_interval:
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            self._total_frames += 1
            self._frame_index  += 1

            snapshot = self._analyze_frame(frame.copy(), now)
            if snapshot:
                self._face_detected_frames += 1
                self._last_snapshot_time    = now
                with self._lock:
                    self._snapshots.append(snapshot)
                    avg = sum(s.face_score for s in self._snapshots) / len(self._snapshots)
                    self._overlay_emotion = snapshot.dominant_emotion
                    self._overlay_score   = snapshot.face_score
                    self._overlay_count   = len(self._snapshots)
                    self._overlay_avg     = round(avg, 4)
                    self._overlay_color   = self._emotion_color(snapshot.dominant_emotion)

                print(
                    f"  [#{self._overlay_count:>3}] "
                    f"{snapshot.dominant_emotion.upper():<10} | "
                    f"score={snapshot.face_score:.3f} | "
                    f"avg={self._overlay_avg:.3f} | "
                    f"t={snapshot.timestamp:.1f}s"
                )
            else:
                self._last_snapshot_time = now
                with self._lock:
                    self._overlay_emotion = "No face detected"
                    self._overlay_color   = (0, 100, 255)

    # ─────────────────────────────────────────
    # PRIVATE: Analyze single frame
    # ─────────────────────────────────────────

    def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> Optional[EmotionSnapshot]:
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=True,
                silent=True,
            )
            analysis     = results[0] if isinstance(results, list) else results
            raw_emotions: dict = analysis["emotion"]
            dominant: str      = analysis["dominant_emotion"]
            region: dict       = analysis.get("region", {})

            total = sum(raw_emotions.values()) or 1.0
            norm_emotions = {k: v / total for k, v in raw_emotions.items()}

            face_score = round(
                min(max(
                    sum(EMOTION_SCORES.get(e, 0.5) * p for e, p in norm_emotions.items()),
                    0.0), 1.0),
                4
            )
            confidence = round(norm_emotions.get(dominant, 0.0), 4)

            self._emotion_buffer.append(norm_emotions)
            smoothed = {
                k: sum(e.get(k, 0.0) for e in self._emotion_buffer) / len(self._emotion_buffer)
                for k in norm_emotions
            }

            snapshot_b64 = (
                self._crop_and_encode(frame, region)
                if (self.save_face_crops and region) else None
            )

            return EmotionSnapshot(
                timestamp=round(timestamp - self._start_time, 3),
                dominant_emotion=dominant,
                emotion_scores={k: round(v, 2) for k, v in raw_emotions.items()},
                normalized_emotions={k: round(v, 4) for k, v in smoothed.items()},
                face_score=face_score,
                confidence=confidence,
                frame_index=self._frame_index,
                snapshot_b64=snapshot_b64,
            )
        except Exception:
            return None

    def _crop_and_encode(self, frame: np.ndarray, region: dict) -> Optional[str]:
        try:
            x, y, w, h  = region["x"], region["y"], region["w"], region["h"]
            pad_w, pad_h = int(w * 0.2), int(h * 0.2)
            x1, y1 = max(0, x - pad_w),          max(0, y - pad_h)
            x2, y2 = min(frame.shape[1], x+w+pad_w), min(frame.shape[0], y+h+pad_h)
            _, buf  = cv2.imencode(".jpg", frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode("utf-8")
        except Exception:
            return None

    # ─────────────────────────────────────────
    # PRIVATE: Overlay (standalone mode only)
    # ─────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        bar  = frame.copy()
        cv2.rectangle(bar, (0, h - 90), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

        with self._lock:
            emotion = self._overlay_emotion
            score   = self._overlay_score
            count   = self._overlay_count
            avg     = self._overlay_avg
            color   = self._overlay_color

        cv2.putText(frame, f"Emotion: {emotion.upper()}", (12, h - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
        cv2.putText(frame, f"Frame Score: {score:.3f}", (12, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"Avg Score: {avg:.3f}   |   Snapshots: {count}   |   Press Q or ENTER to stop",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    @staticmethod
    def _emotion_color(emotion: str) -> tuple:
        return {
            "happy":    (0, 220, 80),
            "surprise": (0, 200, 255),
            "neutral":  (180, 180, 180),
            "fear":     (0, 140, 255),
            "sad":      (200, 100, 50),
            "angry":    (0, 50, 255),
            "disgust":  (50, 0, 200),
        }.get(emotion, (180, 180, 180))

    # ─────────────────────────────────────────
    # PRIVATE: Build result from collected snapshots
    # ─────────────────────────────────────────

    def _build_result(self) -> FaceAnalysisResult:
        with self._lock:
            snapshots = list(self._snapshots)

        if not snapshots:
            return FaceAnalysisResult(session_id=self.session_id)

        avg_score = round(sum(s.face_score for s in snapshots) / len(snapshots), 4)

        emotion_totals: dict = {}
        for snap in snapshots:
            for emo, prob in snap.normalized_emotions.items():
                emotion_totals[emo] = emotion_totals.get(emo, 0.0) + prob

        n = len(snapshots)
        emotion_dist     = {k: round(v / n, 4) for k, v in emotion_totals.items()}
        dominant_overall = max(emotion_dist, key=emotion_dist.get) if emotion_dist else "neutral"
        face_ratio       = self._face_detected_frames / max(self._total_frames, 1)

        return FaceAnalysisResult(
            session_id=self.session_id,
            snapshots=snapshots,
            average_face_score=avg_score,
            dominant_emotion_overall=dominant_overall,
            emotion_distribution=emotion_dist,
            total_frames_analyzed=self._frame_index,
            session_duration_seconds=round(time.time() - self._start_time, 2),
            face_detected_ratio=round(face_ratio, 4),
        )


# ─────────────────────────────────────────────
# Fusion-Ready Output
# ─────────────────────────────────────────────

def extract_fusion_payload(result: FaceAnalysisResult) -> dict:
    return {
        "session_id": result.session_id,
        "face": {
            "average_score":              result.average_face_score,
            "dominant_emotion":           result.dominant_emotion_overall,
            "emotion_distribution":       result.emotion_distribution,
            "face_detected_ratio":        result.face_detected_ratio,
            "total_snapshots":            len(result.snapshots),
            "session_duration_seconds":   result.session_duration_seconds,
            "timeline": [
                {
                    "t":         s.timestamp,
                    "emotion":   s.dominant_emotion,
                    "score":     s.face_score,
                    "confidence": s.confidence,
                }
                for s in result.snapshots
            ],
        }
    }


# ─────────────────────────────────────────────
# Entry Point (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Face Analysis Model — DeepFace")
    print("  Default camera will open automatically.")
    print("  Press ENTER (terminal) or Q (window) to stop.")
    print("=" * 60)

    analyzer = FaceAnalyzer(
        session_id="session_001",
        snapshot_interval=2.0,
        detector_backend="opencv",
        save_face_crops=True,
        frame_skip=2,
    )

    result = analyzer.run()

    print("\n" + "=" * 60)
    print("  SESSION RESULTS")
    print("=" * 60)
    print(f"  Average Face Score    : {result.average_face_score}")
    print(f"  Dominant Emotion      : {result.dominant_emotion_overall.upper()}")
    print(f"  Session Duration      : {result.session_duration_seconds}s")
    print(f"  Face Detected Ratio   : {result.face_detected_ratio:.1%}")
    print(f"  Snapshots Captured    : {len(result.snapshots)}")

    if result.emotion_distribution:
        print("\n  Emotion Distribution:")
        for emo, prob in sorted(result.emotion_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {emo:<12} {bar:<30} {prob:.3f}")
    else:
        print("\n  No faces detected during session.")

    print("\n  Fusion Payload:")
    payload = extract_fusion_payload(result)
    print(json.dumps(
        {k: v for k, v in payload["face"].items() if k != "timeline"},
        indent=4
    ))
    print("=" * 60)