#!/usr/bin/env python3
import queue, time, unicodedata, threading, re
from pathlib import Path
from collections import deque
import numpy as np
import whisper
import webrtcvad
import noisereduce as nr
from rapidfuzz import fuzz
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray

# New imports for YAMNet
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd 
from typing import List

# Suppress TensorFlow warnings for cleaner ROS output
tf.get_logger().setLevel('ERROR')


class Settings:
    MODEL_SIZE = "small"
    LANGUAGE = "en" # Added for clarity
    SAMPLE_RATE = 16000
    VAD_AGGRESSIVENESS = 1
    VAD_FRAME_MS = 30
    VAD_MIN_SILENCE_MS = 700
    WORDS_FILE = "words_en.txt"
    FUZZY_THRESHOLD = 75
    ALERT_COOLDOWN_SEC = 5.0
    REPEAT_THRESHOLD = 3
    REPEAT_WINDOW_SEC = 5
    PRINT_TRANSCRIPTS = True
    ROS_TOPIC_NAME = "/detected_keywords_en"
    
    # YAMNet Settings
    YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
    YAMNET_SCORE_THRESHOLD = 0.3 # Lowered to 0.3 for higher sensitivity to distress
    
    # AudioSet classes for distress/panic
    DISTRESS_SOUNDS = [
        "Screaming", "Crying, sobbing", "Gasp", "Wail, moan", "Shout", 
        "Cough", "Throat clearing", "Anxiety", "Fear", "Yell"
    ]


def keyword_in_text(keyword, text):
    pattern = rf'\b{re.escape(keyword)}\b'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


class KeywordRepeater:
    # (Existing KeywordRepeater class remains unchanged)
    def __init__(self, repetition_threshold, time_window_sec):
        self.repetition_threshold = repetition_threshold
        self.time_window_sec = time_window_sec
        self.detected_times = {}

    def add(self, keyword, count=1):
        now = time.time()
        if keyword not in self.detected_times:
            self.detected_times[keyword] = deque()
        dq = self.detected_times[keyword]
        for _ in range(count):
            dq.append(now)
        while dq and now - dq[0] > self.time_window_sec:
            dq.popleft()

    def is_repeated(self, keyword):
        return len(self.detected_times.get(keyword, [])) >= self.repetition_threshold


def run_yamnet_distress_detection(audio_data: np.ndarray, model, class_names: List[str], score_threshold: float) -> List[str]:
    """
    Runs YAMNet on the audio segment, forced to CPU for stability, and uses MAX aggregation.
    """
    # CRITICAL FIX: Explicitly run YAMNet inference on the CPU to avoid JIT compilation errors.
    with tf.device('/cpu:0'):
        # YAMNet inference: scores are returned per frame (0.48s patch)
        scores, _, _ = model(audio_data)
        
        # Aggregation change: Use MAX instead of MEAN to detect short, peak distress sounds (like a gasp or yell)
        clip_scores = tf.reduce_max(scores, axis=0)
    
    detected_events = []
    
    # Convert clip_scores to a numpy array for efficient filtering
    clip_scores_np = clip_scores.numpy()
    
    # Find indices where the score meets the threshold
    high_score_indices = np.where(clip_scores_np >= score_threshold)[0]
    
    for class_index in high_score_indices:
        display_name = class_names[class_index]
        # Check against our predefined list of distress/panic sounds
        if display_name in Settings.DISTRESS_SOUNDS:
            detected_events.append(display_name)
            
    return sorted(list(set(detected_events)))


class KeywordPublisher(Node):
    def __init__(self, topic_name):
        super().__init__('keyword_detector_en')
        self.publisher = self.create_publisher(String, topic_name, 10)
        self.sub = self.create_subscription(Int16MultiArray, '/audio_stream', self.audio_callback_ros, 10)

        self.audio_queue = queue.Queue()
        self.speech_buffer = deque()
        self.is_speaking = False
        self.silent_frames_after_speech = 0
        self.last_detected_times = {}

        self.settings = Settings()
        self.vad = webrtcvad.Vad(self.settings.VAD_AGGRESSIVENESS)
        
        # Whisper is correctly set to device="cpu"
        self.whisper_model = whisper.load_model(self.settings.MODEL_SIZE, device="cpu")
        self.keywords = self._load_keywords()
        self.keyword_repeater = KeywordRepeater(self.settings.REPEAT_THRESHOLD, self.settings.REPEAT_WINDOW_SEC)
        
        # Initialize YAMNet, forced to CPU
        with tf.device('/cpu:0'):
            self.yamnet_model = hub.load(self.settings.YAMNET_MODEL_HANDLE)
            self.yamnet_class_names = self._load_yamnet_class_map(self.yamnet_model)

        self.get_logger().info(f"Loaded {len(self.keywords)} English keywords and YAMNet.")
        threading.Thread(target=self.process_loop, daemon=True).start()

    def _normalize_text(self, text):
        return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))

    def _load_keywords(self):
        path = Path(self.settings.WORDS_FILE)
        if not path.exists():
            self.get_logger().warn(f"Words file not found: {path}")
            return []
        return [self._normalize_text(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_yamnet_class_map(self, yamnet_model):
        """Loads the AudioSet class names from the YAMNet model path, ensuring CPU execution."""
        with tf.device('/cpu:0'):
            class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        return pd.read_csv(class_map_path)['display_name'].tolist()

    def audio_callback_ros(self, msg):
        # Convert Int16 data to float32 for model processing
        audio = np.array(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio)

    def process_loop(self):
        self.get_logger().info("Listening to /audio_stream for English speech...")
        while rclpy.ok():
            try:
                data = self.audio_queue.get(timeout=0.1)
                # Convert float32 back to int16 for VAD (which requires raw 16-bit PCM)
                pcm = (data * 32768).astype(np.int16)
                is_speech = self.vad.is_speech(pcm.tobytes(), self.settings.SAMPLE_RATE)

                # --- VAD Logic (Same as original) ---
                if self.is_speaking:
                    self.speech_buffer.append(data)
                    if not is_speech:
                        self.silent_frames_after_speech += 1
                        if self.silent_frames_after_speech * self.settings.VAD_FRAME_MS >= self.settings.VAD_MIN_SILENCE_MS:
                            self.is_speaking = False
                            threading.Thread(target=self._process_speech_buffer, daemon=True).start()
                    else:
                        self.silent_frames_after_speech = 0
                elif is_speech:
                    self.is_speaking = True
                    self.silent_frames_after_speech = 0
                    self.speech_buffer.clear()
                    self.speech_buffer.append(data)
            except queue.Empty:
                continue

    def _process_speech_buffer(self):
        if not self.speech_buffer:
            return

        speech_segment = np.concatenate(list(self.speech_buffer))
        self.speech_buffer.clear()

        if len(speech_segment) < self.settings.SAMPLE_RATE // 5 or np.all(np.abs(speech_segment) < 1e-4):
            return 

        reduced = nr.reduce_noise(y=speech_segment, sr=self.settings.SAMPLE_RATE, stationary=True)
        
        # --- 1. Enhanced Non-Verbal Distress Detection (YAMNet) ---
        try:
            distress_sounds = run_yamnet_distress_detection(
                reduced, 
                self.yamnet_model, 
                self.yamnet_class_names, 
                self.settings.YAMNET_SCORE_THRESHOLD
            )
            
            if distress_sounds:
                distress_str = ", ".join(distress_sounds)
                self.get_logger().warn(f" URGENT: Non-verbal distress detected: {distress_str}!")
                msg = String()
                msg.data = f"URGENT_ACOUSTIC_DETECTED: {distress_str}"
                self.publisher.publish(msg)
                
        except Exception as e:
            # Catch YAMNet execution failure and log it, but continue to Whisper
            self.get_logger().error(f"YAMNet execution FAILED (likely GPU config issue): {e}")
            # Continue to Whisper as a fallback

        # --- 2. Keyword Detection (Whisper) ---
        result = self.whisper_model.transcribe(reduced, language=self.settings.LANGUAGE, fp16=False, temperature=0.0)

        detected_lang = result.get("language", "")
        if detected_lang != self.settings.LANGUAGE:
            # Only ignore if NO distress sounds were found.
            if not distress_sounds:
                self.get_logger().info(f"Ignored non-{self.settings.LANGUAGE.upper()} speech (detected {detected_lang})")
            return

        text_raw = result.get("text", "").strip()
        if not text_raw:
            return

        if self.settings.PRINT_TRANSCRIPTS:
            self.get_logger().info(f"🎙️ Heard ({self.settings.LANGUAGE.upper()}): {text_raw}")

        text_norm = self._normalize_text(text_raw)
        now = time.time()
        detected_keywords = set()

        for kw in self.keywords:
            score = fuzz.partial_ratio(kw, text_norm)
            
            if score >= self.settings.FUZZY_THRESHOLD and keyword_in_text(kw, text_norm):
                count = len(re.findall(rf'\b{re.escape(kw)}\b', text_norm))
                self.keyword_repeater.add(kw, count)

                if self.keyword_repeater.is_repeated(kw):
                    last_time = self.last_detected_times.get(kw, 0)
                    if now - last_time >= self.settings.ALERT_COOLDOWN_SEC:
                        detected_keywords.add(kw)
                        self.last_detected_times[kw] = now

        if detected_keywords:
            detected_str = ", ".join(sorted(detected_keywords))
            self.get_logger().warn(f"Detected REPEATED keywords ({self.settings.LANGUAGE.upper()}): {detected_str}")
            msg = String()
            msg.data = detected_str
            self.publisher.publish(msg)


def main():
    rclpy.init()
    node = KeywordPublisher(Settings.ROS_TOPIC_NAME)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
