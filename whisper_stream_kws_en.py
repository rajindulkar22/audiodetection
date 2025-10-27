#!/usr/bin/env python3
import queue, time, unicodedata, threading, re
from pathlib import Path
from collections import deque
import numpy as np
# import whisper <-- REMOVED
from faster_whisper import WhisperModel # <-- NEW IMPORT
import webrtcvad
from rapidfuzz import fuzz
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray
import torch 
from ament_index_python.packages import get_package_share_directory

class Settings:
    MODEL_SIZE = "large-v2" # You can now try a more accurate model like "base.en" or "small.en"
    LANGUAGE = "en" 
    SAMPLE_RATE = 16000
    VAD_AGGRESSIVENESS = 3
    VAD_FRAME_MS = 30
    VAD_MIN_SILENCE_MS = 1000
    WORDS_FILE = "words_en.txt"
    FUZZY_THRESHOLD = 70
    ALERT_COOLDOWN_SEC = 2.0
    REPEAT_THRESHOLD = 3
    REPEAT_WINDOW_SEC = 5
    PRINT_TRANSCRIPTS = True
    ROS_TOPIC_NAME = "/detected_keywords_en"
    
def keyword_in_text(keyword, text):
    pattern = rf'\b{re.escape(keyword)}\b'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

class KeywordRepeater:
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
        
        # --- NEW: faster-whisper GPU Initialization ---
        # "int8_float16" uses quantization to be even faster and use less VRAM
        self.whisper_model = WhisperModel(self.settings.MODEL_SIZE, device="cuda", compute_type="int8_float16")
        self.get_logger().info(f"Faster-Whisper model '{self.settings.MODEL_SIZE}' loaded onto CUDA.")
        
        self.keywords = self._load_keywords()
        self.keyword_repeater = KeywordRepeater(self.settings.REPEAT_THRESHOLD, self.settings.REPEAT_WINDOW_SEC)
        
        self.get_logger().info(f"Loaded {len(self.keywords)} English keywords.")
        threading.Thread(target=self.process_loop, daemon=True).start()

    def _normalize_text(self, text):
        return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))

    def _load_keywords(self):
        package_share_path = get_package_share_directory('distress_detector')
        path = Path(package_share_path) / self.settings.WORDS_FILE
        
        if not path.exists():
            self.get_logger().warn(f"Words file not found: {path}")
            return []
        return [self._normalize_text(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def audio_callback_ros(self, msg):
        audio = np.array(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio)

    def process_loop(self):
        self.get_logger().info("Listening to /audio_stream for English speech...")
        while rclpy.ok():
            try:
                data = self.audio_queue.get(timeout=0.1)
                pcm = (data * 32768).astype(np.int16)
                is_speech = self.vad.is_speech(pcm.tobytes(), self.settings.SAMPLE_RATE)

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

        if len(speech_segment) < self.settings.SAMPLE_RATE // 5:
            return 

        # --- NEW: faster-whisper Transcribe Call ---
        # The transcribe method returns an iterator of segments and an info object
        segments, info = self.whisper_model.transcribe(speech_segment, language=self.settings.LANGUAGE, beam_size=5)
        
        # Combine the text from all segments into a single string
        text_raw = " ".join(segment.text for segment in segments).strip()
        detected_lang = info.language

        if detected_lang != self.settings.LANGUAGE:
            self.get_logger().info(f"Ignored non-{self.settings.LANGUAGE.upper()} speech (detected {detected_lang})")
            return

        if not text_raw:
            return

        if self.settings.PRINT_TRANSCRIPTS:
            self.get_logger().info(f" Heard ({self.settings.LANGUAGE.upper()}): {text_raw}")

        text_norm = self._normalize_text(text_raw)
        now = time.time()
        detected_keywords = set()
        segment_alerted = False

        for kw in self.keywords:
            score = fuzz.partial_ratio(kw, text_norm)

            if score >= self.settings.FUZZY_THRESHOLD and keyword_in_text(kw, text_norm):
                is_urgent = False
                
                repetitions_needed = self.settings.REPEAT_THRESHOLD - 1
                urgent_pattern = rf'\b{re.escape(kw)}(?:[,!\s]+\b{re.escape(kw)}\b){{{repetitions_needed},}}'
                
                if re.search(urgent_pattern, text_norm, flags=re.IGNORECASE):
                    is_urgent = True
                else:
                    if not segment_alerted:
                        self.keyword_repeater.add(kw, count=1)
                        segment_alerted = True
                    if self.keyword_repeater.is_repeated(kw):
                        is_urgent = True

                if is_urgent:
                    last_time = self.last_detected_times.get(kw, 0)
                    if now - last_time >= self.settings.ALERT_COOLDOWN_SEC:
                        detected_keywords.add(kw)
                        self.get_logger().warn(f"Detected URGENT keyword ({self.settings.LANGUAGE.upper()}): {kw}")
                        self.last_detected_times[kw] = now
                        break 

        if detected_keywords:
            detected_str = ", ".join(sorted(detected_keywords))
            msg = String()
            msg.data = detected_str
            self.publisher.publish(msg)

def main():
    rclpy.init()
    # Note: Assuming this script is named 'keyword_detector_node.py' in your distress_detector package
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
