#!/usr/bin/env python3
import queue
import time
import unicodedata
from pathlib import Path
from collections import deque
import threading
import numpy as np
import whisper
import webrtcvad
import noisereduce as nr
from rapidfuzz import fuzz

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray


# ---------- CONFIG ----------
class Settings:
    MODEL_SIZE = "small"         # "tiny" for speed, "small" for accuracy
    LANGUAGE = "de"              # this is the German detector
    SAMPLE_RATE = 16000          # Whisper expects 16kHz
    VAD_AGGRESSIVENESS = 2       # 0 = sensitive, 3 = strict
    VAD_FRAME_MS = 30
    VAD_MIN_SILENCE_MS = 700
    VAD_MIN_SPEECH_MS = 250
    WORDS_FILE = "words_de.txt"  # list of German keywords
    FUZZY_THRESHOLD = 85
    ALERT_COOLDOWN_SEC = 5.0
    PRINT_TRANSCRIPTS = True
    ROS_TOPIC_NAME = "/detected_keywords_de"


# ---------- ROS NODE ----------
class KeywordPublisher(Node):
    def __init__(self, topic_name):
        super().__init__('keyword_detector_de')
        self.publisher = self.create_publisher(String, topic_name, 10)
        # Subscribe to shared audio topic
        self.sub = self.create_subscription(
            Int16MultiArray, '/audio_stream', self.audio_callback_ros, 10)
        self.get_logger().info("German Whisper detector initialized")

        # Internal buffers
        self.audio_queue = queue.Queue()
        self.speech_buffer = deque()
        self.is_speaking = False
        self.silent_frames_after_speech = 0
        self.last_alert_time = 0.0

        # Load models and keywords
        self.settings = Settings()
        self.whisper_model = whisper.load_model(self.settings.MODEL_SIZE)
        self.vad = webrtcvad.Vad(self.settings.VAD_AGGRESSIVENESS)
        self.keywords = self._load_words()
        self.get_logger().info(f"Loaded {len(self.keywords)} keywords")

        # Start background processing
        threading.Thread(target=self.process_loop, daemon=True).start()

    # ---------- Load + Normalize ----------
    def _load_words(self):
        path = Path(self.settings.WORDS_FILE)
        if not path.exists():
            self.get_logger().warn(f" Words file not found: {path}")
            return []
        return [
            self._normalize_text(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _normalize_text(self, text):
        return "".join(
            c for c in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(c)
        )

    # ---------- Audio Input via ROS ----------
    def audio_callback_ros(self, msg):
        """Receive Int16 audio from /audio_stream and push to processing queue."""
        audio = np.array(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio)

    # ---------- Process Loop ----------
    def process_loop(self):
        frame_len = int(self.settings.SAMPLE_RATE * self.settings.VAD_FRAME_MS / 1000)
        self.get_logger().info(" Listening to /audio_stream for German speech...")
        while rclpy.ok():
            try:
                data = self.audio_queue.get(timeout=0.1)
                pcm = (data * 32768).astype(np.int16)
                is_speech = self.vad.is_speech(pcm.tobytes(), self.settings.SAMPLE_RATE)

                if self.is_speaking:
                    self.speech_buffer.append(data)
                    if not is_speech:
                        self.silent_frames_after_speech += 1
                        silence_ms = self.silent_frames_after_speech * self.settings.VAD_FRAME_MS
                        if silence_ms >= self.settings.VAD_MIN_SILENCE_MS:
                            self.is_speaking = False
                            threading.Thread(
                                target=self._process_speech_buffer, daemon=True
                            ).start()
                    else:
                        self.silent_frames_after_speech = 0
                elif is_speech:
                    self.is_speaking = True
                    self.silent_frames_after_speech = 0
                    self.speech_buffer.clear()
                    self.speech_buffer.append(data)
            except queue.Empty:
                continue

    # ---------- Speech Processing ----------
    def _process_speech_buffer(self):
        if not self.speech_buffer:
            return

        speech_segment = np.concatenate(list(self.speech_buffer))
        self.speech_buffer.clear()

        # Denoise
        reduced = nr.reduce_noise(
            y=speech_segment, sr=self.settings.SAMPLE_RATE, stationary=True
        )

        # Transcribe
        result = self.whisper_model.transcribe(
            reduced, language="de", fp16=False, temperature=0.0
        )

        # Only continue if Whisper really thinks this is German
       # detected_lang = result.get("language", "")
        #if detected_lang != "de":
           #self.get_logger().info(f" Ignored non-German speech (detected {detected_lang})")
           #return
        text_raw = result.get("text", "").strip()
        if not text_raw:
            return

        if self.settings.PRINT_TRANSCRIPTS:
            self.get_logger().info(f"🎙️ Heard (DE): {text_raw}")

        text_norm = self._normalize_text(text_raw)
        detected = set()
        for kw in self.keywords:
            if fuzz.partial_ratio(kw, text_norm) >= self.settings.FUZZY_THRESHOLD:
                detected.add(kw)

        now = time.time()
        if detected and (now - self.last_alert_time) >= self.settings.ALERT_COOLDOWN_SEC:
            self.last_alert_time = now
            detected_str = ", ".join(sorted(detected))
            self.get_logger().warn(f" Detected (DE): {detected_str}")
            msg = String()
            msg.data = detected_str
            self.publisher.publish(msg)


# ---------- MAIN ----------
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
