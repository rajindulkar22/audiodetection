#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
BLOCK_SIZE = 480      # 30 ms = ideal for VAD
DUPLICATE_WINDOW = 3  # seconds


class AudioHubNode(Node):
    def __init__(self):
        super().__init__('audio_hub_node')

        # ---- Publishers ----
        self.pub_audio = self.create_publisher(Int16MultiArray, '/audio_stream', 10)
        self.pub_alerts = self.create_publisher(String, '/detected_keywords', 10)

        # ---- Subscribers ----
        self.sub_en = self.create_subscription(String, '/detected_keywords_en', self.callback_en, 10)
        self.sub_de = self.create_subscription(String, '/detected_keywords_de', self.callback_de, 10)

        # ---- Duplicate filter ----
        self.last_msg = ""
        self.last_time = 0.0

        # ---- Start mic ----
        self.stream = sd.InputStream( 
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=BLOCK_SIZE,
            callback=self.audio_callback,
            device=5
        )
        self.stream.start()

        self.get_logger().info(" Audio hub started (publishes /audio_stream, merges alerts)")

    # ---- AUDIO FEED ----
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f"Audio status: {status}")
        msg = Int16MultiArray()
        msg.data = indata.flatten().tolist()
        self.pub_audio.publish(msg)

    # ---- DUPLICATE-AWARE MERGE ----
    def _publish_unified(self, tag, text):
        now = time.time()
        if text == self.last_msg and (now - self.last_time) < DUPLICATE_WINDOW:
            return  # skip duplicate
        self.last_msg = text
        self.last_time = now

        msg = String()
        msg.data = f"[{tag}] {text}"
        self.pub_alerts.publish(msg)
        self.get_logger().info(f"Unified alert: [{tag}] {text}")

    def callback_en(self, msg):
        self._publish_unified("EN", msg.data)

    def callback_de(self, msg):
        self._publish_unified("DE", msg.data)


def main():
    rclpy.init()
    node = AudioHubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
