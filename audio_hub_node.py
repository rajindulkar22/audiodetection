#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
import sounddevice as sd
import numpy as np

# --- SETTINGS ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 480      # 30 ms = ideal for VAD (16000 * 0.030)
DUPLICATE_WINDOW = 3  # seconds
AUDIO_DEVICE_ID = 18   # Specify the device ID directly


class AudioHubNode(Node):
    def __init__(self):
        super().__init__('audio_hub_node')

        # ---- Publishers ----
        # Audio stream for VAD/ASR nodes (e.g., German and English detectors)
        self.pub_audio = self.create_publisher(Int16MultiArray, '/audio_stream', 10)
        # Unified alert topic for system-wide consumption
        self.pub_alerts = self.create_publisher(String, '/detected_keywords', 10)

        # ---- Subscribers ----
        self.sub_en = self.create_subscription(String, '/detected_keywords_en', self.callback_en, 10)
        self.sub_de = self.create_subscription(String, '/detected_keywords_de', self.callback_de, 10)

        # ---- Duplicate filter state ----
        self.last_msg_data = ""
        self.last_time = 0.0
        self.stream = None

        # ---- Start mic and stream ----
        self._start_audio_stream()

        self.get_logger().info(" Audio hub initialized. Publishing /audio_stream and merging alerts.")
        self.get_logger().info(f"Using device ID: {AUDIO_DEVICE_ID}")


    def _start_audio_stream(self):
        """Initializes and starts the sounddevice input stream with error handling."""
        try:
            self.stream = sd.InputStream( 
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback,
                device=AUDIO_DEVICE_ID # Use the defined device ID
            )
            self.stream.start()
        except sd.PortAudioError as e:
            self.get_logger().error(f"Failed to start audio stream (Device ID {AUDIO_DEVICE_ID}): {e}")
            self.get_logger().error("Please check that the microphone is connected and the device ID is correct.")
            # Set stream to None so stop() doesn't fail in finally block
            self.stream = None
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred during audio stream setup: {e}")
            self.stream = None


    # ---- AUDIO FEED CALLBACK ----
    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice when a new block of audio data is available."""
        if status:
            self.get_logger().warn(f"Audio stream status warning: {status}")
        
        msg = Int16MultiArray()
        # Flatten the indata array (it should be 1D anyway) and convert to list for the ROS message
        msg.data = indata.flatten().tolist()
        self.pub_audio.publish(msg)

    # ---- DUPLICATE-AWARE MERGE LOGIC ----
    def _publish_unified(self, tag: str, text: str):
        """
        Merges language-specific alerts into a single topic, filtering duplicates.
        """
        now = time.time()
        
        # Format the incoming alert data for comparison
        current_msg_data = f"[{tag}] {text}"

        # 1. Duplicate check: Same message content within the time window
        if current_msg_data == self.last_msg_data and (now - self.last_time) < DUPLICATE_WINDOW:
            # self.get_logger().debug(f"Skipped duplicate alert: {current_msg_data}")
            return  

        # Update filter state
        self.last_msg_data = current_msg_data
        self.last_time = now

        # Publish the unified alert
        msg = String()
        msg.data = current_msg_data
        self.pub_alerts.publish(msg)
        self.get_logger().warn(f" UNIFIED ALERT: {current_msg_data}")


    def callback_en(self, msg: String):
        """Callback for English keyword detections."""
        self._publish_unified("EN", msg.data)

    def callback_de(self, msg: String):
        """Callback for German keyword detections."""
        self._publish_unified("DE", msg.data)


def main():
    rclpy.init()
    node = AudioHubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Gracefully stop the audio stream if it was successfully started
        if node.stream:
            node.stream.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
