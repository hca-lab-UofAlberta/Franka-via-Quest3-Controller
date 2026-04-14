#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
import pyrealsense2 as rs

SENSOR_QOS = QoSPresetProfiles.SENSOR_DATA.value
ENCODING = 'yuv422_yuy2'   # ROS encoding name for YUYV
BYTES_PER_PIXEL = 2

CAM_WRIST_SERIAL = 241122306284
CAM_HEAD_SERIAL = 234322305598
CAM_SIDE_SERIAL = 241122302482

CAM_FPS = 30


class SingleCameraNode(Node):
    def __init__(self, name: str, serial: str, topic: str, frame_id: str,
                 width: int, height: int, fps: int):
        super().__init__(name)
        self.serial = serial
        self.topic = topic
        self.frame_id = frame_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pub = self.create_publisher(Image, self.topic, SENSOR_QOS)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.yuyv, self.fps)
        profile = self.pipeline.start(cfg)

        # Drop frames - publish one frame as it arrives
        try:
            dev = profile.get_device()
            color = dev.first_color_sensor()
            color.set_option(rs.option.frames_queue_size, 1)
        except Exception:
            pass

        self.timer = self.create_timer(1.0 / self.fps, self.publish_once)
        self.get_logger().info(
            f"[{self.get_name()}] serial={self.serial}, topic={self.topic}, "
            f"res={self.width}x{self.height}@{self.fps}Hz format=YUYV"
        )

    def publish_once(self):
        frames = self.pipeline.wait_for_frames()
        if not frames or frames.size() == 0:
            return
        color = frames.get_color_frame()
        if not color:
            return

        yuyv = memoryview(color.get_data())

        # ── Real-time preview ──────────────────────────────────────────
        yuyv_np = np.frombuffer(yuyv, dtype=np.uint8).reshape((self.height, self.width, 2))
        bgr = cv2.cvtColor(yuyv_np, cv2.COLOR_YUV2BGR_YUYV)
        cv2.imshow(f"Camera: {self.topic}", bgr)
        cv2.waitKey(1)
        # ──────────────────────────────────────────────────────────────

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = ENCODING
        msg.is_bigendian = 0
        msg.step = self.width * BYTES_PER_PIXEL
        msg.data = yuyv.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, help="RealSense camera name (wrist, head, side)")
    parser.add_argument("--serial", default=None, help="Override camera serial number (e.g. 241122306284)")
    parser.add_argument("--frame-id", default="camera", help="frame_id")
    parser.add_argument("--width", type=int, default=int(os.getenv('IMAGE_WIDTH', '640')))
    parser.add_argument("--height", type=int, default=int(os.getenv('IMAGE_HEIGHT', '480')))
    parser.add_argument("--fps", type=int, default=int(os.getenv('FRAME_RATE', '15')))
    parser.add_argument("--node-name", default=None)
    args = parser.parse_args()

    rclpy.init()
    node_name = args.node_name or f"cam_{args.camera}"
    if args.camera == "wrist":
        camera_serial = str(args.serial or CAM_WRIST_SERIAL)
        camera_topic = '/cam_wrist'
    elif args.camera == "head":
        camera_serial = str(args.serial or CAM_HEAD_SERIAL)
        camera_topic = '/cam_head'
    elif args.camera == "side":
        camera_serial = str(args.serial or CAM_SIDE_SERIAL)
        camera_topic = '/cam_side'
    else:
        raise ValueError(f"Unknown camera name: {args.camera}. Choose from: wrist, head, side")

    node = SingleCameraNode(node_name, camera_serial, camera_topic, args.frame_id,
                            args.width, args.height, args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node:
                node.destroy_node()
        finally:
            print("Shutting down rclpy...")
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


if __name__ == "__main__":
    main()