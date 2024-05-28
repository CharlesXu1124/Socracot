#!/usr/bin/env python3
import cv2
import onnxruntime as ort
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from numpy import asarray
import math
import struct
from ultralytics import RTDETR
import json
from openai import OpenAI


coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

class RosDetrNode(Node):
    def __init__(self):
        super().__init__("ros_detr_node")
        self.robot_command_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_subscripber = self.create_subscription(
            Image,
            '/Tiago_Lite/Astra_rgb/image_color',
            self.img_callback,
            10)
        
        self.depth_subscriber = self.create_subscription(
            PointCloud2,
            "/Tiago_Lite/Astra_depth/point_cloud",
            self.depth_callback,
            10
        )

        # Initialize DETR object detector
        self.model = RTDETR('rtdetr-l.pt')
        self.get_logger().info("============DETR Model Ready===========")
        self.bridge = CvBridge()
        
        self.counter = 0
        
        self.data = []
        
        self.depth = PointCloud2()

        # prevent variable not used warning
        self.img_subscripber
        self.depth_subscriber

        self.client = OpenAI()



    def img_callback(self, Image):
        self.counter += 1

        if self.counter == 1:
            self.response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "system",
                    "content": [
                        {
                        "type": "text",
                        "text": "You are a Tiago robot in simulation environment, \
                            you are equipped with Astra depth camera which can give you \
                            information about detected objects and their positions relative \
                            to you, you already have the sensors activated and initialized, \
                            you are tasked with moving to the beer bottle on the table, \
                            please break down the task into smaller substasks, please give your answers \
                            wrapped in curly braces and do not output anything else"
                        }
                    ]
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            print(self.response)

            # Create a Twist message
            twist = Twist()
            # header.frame_id = 'base_link'  # Change this frame_id as needed
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.3

            self.robot_command_publisher.publish(twist)

        if self.counter < 10 and (self.counter % 10 == 0) and self.depth is not None:
            cv_image = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
            results = self.model(image_rgb, save=False)[0]

            boxes = results.boxes.data.tolist()

            if len(self.depth.data) == 0:
                return

            for obj in boxes:
                left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                confidence = obj[4]
                label = int(obj[5])

                center_x = int((left + right) / 2)
                center_y = int((top + bottom) / 2)

                offset = center_y * 640 + center_x

                x = struct.unpack("f", bytes([self.depth.data[offset * 12],
                    self.depth.data[offset * 12 + 1],
                    self.depth.data[offset * 12 + 2],
                    self.depth.data[offset * 12 + 3]
                ]))

                y = struct.unpack("f", bytes([self.depth.data[offset * 12 + 4],
                    self.depth.data[offset * 12 + 5],
                    self.depth.data[offset * 12 + 6],
                    self.depth.data[offset * 12 + 7]
                ]))

                z = struct.unpack("f", bytes([self.depth.data[offset * 12 + 8],
                    self.depth.data[offset * 12 + 9],
                    self.depth.data[offset * 12 + 10],
                    self.depth.data[offset * 12 + 11]
                ]))

                if math.isnan(x[0]) or math.isnan(y[0]) or math.isnan(z[0]):
                    print("invalid distance")
                    return

                objects = {
                    "detected object": coco_classes[label],
                    "object position": {
                        "x": x,
                        "y": y,
                        "z": z
                    },
                    "confidence": confidence,
                }

                self.data.append(objects)

                print(self.data)

            # clear the detected objects
            self.data = []

    def depth_callback(self, Image):
        self.depth = Image

    def construct_prompt(self):
        pass

def main(args=None):
    rclpy.init(args=args)
    RosDETRNode = RosDetrNode()
    rclpy.spin(RosDETRNode)
    RosDETRNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()