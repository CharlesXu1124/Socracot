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

        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']

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
        self.task_msg = None
        self.prompt_msg = None

        self.client = OpenAI()

        self.task = "moving to the beer bottle on the table"

        self.is_planning = False
        
        self.init_task()
        
    def init_task(self):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "#!/usr/bin/env python3\nimport cv2\nimport onnxruntime as ort\nimport rclpy\nfrom rclpy.node import Node\nimport numpy as np\nfrom sensor_msgs.msg import Image\nfrom sensor_msgs.msg import PointCloud2\nfrom std_msgs.msg import Float32MultiArray\nfrom std_msgs.msg import Header\nfrom geometry_msgs.msg import Twist\nfrom cv_bridge import CvBridge\n\nfrom numpy import asarray\nimport math\nimport struct\nfrom ultralytics import RTDETR\nimport json\nfrom openai import OpenAI\n\n\nclass RosDetrNode(Node):\n    def __init__(self):\n        super().__init__(\"ros_detr_node\")\n        self.robot_command_publisher = self.create_publisher(Twist, '/cmd_vel', 10)\n        self.img_subscripber = self.create_subscription(\n            Image,\n            '/Tiago_Lite/Astra_rgb/image_color',\n            self.img_callback,\n            10)\n\n        self.depth_subscriber = self.create_subscription(\n            PointCloud2,\n            \"/Tiago_Lite/Astra_depth/point_cloud\",\n            self.depth_callback,\n            10\n        )\n        \n        self.coco_classes = [\n            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',\n            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',\n            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',\n            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',\n            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',\n            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',\n            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',\n            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',\n            'hair drier', 'toothbrush']\n\n        # Initialize DETR object detector\n        self.model = RTDETR('rtdetr-l.pt')\n        self.get_logger().info(\"============DETR Model Ready===========\")\n        self.bridge = CvBridge()\n\n        self.counter = 0\n\n        self.data = []\n\n        self.depth = PointCloud2()\n\n        # prevent variable not used warning\n        self.img_subscripber\n        self.depth_subscriber\n        self.task_msg = None\n        self.prompt_msg = None\n\n        self.client = OpenAI()\n\n        self.task = \"moving to the beer bottle on the table\"\n\n        self.is_planning = False\n\n    # update observation: objects detected and their positions relative to the robot\n    def img_callback(self, Image):\n        self.counter += 1\n\n        # call LLM every 20 images\n        if (not self.is_planning) and (self.counter % 20 == 0) and (self.depth is not None):\n            # start planning\n            self.is_planning = True\n            cv_image = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')\n            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)\n            results = self.model(image_rgb, save=False)[0]\n\n            boxes = results.boxes.data.tolist()\n\n            if len(self.depth.data) == 0:\n                return\n\n            for obj in boxes:\n                left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])\n                confidence = obj[4]\n                label = int(obj[5])\n\n                center_x = int((left + right) / 2)\n                center_y = int((top + bottom) / 2)\n\n                offset = center_y * 640 + center_x\n\n                x = struct.unpack(\"f\", bytes([self.depth.data[offset * 12],\n                    self.depth.data[offset * 12 + 1],\n                    self.depth.data[offset * 12 + 2],\n                    self.depth.data[offset * 12 + 3]\n                ]))\n\n                y = struct.unpack(\"f\", bytes([self.depth.data[offset * 12 + 4],\n                    self.depth.data[offset * 12 + 5],\n                    self.depth.data[offset * 12 + 6],\n                    self.depth.data[offset * 12 + 7]\n                ]))\n\n                z = struct.unpack(\"f\", bytes([self.depth.data[offset * 12 + 8],\n                    self.depth.data[offset * 12 + 9],\n                    self.depth.data[offset * 12 + 10],\n                    self.depth.data[offset * 12 + 11]\n                ]))\n\n                if math.isnan(x[0]) or math.isnan(y[0]) or math.isnan(z[0]):\n                    print(\"invalid distance\")\n                    continue\n                \n                detected_objects = {\n                    \"detected object\": self.coco_classes[label],\n                    \"object position\": {\n                        \"x\": x,\n                        \"y\": y,\n                        \"z\": z\n                    },\n                    \"confidence\": confidence,\n                }\n\n                self.data.append(detected_objects)\n\n            # move every 20 frames\n            self.move_robot()\n            # clear detected objects\n            self.data = []\n            # set the planning flag to false\n            self.is_planning = False\n\n    # update depth map for use by img_callback\n    def depth_callback(self, Image):\n        self.depth = Image\n\n    # move the robot to complete task\n    def move_robot(self):\n        # TODO calculate the linear speed in x, y, z direction\n        # TODO calculate the angular speed around x, y, and z axis\n\ndef main(args=None):\n    rclpy.init(args=args)\n    RosDETRNode = RosDetrNode()\n    rclpy.spin(RosDETRNode)\n    RosDETRNode.destroy_node()\n    rclpy.shutdown()\n\nif __name__ == '__main__':\n    main()\n    \n"
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are an expert in robotics localization and navigation. You are a Tiago robot in simulation environment, you are equipped with Astra depth camera which can give you information about detected objects and their positions relative to you, you also have two wheels on your chassis and you can rotate in place or move linearly. You are tasked with moving to 1m range of the beer bottle on the table, please complete the move_robot function which sends command to the robot to complete the task. Please only give the code for the python function and do not output anything else, please also specify the linear and angular velocities around 3 axis in the code. Please only output the completed code for the python function move_robot, please also make sure to output linear and angular velocities in 3 axes in the code. You should handle cases in which the object is not detected at first, and you should actively make the robots explore the environment if the object is not detected at first. Please make the function robust and reliably lead to the goal. "
                    }
                ]
                },
                {
                "role": "assistant",
                "content": [
                    {
                    "type": "text",
                    "text": "```python\ndef move_robot(self):\n    # First, check if the beer bottle is detected in the current data\n    target_object = None\n    for detected in self.data:\n        if detected[\"detected object\"] == \"bottle\":\n            target_object = detected\n            break\n    \n    # If the beer bottle is not detected, make the robot rotate to explore\n    if target_object is None:\n        cmd_msg = Twist()\n        # Set a small angular velocity to rotate in place\n        cmd_msg.angular.z = 0.2  # Angular velocity around the z-axis\n        self.robot_command_publisher.publish(cmd_msg)\n        self.get_logger().info(\"Searching for the beer bottle...\")\n        return\n    \n    # If the beer bottle is detected, calculate the relative position and move towards it\n    target_x = target_object[\"object position\"][\"x\"][0]\n    target_y = target_object[\"object position\"][\"y\"][0]\n    target_z = target_object[\"object position\"][\"z\"][0]\n    \n    self.get_logger().info(f\"Beer bottle detected at (x: {target_x}, y: {target_y}, z: {target_z})\")\n    \n    # Compute distance to the target\n    distance = math.sqrt(target_x**2 + target_y**2 + target_z**2)\n    \n    # If the robot is within 1 meter of the target\n    if distance <= 1.0:\n        cmd_msg = Twist()\n        self.get_logger().info(\"Target is within 1 meter range, stopping.\")\n        # Stop the robot\n        cmd_msg.linear.x = 0.0\n        cmd_msg.linear.y = 0.0\n        cmd_msg.linear.z = 0.0\n        cmd_msg.angular.x = 0.0\n        cmd_msg.angular.y = 0.0\n        cmd_msg.angular.z = 0.0\n        self.robot_command_publisher.publish(cmd_msg)\n        return\n    \n    # If the robot is not within 1 meter, move towards the target\n    cmd_msg = Twist()\n    \n    # Control linear velocities\n    # Move in the x direction\n    cmd_msg.linear.x = 0.1 * target_x  # Tune the gain as needed\n    # Move in the y direction\n    cmd_msg.linear.y = 0.1 * target_y  # Tune the gain as needed\n    # No movement in z direction for ground robot\n    cmd_msg.linear.z = 0.0\n    \n    # Control angular velocities\n    # No rotation around x and y axes for ground robot\n    cmd_msg.angular.x = 0.0\n    cmd_msg.angular.y = 0.0\n    # Rotate around z axis to face the target\n    cmd_msg.angular.z = -0.1 * math.atan2(target_y, target_x)  # Tune the gain as needed\n    \n    self.robot_command_publisher.publish(cmd_msg)\n```"
                    }
                ]
                }
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response.choices[0].message.content)

    # update observation: objects detected and their positions relative to the robot
    def img_callback(self, Image):
        self.counter += 1

        # call LLM every 20 images
        if (not self.is_planning) and (self.counter % 20 == 0) and (self.depth is not None):
            # start planning
            self.is_planning = True
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
                    continue

                detected_objects = {
                    "detected object": self.coco_classes[label],
                    "object position": {
                        "x": x,
                        "y": y,
                        "z": z
                    },
                    "confidence": confidence,
                }

                self.data.append(detected_objects)

            # move every 20 frames
            self.move_robot()
            # clear detected objects
            self.data = []
            # set the planning flag to false
            self.is_planning = False

    # update depth map for use by img_callback
    def depth_callback(self, Image):
        self.depth = Image

    # move the robot to complete task
    def move_robot(self):
        # TODO calculate the linear speed in x, y, z direction
        # TODO calculate the angular speed around x, y, and z axis

        # First, check if the beer bottle is detected in the current data
        target_object = None
        for detected in self.data:
            if detected["detected object"] == "bottle":
                target_object = detected
                break

        # If the beer bottle is not detected, make the robot rotate to explore
        if target_object is None:
            cmd_msg = Twist()
            # Set a small angular velocity to rotate in place
            cmd_msg.angular.z = 0.2  # Angular velocity around the z-axis
            self.robot_command_publisher.publish(cmd_msg)
            self.get_logger().info("Searching for the beer bottle...")
            return

        # If the beer bottle is detected, calculate the relative position and move towards it
        target_x = target_object["object position"]["x"][0]
        target_y = target_object["object position"]["y"][0]
        target_z = target_object["object position"]["z"][0]

        self.get_logger().info(f"Beer bottle detected at (x: {target_x}, y: {target_y}, z: {target_z})")

        # Compute distance to the target
        distance = math.sqrt(target_x**2 + target_y**2 + target_z**2)

        # If the robot is within 1 meter of the target
        if distance <= 1.0:
            cmd_msg = Twist()
            self.get_logger().info("Target is within 1 meter range, stopping.")
            # Stop the robot
            cmd_msg.linear.x = 0.0
            cmd_msg.linear.y = 0.0
            cmd_msg.linear.z = 0.0
            cmd_msg.angular.x = 0.0
            cmd_msg.angular.y = 0.0
            cmd_msg.angular.z = 0.0
            self.robot_command_publisher.publish(cmd_msg)
            return

        # If the robot is not within 1 meter, move towards the target
        cmd_msg = Twist()

        # Control linear velocities
        # Move in the x direction
        cmd_msg.linear.x = 0.1 * target_x  # Tune the gain as needed
        # Move in the y direction
        cmd_msg.linear.y = 0.1 * target_y  # Tune the gain as needed
        # No movement in z direction for ground robot
        cmd_msg.linear.z = 0.0

        # Control angular velocities
        # No rotation around x and y axes for ground robot
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        # Rotate around z axis to face the target
        cmd_msg.angular.z = -0.1 * math.atan2(target_y, target_x)  # Tune the gain as needed

        self.robot_command_publisher.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    RosDETRNode = RosDetrNode()
    rclpy.spin(RosDETRNode)
    RosDETRNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
