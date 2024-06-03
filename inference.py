#!/usr/bin/env python3
import cv2
import onnxruntime as ort
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from numpy import asarray
import math
import struct
from ultralytics import RTDETR
import json
import re
from openai import OpenAI


class RosDetrNode(Node):
    def __init__(self):
        super().__init__("ros_detr_node")
        self.robot_command_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # subscribe to the rgb camera message from Astra camera
        self.img_subscripber = self.create_subscription(
            Image,
            '/Tiago_Lite/Astra_rgb/image_color',
            self.img_callback,
            10)

        # subscribe to the pointcloud message from Astra camera
        self.depth_subscriber = self.create_subscription(
            PointCloud2,
            "/Tiago_Lite/Astra_depth/point_cloud",
            self.depth_callback,
            10
        )

        self.odometry_subscribe = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
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
        self.model = RTDETR('rtdetr-x.pt')
        self.get_logger().info("============DETR Model Ready===========")
        self.bridge = CvBridge()

        self.counter = 0

        self.data = []

        self.depth = PointCloud2()

        self.odom = Odometry()

        # prevent variable not used warning
        self.img_subscripber
        self.depth_subscriber
        self.task_msg = None
        self.prompt_msg = None

        self.client = OpenAI()

        # variable reserved for storing robot memory
        self.memory = {}

        # variable reserved for storing robot state information
        self.state = []

        # previous position of the robot
        self.previous_position = []

        # variable for storing current image frame
        self.image = None

        self.task = ""

        self.is_planning = False

        self.func_str = ""

        # variable for switching on and off socratic improvement
        self.socratic = True

        self.init_task()

    def init_task(self):
        self.task = input("Enter the task for the robot: ")

        # construct the system prompt for chain-of-thought query
        system_prompt = "You are an expert in robotics navigation, perception and planning."

        # read python script from file
        script_file = open("function.txt", "r")
        script_prompt = script_file.read()
        script_file.close()

        # read chain-of-thought prompt from file
        cot_prompt_file = open("cot.txt", "r")
        cot_prompt = cot_prompt_file.read()
        cot_prompt_file.close()

        # read user prompt from file
        user_prompt_file = open("user.txt", "r")
        user_prompt = user_prompt_file.read()
        user_prompt_file.close()

        # construct user prompt for query
        user_query = cot_prompt + "\n You are tasked with {%s}, please break the task into subtasks, please make each subtask \
            doable and easy to implement. Please only output the subtasks and nothing else" % self.task


        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": system_prompt
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": user_query
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
        cot_response = response.choices[0].message.content

        print("Chain of thought substasks: \n" + cot_response)

        # apply the socratic improvement
        if self.socratic:
            cot_response = self.socracot(self.task, system_prompt, cot_response)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "Here is a breakdown of the task: \n" + cot_response + "\n Here is the script for controlling the robot: \n %s" % script_prompt
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": user_prompt
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

        func_str = response.choices[0].message.content

        print(func_str)

        self.func_str = func_str[func_str.find("def"):-3]

        if self.socratic:
            self.func_str = self.socracode(script_prompt)


    def socracot(self, task, system_prompt, llm_answer):

        query_messages=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Here is the problem statement: {%s} \
                        and here is the LLM response for breakdown of the task: {%s} \
                        Please think adversarial cases in which the above list of subtasks can fail"
                            % (task, llm_answer)
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=query_messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        adversarial_response = response.choices[0].message.content

        # add the adversarial cases suggested by LLM to the query message
        query_messages.append(
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": adversarial_response
                    }
                ]
            }
        )

        # add the user prompt to the query message
        query_messages.append(
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "based on the above analysis, give a modified list of subtasks to achieve the objective that addresses these corner cases. \
                        You should make each subtask doable and straightforward. Please only output the list of subtasks."
                    }
                ]
            }
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=query_messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        cot_response = response.choices[0].message.content

        print("modified chain of thought: \n" + cot_response)
        return cot_response

    def socracode(self, script_prompt):
        query_messages=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": script_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Here is the task you need to solve: {%s} \
                        and here is the LLM generated code for completing the task: {%s} \
                        Please think about what could go wrong with the code and fix it if that is the case. \
                        Please fix any syntax error and do not assume the code just works for granted. \
                        Please only give the modified code and nothing else, please do not explain it after."
                            % (self.task, self.func_str)
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=query_messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        func_str = response.choices[0].message.content

        func_response = func_str[func_str.find("def"):-3]
        print("modified chain of thought: \n" + func_response)
        return func_response

    # update observation: objects detected and their positions relative to the robot
    def img_callback(self, Image):
        self.counter += 1

        # call LLM every 20 images
        if (not self.is_planning) and (self.counter % 20 == 0) and (self.depth is not None):
            # start planning
            self.is_planning = True
            cv_image = self.bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)

            self.image = image_rgb
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
            if self.func_str:
                self.execute_robot_code()
            # clear detected objects
            self.data = []
            # set the planning flag to false
            self.is_planning = False

    # update depth map for use by img_callback
    def depth_callback(self, Image):
        self.depth = Image

    # update odometry information for use by the robot
    def odom_callback(self, odom):
        self.odom = odom

    # Function to execute the provided code string
    def execute_robot_code(self):
        local_vars = {}
        exec(self.func_str, globals(), local_vars)
        local_vars["move_robot"](self)


def main(args=None):
    rclpy.init(args=args)
    RosDETRNode = RosDetrNode()
    rclpy.spin(RosDETRNode)
    RosDETRNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


def move_robot(self):
        # Import necessary libraries within the function
        import math
        import numpy as np
        from geometry_msgs.msg import Twist

        # Helper to determine if the robot should rotate
        def should_rotate(memory, current_position):
            if "last_rotate_pos" in memory:
                last_pos = memory["last_rotate_pos"]
                dist = math.sqrt((current_position[0] - last_pos[0])**2 + (current_position[1] - last_pos[1])**2)
                return dist > 1.0  # Rotate every 1 unit distance
            return True

        # Extract orientation from odometry and convert quaternion to euler
        def quaternion_to_euler(x, y, z, w):
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(t0, t1)
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch = math.asin(t2)
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(t3, t4)
            return roll, pitch, yaw

        # Process odometry
        self.state = [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y]
        orientation_q = self.odom.pose.pose.orientation
        roll, pitch, current_yaw = quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)

        # Decide if rotation is required
        if should_rotate(self.memory, self.state):
            target_yaw = current_yaw + math.pi  # 180 degrees turn
            if target_yaw > math.pi:
                target_yaw -= 2 * math.pi
            self.memory["last_rotate_pos"] = self.state.copy()
            self.memory["target_yaw"] = target_yaw
            self.memory["is_rotating"] = True
        
        if self.memory.get("is_rotating", False):
            error = self.memory["target_yaw"] - current_yaw
            if abs(error) < 0.1:  # Threshold to consider rotation as complete
                self.memory["is_rotating"] = False
            else:
                twist = Twist()
                twist.angular.z = 0.5 * error
                self.robot_command_publisher.publish(twist)
                return

        # If not rotating, move linearly forward
        twist = Twist()
        twist.linear.x = 0.5  # Move forward
        self.robot_command_publisher.publish(twist)

        # Store current state and memory in previous position (for next cycle)
        if len(self.previous_position) == 0:
            self.previous_position = self.state.copy()
        else:
            delta_x = self.state[0] - self.previous_position[0]
            delta_y = self.state[1] - self.previous_position[1]
            delta_dist = math.sqrt(delta_x**2 + delta_y**2)
            if delta_dist > 0.2:  # Threshold for state update
                self.previous_position = self.state.copy()
