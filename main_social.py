#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
社交场景训练系统主程序
集成实时情绪识别、打招呼场景、情绪识别训练场景和训练报告功能
"""

import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# 设置TensorFlow环境变量，减少警告输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息，不显示警告

# 情绪映射字典
EMOTION_MAP = {
    0: '愤怒', 1: '轻蔑', 2: '厌恶', 3: '恐惧',
    4: '快乐', 5: '悲伤', 6: '惊讶'
}

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在图像上绘制中文字符（使用PIL）
    
    Args:
        img: OpenCV图像 (BGR格式)
        text: 要显示的中文文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (BGR格式)
    
    Returns:
        显示中文文本后的图像
    """
    # 将OpenCV图像转换为PIL图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试加载中文字体
    font = None
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                print(f"使用字体: {font_path}")
                break
        except:
            continue
    
    # 如果找不到中文字体，使用默认字体（显示英文）
    if font is None:
        print("警告: 未找到中文字体，使用默认字体（可能不支持中文）")
        font = ImageFont.load_default()
    
    # 绘制文本（注意颜色转换：BGR -> RGB）
    rgb_color = (color[2], color[1], color[0])  # BGR转RGB
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转换回OpenCV格式
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return img_bgr

class SocialTrainingSystem:
    """社交训练系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.cap = None
        self.running = True
        self.current_mode = 'menu'
        self.window_name = "Social Training System"  # 使用英文窗口名称避免乱码
        
        # 菜单选项
        self.menu_options = [
            "1: 实时情绪识别",
            "2: 打招呼场景", 
            "3: 情绪识别训练",
            "4: 查看训练报告",
            "q: 退出程序"
        ]
        
        # 情绪识别模型
        self.model = None
        self.model_path = './models/ckplus_emotion_model.h5'
        
        # 日志文件
        self.log_file = 'social_log.csv'
        os.makedirs('logs', exist_ok=True)
        
        # 实时识别参数
        self.last_recognition_time = 0
        self.recognition_interval = 1  # 1秒识别一次
        self.current_emotion = "未检测"
        self.current_confidence = 0.0
        
        print("社交训练系统初始化完成")
    
    def _initialize_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("错误: 摄像头不可用")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("摄像头初始化成功")
            return True
            
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def _load_emotion_model(self):
        """加载情绪识别模型"""
        try:
            if not os.path.exists(self.model_path):
                print(f"模型文件不存在: {self.model_path}")
                return False
            
            self.model = load_model(self.model_path)
            print("情绪识别模型加载成功")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def _preprocess_face(self, face_img):
        """预处理人脸图像"""
        if face_img is None:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # 缩放到48x48（模型输入大小）
        resized = cv2.resize(gray, (48, 48))
        
        # 归一化
        normalized = resized / 255.0
        
        # 添加批次和通道维度
        input_data = np.expand_dims(normalized, axis=(0, -1))
        
        return input_data
    
    def _recognize_emotion(self, frame):
        """识别帧中的情绪"""
        if self.model is None:
            return "模型未加载", 0.0
        
        # 人脸检测
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return "未检测到人脸", 0.0
        
        # 使用第一个检测到的人脸
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # 预处理
        input_data = self._preprocess_face(face_roi)
        if input_data is None:
            return "预处理失败", 0.0
        
        # 预测
        predictions = self.model.predict(input_data, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        
        # 映射到中文情绪
        emotion = EMOTION_MAP.get(emotion_idx, "未知")
        
        return emotion, confidence
    
    def _display_menu(self, frame):
        """在帧上显示菜单"""
        if frame is None:
            # 创建黑色背景作为备用显示
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            put_chinese_text(frame, "摄像头不可用", (50, 50), 20, (255, 255, 255))
        
        # 添加菜单标题
        frame = put_chinese_text(frame, "社交训练系统", (50, 30), 24, (255, 255, 0))
        
        # 添加菜单选项
        y_offset = 80
        for option in self.menu_options:
            frame = put_chinese_text(frame, option, (50, y_offset), 18, (255, 255, 255))
            y_offset += 35
        
        # 添加操作提示
        frame = put_chinese_text(frame, "请按对应数字键选择功能", (50, y_offset + 20), 16, (0, 255, 0))
        frame = put_chinese_text(frame, "在任何模式中按 0 返回主菜单", (50, y_offset + 50), 16, (0, 255, 0))
        
        return frame
    
    def _read_camera_frame(self):
        """读取摄像头帧"""
        if not self.cap or not self.cap.isOpened():
            return None, False
        
        try:
            ret, frame = self.cap.read()
            return frame, ret
            
        except Exception as e:
            print(f"摄像头读取异常: {e}")
            return None, False
    
    def run_realtime_mode(self):
        """运行实时情绪识别模式"""
        print("进入实时情绪识别模式，按0返回主菜单")
        
        while self.current_mode == 'realtime':
            frame, success = self._read_camera_frame()
            
            if not success:
                # 创建黑色背景显示错误信息
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = put_chinese_text(frame, "摄像头读取失败", (50, 200), 20, (255, 0, 0))
            else:
                # 定期进行情绪识别
                current_time = time.time()
                if current_time - self.last_recognition_time >= self.recognition_interval:
                    self.last_recognition_time = current_time
                    emotion, confidence = self._recognize_emotion(frame)
                    
                    if confidence >= 0.4:
                        self.current_emotion = emotion
                        self.current_confidence = confidence
                    else:
                        self.current_emotion = "不确定"
                        self.current_confidence = 0.0
                
                # 在帧上显示情绪信息
                frame = put_chinese_text(frame, f"情绪: {self.current_emotion}", (20, 30), 18, (0, 255, 0))
                frame = put_chinese_text(frame, f"置信度: {self.current_confidence:.2f}", (20, 60), 18, (0, 255, 0))
            
            # 显示操作提示
            frame = put_chinese_text(frame, "按0返回主菜单", (20, 90), 16, (255, 0, 0))
            
            # 显示帧
            cv2.imshow(self.window_name, frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):
                self.current_mode = 'menu'
                print("返回主菜单")
    
    def run_greeting_scene(self):
        """运行打招呼场景"""
        print("\n=== 开始打招呼场景 ===")
        print("规则：根据小明的情绪选择正确的打招呼方式")
        print("按 '0' 键可返回主菜单\n")
        
        # 情绪选项和对应答案
        emotion_options = ['快乐', '悲伤', '惊讶']
        emotion_to_answer = {
            '快乐': 1,  # 对应按键1：你好呀
            '悲伤': 2,  # 对应按键2：你怎么了
            '惊讶': 3   # 对应按键3：哇，好巧
        }
        
        # 场景配置
        total_rounds = 5
        current_round = 0
        correct_count = 0
        
        # 主循环
        while current_round < total_rounds:
            # 生成随机情绪
            current_emotion = np.random.choice(emotion_options)
            correct_answer = emotion_to_answer[current_emotion]
            
            # 创建场景界面
            frame = self._create_greeting_scene_frame(current_emotion, current_round + 1)
            cv2.imshow(self.window_name, frame)
            
            # 等待用户按键
            key = cv2.waitKey(0) & 0xFF
            
            # 检查退出
            if key == ord('0'):
                print("返回主菜单")
                break
            
            # 检查答案
            user_answer = key - ord('0')  # 将按键转换为数字
            is_correct = (user_answer == correct_answer)
            
            if is_correct:
                correct_count += 1
                feedback = "回答正确！"
                feedback_color = (0, 255, 0)  # 绿色
            else:
                feedback = f"回答错误，正确的是{correct_answer}"
                feedback_color = (0, 0, 255)  # 红色
            
            # 显示反馈
            feedback_frame = self._create_greeting_feedback_frame(feedback, feedback_color)
            cv2.imshow(self.window_name, feedback_frame)
            cv2.waitKey(1500)  # 显示反馈1.5秒
            
            current_round += 1
        
        # 记录日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'scene': 'greeting',
            'score': correct_count,
            'total': total_rounds
        }
        
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_file, mode='a', header=not os.path.exists(self.log_file), index=False)
        
        print(f"打招呼场景结束，得分: {correct_count}/{total_rounds}")
        self.current_mode = 'menu'
    
    def _create_greeting_scene_frame(self, emotion, round_number):
        """创建打招呼场景界面"""
        # 创建白色背景
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 添加标题
        frame = put_chinese_text(frame, f"第 {round_number} 轮 - 打招呼场景", 
                               (50, 30), 24, (0, 0, 0))
        
        # 添加虚拟角色描述
        frame = put_chinese_text(frame, "虚拟角色：小明", 
                               (50, 80), 18, (0, 0, 0))
        frame = put_chinese_text(frame, f"小明看起来{emotion}...", 
                               (50, 110), 18, (0, 0, 0))
        
        # 绘制简单的虚拟角色（圆形头像）
        center_x, center_y = 150, 180
        cv2.circle(frame, (center_x, center_y), 40, (200, 200, 200), -1)  # 灰色圆形
        
        # 根据情绪添加表情
        if emotion == '快乐':
            # 笑脸
            cv2.ellipse(frame, (center_x-15, center_y-5), (8, 4), 0, 0, 180, (0, 0, 0), 2)
            cv2.ellipse(frame, (center_x+15, center_y-5), (8, 4), 0, 0, 180, (0, 0, 0), 2)
            cv2.ellipse(frame, (center_x, center_y+8), (16, 8), 0, 0, 180, (0, 0, 0), 2)
        elif emotion == '悲伤':
            # 悲伤脸
            cv2.circle(frame, (center_x-15, center_y-5), 3, (0, 0, 0), -1)
            cv2.circle(frame, (center_x+15, center_y-5), 3, (0, 0, 0), -1)
            cv2.ellipse(frame, (center_x, center_y+12), (12, 6), 0, 180, 0, (0, 0, 0), 2)
        else:  # 惊讶
            # 惊讶脸
            cv2.circle(frame, (center_x-15, center_y-5), 4, (0, 0, 0), -1)
            cv2.circle(frame, (center_x+15, center_y-5), 4, (0, 0, 0), -1)
            cv2.circle(frame, (center_x, center_y+8), 6, (0, 0, 0), 2)
        
        # 添加选项说明
        frame = put_chinese_text(frame, "请按对应数字键选择打招呼方式：", 
                               (50, 250), 18, (0, 0, 0))
        frame = put_chinese_text(frame, "1. 开心时说'你好呀'", 
                               (70, 280), 16, (0, 0, 0))
        frame = put_chinese_text(frame, "2. 悲伤时说'你怎么了'", 
                               (70, 310), 16, (0, 0, 0))
        frame = put_chinese_text(frame, "3. 惊讶时说'哇，好巧'", 
                               (70, 340), 16, (0, 0, 0))
        
        # 添加操作提示
        frame = put_chinese_text(frame, "按0返回主菜单", 
                               (50, 400), 16, (255, 0, 0))
        
        return frame
    
    def _create_greeting_feedback_frame(self, feedback, color):
        """创建打招呼场景反馈界面"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        frame = put_chinese_text(frame, feedback, (50, 200), 24, color)
        frame = put_chinese_text(frame, "按任意键继续...", (50, 250), 16, (100, 100, 100))
        return frame
    
    def run_emotion_training_scene(self):
        """运行情绪识别训练场景"""
        print("\n=== 开始情绪识别训练场景 ===")
        print("规则：识别图片中人物的情绪")
        print("目标：连续答对3次或完成5题")
        print("按 '0' 键可返回主菜单\n")
        
        # 场景配置
        max_questions = 5
        consecutive_correct_needed = 3
        current_question = 0
        correct_count = 0
        consecutive_correct = 0
        
        # 情绪类别
        emotion_categories = ['愤怒', '轻蔑', '厌恶', '恐惧', '快乐', '悲伤', '惊讶']
        
        # 加载CK+测试数据
        test_data = self._load_ckplus_data()
        if not test_data:
            print("警告: 无法加载CK+测试数据，将使用模拟数据")
            test_data = None
        
        # 主循环
        while (current_question < max_questions and 
               consecutive_correct < consecutive_correct_needed):
            
            # 获取测试图片数据
            if test_data:
                # 从CK+数据中随机选择一张图片
                current_item = test_data[np.random.randint(0, len(test_data))]
                current_emotion_idx = current_item['emotion']
                current_pixels = current_item['pixels']
                current_emotion = emotion_categories[current_emotion_idx]
            else:
                # 使用模拟数据
                current_emotion = np.random.choice(emotion_categories)
                current_pixels = None
            
            # 创建训练界面
            frame = self._create_emotion_training_frame(current_emotion, current_question + 1, current_pixels)
            cv2.imshow(self.window_name, frame)
            
            # 等待用户输入
            key = cv2.waitKey(0) & 0xFF
            
            # 检查退出
            if key == ord('0'):
                print("返回主菜单")
                break
            
            # 检查答案（用户输入1-7对应情绪）
            user_answer = key - ord('0')
            correct_answer = emotion_categories.index(current_emotion) + 1
            
            is_correct = (user_answer == correct_answer)
            
            if is_correct:
                correct_count += 1
                consecutive_correct += 1
                feedback = "回答正确！"
                feedback_color = (0, 255, 0)  # 绿色
            else:
                consecutive_correct = 0
                feedback = f"回答错误，正确的是{correct_answer} ({current_emotion})"
                feedback_color = (0, 0, 255)  # 红色
            
            # 显示反馈
            feedback_frame = self._create_emotion_training_feedback_frame(feedback, feedback_color, consecutive_correct)
            cv2.imshow(self.window_name, feedback_frame)
            cv2.waitKey(1500)  # 显示反馈1.5秒
            
            current_question += 1
            
            # 检查是否达到目标
            if consecutive_correct >= consecutive_correct_needed:
                print(f"恭喜！连续答对{consecutive_correct}次，达成目标！")
                break
        
        # 记录日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'scene': 'emotion_training',
            'correct_count': correct_count,
            'total_questions': current_question,
            'consecutive_correct': consecutive_correct
        }
        
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_file, mode='a', header=not os.path.exists(self.log_file), index=False)
        
        print(f"情绪识别训练结束，正确率: {correct_count}/{current_question}")
        print(f"最高连续答对: {consecutive_correct}次")
        self.current_mode = 'menu'
    
    def _load_ckplus_data(self):
        """加载CK+数据集"""
        csv_path = './data/ck+_48/ckextended.csv'
        
        if not os.path.exists(csv_path):
            print(f"CK+数据文件不存在: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            test_data = []
            
            for index, row in df.iterrows():
                if row['Usage'] == 'Training':  # 只使用训练数据
                    emotion = int(row['emotion'])
                    pixels_str = row['pixels']
                    
                    # 解析像素字符串
                    pixels = [int(x) for x in pixels_str.split()]
                    
                    # 转换为48x48图像
                    if len(pixels) == 2304:  # 48x48=2304
                        test_data.append({
                            'emotion': emotion,
                            'pixels': pixels
                        })
            
            print(f"成功加载 {len(test_data)} 张CK+测试图片")
            return test_data
            
        except Exception as e:
            print(f"加载CK+数据失败: {e}")
            return None
    
    def _create_emotion_training_frame(self, emotion, question_number, pixels=None):
        """创建情绪识别训练界面"""
        # 情绪类别（在类级别定义，确保可访问）
        emotion_categories = ['愤怒', '轻蔑', '厌恶', '恐惧', '快乐', '悲伤', '惊讶']
        
        # 创建白色背景
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 添加标题
        frame = put_chinese_text(frame, f"第 {question_number} 题 - 情绪识别训练", 
                               (50, 30), 24, (0, 0, 0))
        
        # 添加问题描述
        if pixels:
            frame = put_chinese_text(frame, "请识别CK+数据集中人物的情绪：", 
                                   (50, 80), 18, (0, 0, 0))
        else:
            frame = put_chinese_text(frame, "请识别模拟图片中人物的情绪：", 
                                   (50, 80), 18, (0, 0, 0))
        
        # 显示图片区域（调整位置，为选项留出更多空间）
        img_x, img_y = 50, 120
        img_width, img_height = 200, 150
        
        if pixels:
            # 使用真实的CK+数据
            try:
                # 将像素数据转换为48x48图像
                pixel_array = np.array(pixels, dtype=np.uint8)
                face_img = pixel_array.reshape(48, 48)
                
                # 放大图像到合适大小
                face_img_resized = cv2.resize(face_img, (img_width, img_height))
                
                # 转换为彩色图像
                face_img_color = cv2.cvtColor(face_img_resized, cv2.COLOR_GRAY2BGR)
                
                # 将图像添加到主画面
                frame[img_y:img_y+img_height, img_x:img_x+img_width] = face_img_color
                
            except Exception as e:
                print(f"处理CK+图片失败: {e}")
                # 如果处理失败，使用模拟图片
                face_img = self._create_simulated_face_image(emotion, img_width, img_height)
                frame[img_y:img_y+img_height, img_x:img_x+img_width] = face_img
        else:
            # 使用模拟图片
            face_img = self._create_simulated_face_image(emotion, img_width, img_height)
            frame[img_y:img_y+img_height, img_x:img_x+img_width] = face_img
        
        # 添加图片边框
        cv2.rectangle(frame, (img_x-2, img_y-2), 
                     (img_x+img_width+2, img_y+img_height+2), (0, 0, 0), 2)
        
        # 添加选项说明（调整位置到图片右侧）
        frame = put_chinese_text(frame, "请按对应数字键选择情绪：", 
                               (300, 120), 18, (0, 0, 0))
        
        # 显示选项列表（两列布局，避免超出屏幕）
        y_offset = 150
        for i, emotion_name in enumerate(emotion_categories, 1):
            if i <= 4:  # 前4个选项在第一列
                frame = put_chinese_text(frame, f"{i}. {emotion_name}", 
                                       (300, y_offset), 16, (0, 0, 0))
                y_offset += 30
            else:  # 后3个选项在第二列
                frame = put_chinese_text(frame, f"{i}. {emotion_name}", 
                                       (450, y_offset - 90), 16, (0, 0, 0))
                y_offset += 30
        
        # 添加操作提示（调整位置）
        frame = put_chinese_text(frame, "按0返回主菜单", 
                               (300, 350), 16, (255, 0, 0))
        
        return frame
    
    def _create_simulated_face_image(self, emotion, width, height):
        """创建模拟表情图片"""
        # 创建表情图片区域
        face_img = np.ones((height, width, 3), dtype=np.uint8) * 200  # 浅灰色背景
        
        # 绘制人脸轮廓
        face_center = (width//2, height//2)
        face_radius = min(width, height) // 3
        cv2.circle(face_img, face_center, face_radius, (150, 150, 150), 2)
        
        # 根据情绪绘制不同的表情
        if emotion == '愤怒':
            # 愤怒：皱眉，嘴角向下
            cv2.line(face_img, (face_center[0]-20, face_center[1]-10), 
                    (face_center[0]-10, face_center[1]-15), (0, 0, 0), 2)  # 左眉
            cv2.line(face_img, (face_center[0]+10, face_center[1]-15), 
                    (face_center[0]+20, face_center[1]-10), (0, 0, 0), 2)  # 右眉
            cv2.ellipse(face_img, (face_center[0], face_center[1]+10), 
                       (30, 15), 0, 180, 360, (0, 0, 0), 2)  # 向下嘴角
        elif emotion == '快乐':
            # 快乐：微笑
            cv2.ellipse(face_img, (face_center[0], face_center[1]+10), 
                       (30, 20), 0, 0, 180, (0, 0, 0), 2)  # 微笑
        elif emotion == '悲伤':
            # 悲伤：嘴角向下，眼角下垂
            cv2.ellipse(face_img, (face_center[0], face_center[1]+10), 
                       (30, 15), 0, 180, 360, (0, 0, 0), 2)  # 向下嘴角
            cv2.line(face_img, (face_center[0]-15, face_center[1]-5), 
                    (face_center[0]-15, face_center[1]-10), (0, 0, 0), 2)  # 左眼下垂
            cv2.line(face_img, (face_center[0]+15, face_center[1]-5), 
                    (face_center[0]+15, face_center[1]-10), (0, 0, 0), 2)  # 右眼下垂
        elif emotion == '惊讶':
            # 惊讶：大眼睛，O型嘴
            cv2.circle(face_img, (face_center[0]-15, face_center[1]-5), 8, (0, 0, 0), 2)  # 左眼
            cv2.circle(face_img, (face_center[0]+15, face_center[1]-5), 8, (0, 0, 0), 2)  # 右眼
            cv2.circle(face_img, (face_center[0], face_center[1]+10), 10, (0, 0, 0), 2)  # O型嘴
        elif emotion == '恐惧':
            # 恐惧：睁大眼睛，嘴角紧张
            cv2.ellipse(face_img, (face_center[0]-15, face_center[1]-5), 
                       (6, 8), 0, 0, 360, (0, 0, 0), 2)  # 左眼
            cv2.ellipse(face_img, (face_center[0]+15, face_center[1]-5), 
                       (6, 8), 0, 0, 360, (0, 0, 0), 2)  # 右眼
            cv2.line(face_img, (face_center[0]-10, face_center[1]+10), 
                    (face_center[0]+10, face_center[1]+10), (0, 0, 0), 2)  # 紧张嘴
        elif emotion == '厌恶':
            # 厌恶：皱眉，撇嘴
            cv2.line(face_img, (face_center[0]-20, face_center[1]-10), 
                    (face_center[0]-10, face_center[1]-15), (0, 0, 0), 2)  # 左眉
            cv2.line(face_img, (face_center[0]+10, face_center[1]-15), 
                    (face_center[0]+20, face_center[1]-10), (0, 0, 0), 2)  # 右眉
            cv2.ellipse(face_img, (face_center[0], face_center[1]+10), 
                       (20, 10), 0, 180, 360, (0, 0, 0), 2)  # 撇嘴
        else:  # 轻蔑
            # 轻蔑：单边嘴角上扬
            cv2.line(face_img, (face_center[0]-10, face_center[1]+10), 
                    (face_center[0]+10, face_center[1]+5), (0, 0, 0), 2)  # 不对称嘴
        
        return face_img
    
    def _create_emotion_training_feedback_frame(self, feedback, color, consecutive_correct):
        """创建情绪识别训练反馈界面"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 显示反馈
        frame = put_chinese_text(frame, feedback, (50, 150), 24, color)
        
        # 显示连续答对次数
        if consecutive_correct > 0:
            frame = put_chinese_text(frame, f"连续答对: {consecutive_correct}次", 
                                   (50, 200), 18, (0, 128, 0))
        
        frame = put_chinese_text(frame, "按任意键继续...", (50, 250), 16, (100, 100, 100))
        
        return frame
    
    def show_training_report(self):
        """显示训练报告"""
        print("显示训练报告，按任意键返回主菜单")
        
        # 创建报告界面
        report_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if not os.path.exists(self.log_file):
            # 没有日志文件
            report_frame = put_chinese_text(report_frame, "暂无训练数据", (200, 200), 24, (255, 0, 0))
            report_frame = put_chinese_text(report_frame, "请先完成一些训练场景", (150, 240), 18, (255, 255, 0))
        else:
            # 读取并分析日志（处理格式不一致问题）
            try:
                # 尝试读取CSV文件，处理字段数量不一致的问题
                df = pd.read_csv(self.log_file, on_bad_lines='skip')
                
                if len(df) == 0:
                    report_frame = put_chinese_text(report_frame, "训练报告", (250, 30), 24, (255, 255, 0))
                    report_frame = put_chinese_text(report_frame, "日志文件为空", (250, 200), 24, (255, 0, 0))
                    report_frame = put_chinese_text(report_frame, "请先完成训练场景", (200, 240), 18, (255, 255, 0))
                else:
                    report_frame = put_chinese_text(report_frame, "训练报告", (250, 30), 24, (255, 255, 0))
                    
                    y_offset = 80
                    report_frame = put_chinese_text(report_frame, f"总训练次数: {len(df)}", (50, y_offset), 18, (255, 255, 255))
                    y_offset += 30
                    
                    # 统计各场景的训练次数
                    if 'scene' in df.columns:
                        scene_counts = df['scene'].value_counts()
                        for scene, count in scene_counts.items():
                            report_frame = put_chinese_text(report_frame, f"{scene}: {count}次", (50, y_offset), 16, (255, 255, 0))
                            y_offset += 25
                    
                    # 显示统计信息
                    if 'score' in df.columns:
                        avg_score = df['score'].mean()
                        report_frame = put_chinese_text(report_frame, f"平均得分: {avg_score:.1f}", (50, y_offset + 20), 18, (0, 255, 0))
                    elif 'correct_count' in df.columns:
                        avg_correct = df['correct_count'].mean()
                        report_frame = put_chinese_text(report_frame, f"平均正确率: {avg_correct:.1f}", (50, y_offset + 20), 18, (0, 255, 0))
                    
                    # 添加成功提示
                    report_frame = put_chinese_text(report_frame, "报告加载成功！", (50, y_offset + 50), 16, (0, 255, 0))
                
            except Exception as e:
                print(f"读取日志文件失败: {e}")
                # 显示错误信息
                report_frame = put_chinese_text(report_frame, "训练报告", (250, 30), 24, (255, 255, 0))
                report_frame = put_chinese_text(report_frame, "日志文件格式错误", (200, 200), 24, (255, 0, 0))
                report_frame = put_chinese_text(report_frame, f"错误: {str(e)[:30]}...", (200, 240), 16, (255, 255, 0))
                report_frame = put_chinese_text(report_frame, "请删除social_log.csv后重试", (180, 270), 16, (255, 255, 0))
        
        report_frame = put_chinese_text(report_frame, "按任意键返回主菜单", (200, 400), 16, (255, 0, 0))
        
        cv2.imshow(self.window_name, report_frame)
        cv2.waitKey(0)
        
        self.current_mode = 'menu'
        print("返回主菜单")
    
    def run(self):
        """运行主程序"""
        print("正在启动社交训练系统...")
        
        # 加载模型
        if not self._load_emotion_model():
            print("错误: 无法加载情绪识别模型")
            return
        
        # 初始化摄像头
        if not self._initialize_camera():
            print("警告: 摄像头不可用，将使用无摄像头模式")
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        print("社交训练系统启动成功")
        print("按 1: 实时情绪识别")
        print("按 2: 打招呼场景")
        print("按 3: 情绪识别训练")
        print("按 4: 查看训练报告")
        print("按 q: 退出程序")
        print("在任何模式中按 0 返回主菜单")
        print("菜单已显示，等待按键输入...")
        
        # 主循环
        while self.running:
            # 读取摄像头帧
            frame, success = self._read_camera_frame()
            
            if self.current_mode == 'menu':
                # 菜单模式：显示摄像头画面和菜单
                display_frame = self._display_menu(frame)
                cv2.imshow(self.window_name, display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    print("退出程序")
                elif key == ord('1'):
                    self.current_mode = 'realtime'
                    print("进入实时情绪识别模式")
                elif key == ord('2'):
                    self.current_mode = 'greeting'
                    print("进入打招呼场景")
                elif key == ord('3'):
                    self.current_mode = 'emotion_training'
                    print("进入情绪识别训练")
                elif key == ord('4'):
                    self.current_mode = 'report'
                    print("显示训练报告")
            
            elif self.current_mode == 'realtime':
                self.run_realtime_mode()
            
            elif self.current_mode == 'greeting':
                self.run_greeting_scene()
            
            elif self.current_mode == 'emotion_training':
                self.run_emotion_training_scene()
            
            elif self.current_mode == 'report':
                self.show_training_report()
            
            # 防止无限循环，添加短暂延迟
            time.sleep(0.01)
        
        # 程序退出时释放资源
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    app = SocialTrainingSystem()
    app.run()