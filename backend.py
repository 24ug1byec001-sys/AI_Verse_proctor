# working_server.py - Proper camera detection with YOLO integration
import asyncio
import websockets
import json
import time
import random
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import torch

class WorkingProctoringServer:
    def __init__(self):
        self.connected_clients = set()
        self.camera_active = False
        self.cap = None
        self.camera_checked = False
        
        # YOLO model initialization
        self.yolo_model = None
        self.yolo_initialized = False
        self.initialize_yolo()
        
    def initialize_yolo(self):
        """Initialize YOLO model for object detection"""
        try:
            print("🔄 Loading YOLO model...")
            # Load the YOLOv8 model (automatically downloads if not available)
            self.yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
            
            # Test the model with a dummy tensor to ensure it's loaded
            dummy_input = torch.zeros(1, 3, 640, 640)
            _ = self.yolo_model(dummy_input)
            
            self.yolo_initialized = True
            print("✅ YOLO model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("📋 Install ultralytics: pip install ultralytics")
            self.yolo_initialized = False

    async def handle_client(self, websocket):
        """Handle incoming WebSocket connections"""
        self.connected_clients.add(websocket)
        client_id = id(websocket)
        print(f"✅ Client {client_id} connected. Total clients: {len(self.connected_clients)}")
        
        try:
            # Initialize camera on first connection
            if not self.camera_checked:
                await self.initialize_camera()
                self.camera_checked = True
            
            # Main loop for sending data
            while True:
                # Generate data based on camera status
                if self.camera_active and self.cap is not None:
                    proctoring_data = await self.get_real_camera_data()
                else:
                    proctoring_data = await self.generate_simulated_data()
                
                # Send to client
                try:
                    await websocket.send(json.dumps(proctoring_data))
                except websockets.exceptions.ConnectionClosed:
                    break
                    
                # Control frame rate
                await asyncio.sleep(0.5)  # 2 FPS for testing
                
        except Exception as e:
            print(f"❌ Error handling client {client_id}: {e}")
        finally:
            self.connected_clients.remove(websocket)
            print(f"🔌 Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")

    async def initialize_camera(self):
        """Properly check for camera availability with enhanced detection"""
        print("🔍 Checking for available cameras...")
        self.camera_active = False
        
        # Test different camera indices with proper cleanup
        for i in range(0, 5):  # Check indices 0-4
            cap = None
            try:
                print(f"  Testing camera index {i}...")
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW on Windows for better performance
                
                if cap.isOpened():
                    # Set camera properties first
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    
                    # Try to read multiple frames to confirm camera works
                    frames_read = 0
                    for _ in range(5):  # Try 5 times
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            frames_read += 1
                            await asyncio.sleep(0.1)  # Small delay between frames
                    
                    if frames_read >= 3:  # Require at least 3 successful frames
                        print(f"✅ Camera found at index {i}! ({frames_read}/5 frames read)")
                        self.cap = cap
                        self.camera_active = True
                        break
                    else:
                        print(f"❌ Camera index {i} opened but cannot read stable frames ({frames_read}/5)")
                        cap.release()
                else:
                    print(f"❌ No camera at index {i}")
                    if cap:
                        cap.release()
                        
            except Exception as e:
                print(f"❌ Error testing camera index {i}: {e}")
                if cap is not None:
                    cap.release()
        
        if not self.camera_active:
            print("📷 No working camera found. Using simulated data.")
        else:
            print("📷 Camera initialized successfully!")

    async def get_real_camera_data(self):
        """Get data from real camera with YOLO processing"""
        try:
            if self.cap is None or not self.cap.isOpened():
                self.camera_active = False
                return await self.generate_simulated_data()
            
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("❌ Failed to read frame from camera")
                self.camera_active = False
                return await self.generate_simulated_data()
            
            # Process frame with YOLO
            processed_frame, metrics = await self.process_frame_with_yolo(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'frame': frame_b64,
                'metrics': metrics,
                'timestamp': time.time(),
                'camera_status': 'real',
                'data_source': 'real_camera'
            }
            
        except Exception as e:
            print(f"❌ Error processing camera data: {e}")
            self.camera_active = False
            return await self.generate_simulated_data()

    async def process_frame_with_yolo(self, frame):
        """Process frame using YOLO model for object detection"""
        metrics = {
            'head_yaw': 0.0,
            'gaze_deviation': 0.0,
            'device_detected': False,
            'alert_active': False,
            'latest_alert_type': '',
            'person_detected': False,
            'phone_detected': False,
            'laptop_detected': False,
            'confidence': 0.0,
            'detection_count': 0
        }
        
        try:
            # Run YOLO inference
            if self.yolo_initialized and self.yolo_model is not None:
                results = self.yolo_model(frame, verbose=False)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        metrics['detection_count'] = len(boxes)
                        
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # COCO dataset class indices
                            if cls == 0:  # person
                                metrics['person_detected'] = True
                                metrics['confidence'] = max(metrics['confidence'], conf)
                                
                                # Draw bounding box for person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                            elif cls == 67:  # cell phone
                                metrics['phone_detected'] = True
                                metrics['device_detected'] = True
                                
                                # Draw bounding box for phone
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f'Phone {conf:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                            elif cls == 63:  # laptop
                                metrics['laptop_detected'] = True
                                metrics['device_detected'] = True
                                
                                # Draw bounding box for laptop
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f'Laptop {conf:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Generate alerts based on detections
            if metrics['phone_detected']:
                metrics['alert_active'] = True
                metrics['latest_alert_type'] = "Mobile Phone Detected"
            elif not metrics['person_detected']:
                metrics['alert_active'] = True
                metrics['latest_alert_type'] = "No Person Detected"
            else:
                metrics['alert_active'] = False
                metrics['latest_alert_type'] = "Normal - Person Detected"
                
            # Simulate head pose and gaze (replace with real models if available)
            metrics['head_yaw'] = round(random.uniform(-5, 5), 2)
            metrics['gaze_deviation'] = round(random.uniform(0.1, 0.3), 3)
            
        except Exception as e:
            print(f"❌ Error in YOLO processing: {e}")
            # Fallback to basic processing
            metrics = await self.generate_simulated_metrics()
        
        # Add overlays to frame
        frame = self.add_frame_overlays(frame, metrics)
        
        return frame, metrics

    def add_frame_overlays(self, frame, metrics):
        """Add information overlays to the frame"""
        # Add timestamp and status
        cv2.putText(frame, f"REAL CAMERA - YOLO DETECTION - {time.strftime('%H:%M:%S')}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add status indicator
        status_color = (0, 0, 255) if metrics['alert_active'] else (0, 255, 0)
        status_text = "🔴 ALERT" if metrics['alert_active'] else "🟢 NORMAL"
        cv2.putText(frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add detection information
        cv2.putText(frame, f"Person: {'YES' if metrics['person_detected'] else 'NO'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Phone: {'YES' if metrics['phone_detected'] else 'NO'}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Laptop: {'YES' if metrics['laptop_detected'] else 'NO'}", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Detections: {metrics['detection_count']}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add alert message if any
        if metrics['alert_active']:
            cv2.putText(frame, f"Alert: {metrics['latest_alert_type']}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    async def generate_simulated_data(self):
        """Generate fully simulated data"""
        metrics = await self.generate_simulated_metrics()
        frame = self.create_simulated_frame(
            metrics['alert_active'], 
            metrics['latest_alert_type'],
            metrics['head_yaw'],
            metrics['gaze_deviation']
        )
        
        return {
            'frame': frame,
            'metrics': metrics,
            'timestamp': time.time(),
            'camera_status': 'simulated',
            'data_source': 'simulation'
        }

    async def generate_simulated_metrics(self):
        """Generate realistic simulated proctoring metrics"""
        # Simulate occasional alerts (15% chance)
        has_alert = random.random() < 0.15
        
        if has_alert:
            alert_type = random.choice(['head_pose', 'eye_gaze', 'device'])
            
            if alert_type == 'head_pose':
                head_yaw = random.uniform(18, 35)  # High head movement
                gaze_deviation = random.uniform(0.1, 0.4)
                device_detected = False
                alert_message = "Head Pose Alert - Looking Away"
            elif alert_type == 'eye_gaze':
                head_yaw = random.uniform(-8, 8)
                gaze_deviation = random.uniform(0.7, 1.2)  # High gaze deviation
                device_detected = False
                alert_message = "Eye Gaze Alert - Not Focused"
            else:  # device
                head_yaw = random.uniform(-8, 8)
                gaze_deviation = random.uniform(0.1, 0.4)
                device_detected = True
                alert_message = "Device Detected - Phone/Tablet"
        else:
            # Normal state - focused student
            head_yaw = random.uniform(-8, 8)
            gaze_deviation = random.uniform(0.05, 0.25)
            device_detected = False
            alert_message = ""

        return {
            'head_yaw': round(head_yaw, 2),
            'gaze_deviation': round(gaze_deviation, 3),
            'device_detected': device_detected,
            'alert_active': has_alert,
            'latest_alert_type': alert_message,
            'person_detected': not has_alert or random.random() > 0.3,
            'phone_detected': device_detected,
            'laptop_detected': False,
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'detection_count': random.randint(0, 3)
        }

    def create_simulated_frame(self, has_alert, alert_message, head_yaw, gaze_deviation):
        """Create a simulated video frame that clearly shows it's simulated"""
        # Create a gradient background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add gradient
        for i in range(480):
            color_val = int(50 + (i / 480) * 50)
            frame[i, :, :] = [color_val, color_val, color_val]
        
        # Add status-based color tint
        overlay = frame.copy()
        tint_color = (0, 0, 100) if has_alert else (0, 100, 0)  # Red/Green tint
        cv2.addWeighted(overlay, 0.3, 
                       np.full(frame.shape, tint_color, dtype=np.uint8), 0.7, 0, frame)
        
        # Add large "SIMULATION" watermark
        cv2.putText(frame, "SIMULATED FEED", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Add status text
        status_color = (0, 0, 255) if has_alert else (0, 255, 0)
        status_text = "🔴 " + alert_message if has_alert else "🟢 NORMAL - Student Focused"
        cv2.putText(frame, status_text, (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add metrics
        cv2.putText(frame, f"Head Yaw: {head_yaw:.1f}°", (50, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze Deviation: {gaze_deviation:.3f}", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Device Detected: {'YES' if random.random() > 0.9 else 'NO'}", (50, 370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add fake face detection box
        cv2.rectangle(frame, (250, 150), (390, 290), (255, 255, 255), 2)
        cv2.putText(frame, "Face", (260, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add eye tracking dots
        eye_x = 320 + int(head_yaw * 2)  # Move eyes based on head yaw
        cv2.circle(frame, (eye_x - 30, 200), 8, (0, 255, 255), -1)  # Left eye
        cv2.circle(frame, (eye_x + 30, 200), 8, (0, 255, 255), -1)  # Right eye
        
        # Add timestamp
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "DATA: SIMULATED", (50, 440), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode('utf-8')

    async def close_camera(self):
        """Close camera connection"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_active = False
        print("📷 Camera resources released")

    async def health_check(self):
        """Periodic health check"""
        while True:
            await asyncio.sleep(5)
            status = "🟢 REAL CAMERA" if self.camera_active else "🟡 SIMULATED DATA"
            yolo_status = "✅ YOLO" if self.yolo_initialized else "❌ YOLO"
            print(f"📊 Server Status: {len(self.connected_clients)} clients | {status} | {yolo_status}")

async def main():
    server = WorkingProctoringServer()
    
    # Start health check task
    health_task = asyncio.create_task(server.health_check())
    
    try:
        # Start WebSocket server
        start_server = await websockets.serve(
            lambda ws: server.handle_client(ws), 
            "localhost", 
            8765,
            ping_interval=20,
            ping_timeout=60
        )
        
        print("=" * 60)
        print("🚀 WORKING PROCTORING SERVER WITH YOLO STARTED!")
        print("📍 Server: ws://localhost:8765")
        print("📡 Status: Waiting for connections...")
        print("💡 System will automatically detect camera availability")
        print("🤖 YOLO Model: Integrated for object detection")
        print("=" * 60)
        
        # Keep server running
        await start_server.wait_closed()
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
    finally:
        health_task.cancel()
        await server.close_camera()

if __name__ == "__main__":
    print("Starting Working Proctoring Server with YOLO...")
    asyncio.run(main())