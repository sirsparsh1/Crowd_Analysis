import os
import cv2
import pandas as pd
import time
import argparse
from datetime import datetime
from collections import deque
from math import sqrt

from google.cloud import vision
from google.oauth2 import service_account

class VideoActivityAnalyzer:
    # --- Constants for Configuration ---
    SENTIMENT_MAP = ['UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY']
    
    # Object categories for activity detection
    PHONE_OBJECTS = ['Mobile phone', 'Telephone', 'Electronics', 'Communication Device']
    FOOD_OBJECTS = ['Food', 'Fast food', 'Snack', 'Drink', 'Beverage', 'Bottle']
    SITTING_OBJECTS = ['Chair', 'Couch', 'Bench', 'Furniture']
    
    # History tracking parameters
    POSITION_HISTORY_MAXLEN = 15
    EMOTION_HISTORY_MAXLEN = 10
    ACTIVITY_HISTORY_MAXLEN = 5
    
    # Detection thresholds
    PHONE_PROXIMITY_THRESHOLD = 0.2  # Normalized distance
    FOOD_PROXIMITY_THRESHOLD = 0.3   # Normalized distance
    SITTING_Y_VARIANCE_THRESHOLD = 30  # Pixel variance for sitting detection
    PERSON_MATCHING_THRESHOLD = 0.3  # Normalized distance to match a person between frames

    def __init__(self, key_path, csv_path='crowd_data.csv'):
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Google Cloud key file not found at: {key_path}")
            
        try:
            credentials = service_account.Credentials.from_service_account_file(key_path)
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        except Exception as e:
            raise Exception(f"Failed to initialize Vision API client: {e}")
        
        self.csv_path = csv_path
        self.tracked_persons = {}
        self.next_person_id = 1
        
        self.image_width = None
        self.image_height = None
        
    def get_optimal_frame_interval(self, video_path):
        """Calculate optimal frame extraction interval based on video properties."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"Video Info: {fps} FPS, {total_frames} frames, {duration:.1f} seconds")
        print(f"Resolution: {self.image_width}x{self.image_height}")
        
        # Simple interval: 1 frame per second
        interval = max(1, fps)
            
        print(f"Using frame interval: {interval} (analyzing every {interval} frames)")
        return interval, fps

    def analyze_image(self, frame_content):
        """Detects objects and faces in an image frame using the Vision API."""
        image = vision.Image(content=frame_content)
        try:
            response_face = self.client.face_detection(image=image, max_results=20)
            response_objects = self.client.object_localization(image=image, max_results=50)
            
            if response_face.error.message:
                raise Exception(f'Face detection error: {response_face.error.message}')
            if response_objects.error.message:
                raise Exception(f'Object detection error: {response_objects.error.message}')
            
            return response_objects.localized_object_annotations, response_face.face_annotations
        except Exception as e:
            print(f"Vision API error: {e}")
            return [], []

    def _get_entity_center(self, vertices, is_normalized=True):
        """Calculates the center of a bounding box."""
        if not self.image_width or not self.image_height:
            return 0, 0

        if is_normalized:
            center_x = sum(v.x for v in vertices) / len(vertices) * self.image_width
            center_y = sum(v.y for v in vertices) / len(vertices) * self.image_height
        else:
            center_x = sum(v.x for v in vertices) / len(vertices)
            center_y = sum(v.y for v in vertices) / len(vertices)
        return center_x, center_y
    
    def _update_person_tracking(self, faces_in_frame):
        """Matches faces to existing tracked persons or creates new ones."""
        matched_person_ids = set()
        unmatched_faces = list(faces_in_frame)
        
        # Try to match existing persons
        for person_id, person_data in self.tracked_persons.items():
            if not person_data['face_positions']:
                continue
                
            last_pos = person_data['face_positions'][-1]
            
            best_match_face = None
            min_dist = float('inf')
            
            for face in unmatched_faces:
                face_center_x, face_center_y = self._get_entity_center(face.bounding_poly.vertices, is_normalized=False)
                dist = sqrt((face_center_x - last_pos['x'])**2 + (face_center_y - last_pos['y'])**2)
                
                # Normalize distance by image dimensions for consistent thresholding
                if self.image_width and self.image_height:
                    norm_dist = dist / sqrt(self.image_width**2 + self.image_height**2)
                else:
                    norm_dist = float('inf')

                if norm_dist < self.PERSON_MATCHING_THRESHOLD and norm_dist < min_dist:
                    min_dist = norm_dist
                    best_match_face = face

            if best_match_face:
                self._update_person_data(person_id, best_match_face)
                matched_person_ids.add(person_id)
                unmatched_faces.remove(best_match_face)
        
        # Add new persons for unmatched faces
        for face in unmatched_faces:
            new_id = f"person_{self.next_person_id}"
            self.next_person_id += 1
            self.tracked_persons[new_id] = {
                'face_positions': deque(maxlen=self.POSITION_HISTORY_MAXLEN),
                'emotion_history': deque(maxlen=self.EMOTION_HISTORY_MAXLEN),
                'activity_history': deque(maxlen=self.ACTIVITY_HISTORY_MAXLEN),
                'current_face': face
            }
            self._update_person_data(new_id, face)
            matched_person_ids.add(new_id)

    def _update_person_data(self, person_id, face):
        """Updates a tracked person's data with new information from a face."""
        center_x, center_y = self._get_entity_center(face.bounding_poly.vertices, is_normalized=False)
        width = max(v.x for v in face.bounding_poly.vertices) - min(v.x for v in face.bounding_poly.vertices)
        height = max(v.y for v in face.bounding_poly.vertices) - min(v.y for v in face.bounding_poly.vertices)

        self.tracked_persons[person_id]['face_positions'].append({'x': center_x, 'y': center_y, 'width': width, 'height': height})
        self.tracked_persons[person_id]['current_face'] = face

    def _is_using_phone(self, person_pos, detected_objects):
        """Detects phone usage based on proximity to face."""
        phones = [obj for obj in detected_objects if any(term.lower() in obj.name.lower() for term in self.PHONE_OBJECTS)]
        if not phones: 
            return False

        if not self.image_width or not self.image_height:
            return False

        norm_face_x = person_pos['x'] / self.image_width
        norm_face_y = person_pos['y'] / self.image_height

        for phone in phones:
            phone_x, phone_y = self._get_entity_center(phone.bounding_poly.normalized_vertices)
            distance = sqrt((phone_x - norm_face_x)**2 + (phone_y - norm_face_y)**2)
            if distance < self.PHONE_PROXIMITY_THRESHOLD:
                return True
        return False

    def _is_eating(self, person_pos, detected_objects):
        """Detects eating based on food proximity to face."""
        food_items = [obj for obj in detected_objects if any(term.lower() in obj.name.lower() for term in self.FOOD_OBJECTS)]
        if not food_items: 
            return False

        if not self.image_width or not self.image_height:
            return False

        norm_face_x = person_pos['x'] / self.image_width
        norm_face_y = person_pos['y'] / self.image_height

        for food in food_items:
            food_x, food_y = self._get_entity_center(food.bounding_poly.normalized_vertices)
            distance = sqrt((food_x - norm_face_x)**2 + (food_y - norm_face_y)**2)
            if distance < self.FOOD_PROXIMITY_THRESHOLD:
                return True
        return False

    def _is_sitting(self, person_data, detected_objects):
        """Detects if a person is sitting."""
        positions = person_data['face_positions']
        furniture = [obj for obj in detected_objects if any(term.lower() in obj.name.lower() for term in self.SITTING_OBJECTS)]

        if len(positions) >= 5:
            y_variance = max(p['y'] for p in positions) - min(p['y'] for p in positions)
            if y_variance < self.SITTING_Y_VARIANCE_THRESHOLD:
                return True
        
        return len(furniture) > 0
    
    def _is_bored(self, person_data, activities):
        """Detects boredom based on emotional state and lack of other activities."""
        face = person_data['current_face']
        emotion_scores = {
            'joy': face.joy_likelihood, 'sorrow': face.sorrow_likelihood,
            'anger': face.anger_likelihood, 'surprise': face.surprise_likelihood
        }
        is_neutral = all(score <= 2 for score in emotion_scores.values()) # POSSIBLY or less
        
        is_engaged = activities['is_using_phone'] or activities['is_eating'] or activities['is_talking']
        
        return is_neutral and activities['is_sitting'] and not is_engaged

    def identify_activities(self, objects, faces):
        """Identifies activities for each tracked person."""
        self._update_person_tracking(faces)
        
        activity_data = {}
        person_count = len(self.tracked_persons)

        for person_id, person_data in self.tracked_persons.items():
            if 'current_face' not in person_data or not person_data['face_positions']:
                continue

            face_pos = person_data['face_positions'][-1]
            
            activities = {
                'is_using_phone': self._is_using_phone(face_pos, objects),
                'is_eating': self._is_eating(face_pos, objects),
                'is_sitting': self._is_sitting(person_data, objects),
                'is_talking': person_count > 1,
            }
            activities['is_bored'] = self._is_bored(person_data, activities)

            activity_data[person_id] = activities
        
        return activity_data

    def process_and_store_data(self, frame_number, objects, activity_data, timestamp):
        """Processes and stores data for all detected entities in a frame."""
        data_to_log = []

        # Process objects
        for obj in objects:
            x_center, y_center = self._get_entity_center(obj.bounding_poly.normalized_vertices)
            data_to_log.append({
                'timestamp': timestamp, 'frame_number': frame_number,
                'object_id': f"{obj.name}-{frame_number}-{obj.score:.2f}",
                'object_type': obj.name,
                'location_x': round(x_center, 4), 'location_y': round(y_center, 4),
                'confidence_score': round(obj.score, 3), 'joy_likelihood': 'N/A',
                'sorrow_likelihood': 'N/A', 'anger_likelihood': 'N/A', 'surprise_likelihood': 'N/A',
                'is_using_phone': False, 'is_sitting': False, 'is_eating': False,
                'is_bored': False, 'is_talking': False
            })

        # Process persons
        for person_id, activities in activity_data.items():
            person_data = self.tracked_persons.get(person_id)
            if not person_data or 'current_face' not in person_data:
                continue
            
            face = person_data['current_face']
            face_pos = person_data['face_positions'][-1]
            
            norm_x = face_pos['x'] / self.image_width if self.image_width else 0
            norm_y = face_pos['y'] / self.image_height if self.image_height else 0
            
            data_to_log.append({
                'timestamp': timestamp, 'frame_number': frame_number,
                'object_id': person_id, 'object_type': 'person',
                'location_x': round(norm_x, 4), 'location_y': round(norm_y, 4),
                'confidence_score': round(face.detection_confidence, 3),
                'joy_likelihood': self.SENTIMENT_MAP[face.joy_likelihood],
                'sorrow_likelihood': self.SENTIMENT_MAP[face.sorrow_likelihood],
                'anger_likelihood': self.SENTIMENT_MAP[face.anger_likelihood],
                'surprise_likelihood': self.SENTIMENT_MAP[face.surprise_likelihood],
                **activities
            })

        if data_to_log:
            df = pd.DataFrame(data_to_log)
            dir_name = os.path.dirname(self.csv_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False)

    def process_video(self, video_path):
        """Main video processing function."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at '{video_path}'")

        try:
            frame_interval, _ = self.get_optimal_frame_interval(video_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
                
            frame_count = 0
            processed_frames = 0
            start_time = time.time()

            print("Starting video processing...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = datetime.now().isoformat()
                    
                    success, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        print(f"Warning: Failed to encode frame {frame_count}")
                        continue
                        
                    content = encoded_image.tobytes()
                    detected_objects, detected_faces = self.analyze_image(content)
                    
                    if detected_objects or detected_faces:
                        activity_data = self.identify_activities(detected_objects, detected_faces)
                        self.process_and_store_data(frame_count, detected_objects, activity_data, timestamp)
                        
                        processed_frames += 1
                        elapsed_time = time.time() - start_time
                        print(f"Processed frame {frame_count} ({processed_frames} total) - "
                              f"Found {len(detected_faces)} faces, {len(detected_objects)} objects - "
                              f"Time elapsed: {elapsed_time:.1f}s")

                frame_count += 1
            
            cap.release()
            total_time = time.time() - start_time
            print(f"\nVideo processing complete!")
            print(f"Total frames: {frame_count}")
            print(f"Processed frames: {processed_frames}")
            print(f"Processing time: {total_time:.1f} seconds")
            print(f"Data saved to: {self.csv_path}")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze crowd activity in a video.")
    parser.add_argument("video_file", help="Path to the video file to analyze.")
    parser.add_argument("--key_path", default="secret_key.json", help="Path to the Google Cloud service account key file.")
    parser.add_argument("--csv_path", default="crowd_data.csv", help="Path to save the output CSV file.")
    args = parser.parse_args()

    try:
        analyzer = VideoActivityAnalyzer(key_path=args.key_path, csv_path=args.csv_path)
        analyzer.process_video(args.video_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nPlease make sure you have:")
        print("1. Installed required packages: pip install google-cloud-vision opencv-python pandas")
        print(f"2. A valid Google Cloud key file at '{args.key_path}'")
        print("3. The Vision API enabled in your Google Cloud project.")
        print(f"4. The video file '{args.video_file}' exists.")