import numpy as np
import cv2
from collections import defaultdict
from datetime import datetime

class PersonCounter:
    def __init__(self, frame_width, frame_height, fps):
        # Format: [x1, y1, x2, y2] as percentages of frame dimensions
        self.entry_zone = [0.2, 0, 0.9, 0.5]  # Left 30% of the frame
        self.exit_zone = [0.9, 0, 1.0, 1.0]  # Right 30% of the frame
        
        # Convert percentages to pixel coordinates
        self.entry_zone_pixels = [
            int(self.entry_zone[0] * frame_width),
            int(self.entry_zone[1] * frame_height),
            int(self.entry_zone[2] * frame_width),
            int(self.entry_zone[3] * frame_height)
        ]
        
        self.exit_zone_pixels = [
            int(self.exit_zone[0] * frame_width),
            int(self.exit_zone[1] * frame_height),
            int(self.exit_zone[2] * frame_width),
            int(self.exit_zone[3] * frame_height)
        ]
        
        # Store data about each person
        self.people_data = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'entered': False,
            'exited': False,
            'in_store': False,
            'positions': []
        })
        
        # Metrics
        self.entries = 0
        self.exits = 0
        self.current_count = 0
        self.max_count = 0
        self.min_count = 0
        self.total_time_spent = 0
        self.people_completed = 0  # People who entered and exited
        self.fps = fps
        
        # Store count over time for visualization
        self.count_history = []
        self.timestamp_history = []
    
    def is_in_zone(self, bbox, zone):
        """Check if a bounding box is in a zone"""
        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = zone
        
        # Calculate center of the bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if center is in zone
        return (zx1 <= center_x <= zx2) and (zy1 <= center_y <= zy2)
    
    def update(self, results, frame_number):
        """Update counter with new detections"""
        current_timestamp = datetime.now()
        
        if results[0].boxes.id is None:
            # No detections in this frame
            self.count_history.append(self.current_count)
            self.timestamp_history.append(current_timestamp)
            return
        
        # Get detections
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        # Track people currently in frame
        current_ids = set()
        
        for i, track_id in enumerate(track_ids):
            current_ids.add(track_id)
            bbox = boxes[i]
            
            # Update person data
            person_data = self.people_data[track_id]
            
            # First time seeing this person
            if person_data['first_seen'] is None:
                person_data['first_seen'] = frame_number
            
            # Update last seen
            person_data['last_seen'] = frame_number
            
            # Store position for trajectory analysis
            person_data['positions'].append((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            # Check if person entered the store
            if not person_data['entered'] and self.is_in_zone(bbox, self.entry_zone_pixels):
                person_data['entered'] = True
                person_data['in_store'] = True
                self.entries += 1
                self.current_count += 1
                self.max_count = max(self.max_count, self.current_count)
                if self.min_count == 0:  # Initialize min_count with first person
                    self.min_count = 1
            
            # Check if person exited the store
            if person_data['in_store'] and not person_data['exited'] and self.is_in_zone(bbox, self.exit_zone_pixels):
                person_data['exited'] = True
                person_data['in_store'] = False
                self.exits += 1
                self.current_count -= 1
                self.min_count = min(self.min_count, self.current_count) if self.min_count > 0 else self.current_count
                
                # Calculate time spent
                if person_data['entered'] and person_data['exited']:
                    frames_spent = person_data['last_seen'] - person_data['first_seen']
                    time_spent_seconds = frames_spent / self.fps
                    self.total_time_spent += time_spent_seconds
                    self.people_completed += 1
        
        # Store current count for visualization
        self.count_history.append(self.current_count)
        self.timestamp_history.append(current_timestamp)
    
    def get_metrics(self):
        """Get current metrics"""
        avg_time_spent = 0
        if self.people_completed > 0:
            avg_time_spent = self.total_time_spent / self.people_completed
        
        return {
            'entries': self.entries,
            'exits': self.exits,
            'current_count': self.current_count,
            'max_count': self.max_count,
            'min_count': self.min_count,
            'avg_time_spent': avg_time_spent,
            'people_completed': self.people_completed
        }
    
    def draw_zones(self, frame):
        """Draw entry and exit zones on the frame"""
        # Draw entry zone (green)
        cv2.rectangle(
            frame,
            (self.entry_zone_pixels[0], self.entry_zone_pixels[1]),
            (self.entry_zone_pixels[2], self.entry_zone_pixels[3]),
            (0, 255, 0), 2
        )
        cv2.putText(
            frame, "ENTRADA", 
            (self.entry_zone_pixels[0] + 10, self.entry_zone_pixels[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # Draw exit zone (red)
        cv2.rectangle(
            frame,
            (self.exit_zone_pixels[0], self.exit_zone_pixels[1]),
            (self.exit_zone_pixels[2], self.exit_zone_pixels[3]),
            (0, 0, 255), 2
        )
        cv2.putText(
            frame, "SAIDA", 
            (self.exit_zone_pixels[0] + 10, self.exit_zone_pixels[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        
        return frame
    
    def draw_metrics(self, frame):
        """Draw metrics on the frame"""
        metrics = self.get_metrics()
        
        # Create a semi-transparent overlay for metrics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw metrics text
        cv2.putText(frame, f"Entradas: {metrics['entries']}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Saidas: {metrics['exits']}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Atual: {metrics['current_count']}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Maximo: {metrics['max_count']}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        avg_time = metrics['avg_time_spent']
        if avg_time > 0:
            minutes = int(avg_time // 60)
            seconds = int(avg_time % 60)
            cv2.putText(frame, f"Tempo medio: {minutes}m {seconds}s", (150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
