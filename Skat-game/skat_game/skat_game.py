import cv2
import numpy as np
from ultralytics import YOLO
from openai import OpenAI
import sys
import time
from dotenv import load_dotenv
import os

class GameState:
    def __init__(self):
        self.hand_cards = []
        self.table_cards = []
        self.played_cards = []
        self.score = 0
        self.current_player = 0
        
    def update_state(self, detections):
        self.table_cards = [d for d in detections if self._is_on_table(d)]
        self.hand_cards = [d for d in detections if self._is_in_hand(d)]
        
    def _is_on_table(self, detection):
        boxes = detection.boxes.cpu().numpy()
        for box in boxes:
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            return y_center > 300
        return False
        
    def _is_in_hand(self, detection):
        boxes = detection.boxes.cpu().numpy()
        for box in boxes:
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            return y_center <= 300
        return False
        
    def is_player_turn(self):
        return self.current_player == 0

class VideoSource:
    def __init__(self):
        self.camera_indices = [0, 1, 2]
        self.cap = None
        
    def initialize(self):
        for idx in self.camera_indices:
            print(f"Trying camera index {idx}...")
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                print(f"Successfully opened camera {idx}")
                return True
        
        video_path = input("No camera available. Enter path to video file (or press Enter to quit): ")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                return True
        return False
        
    def read(self):
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None
        
    def release(self):
        if self.cap:
            self.cap.release()

class StrategyEngine:
    """Enhanced strategy engine for Skat game with comprehensive analysis."""
    
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
        # Define system and user prompts
        self.SYSTEM_PROMPT = {
            "role": "system",
            "content": """You are an expert Skat game strategy advisor with decades of experience in professional play and coaching.

            Key Analysis Components:
            1. Game Context
            - Hand composition evaluation
            - Table position assessment
            - Current trick state
            - Game phase analysis
            - Point tracking

            2. Strategic Elements
            - Card rankings and values
            - Suit distribution and control
            - Trump management strategy
            - Opponent probable holdings
            - Point optimization tactics
            - Risk/reward balance

            3. Positional Considerations
            - Lead vs follow position
            - Trump control status
            - Suit establishment opportunities
            - Entry management
            - Defensive requirements

            Output Structure:
            1. Recommended Move:
            [Specific card to play] - Must specify exact card

            2. Strategic Reasoning:
            - Primary objective for this play
            - Position advancement plan
            - Risk management strategy
            - Point accumulation path

            3. Alternative Analysis:
            - Other possible plays
            - Strategic tradeoffs
            - Why main choice is superior

            4. Future Planning:
            - Next trick strategy
            - Hand development path
            - Counter-play preparation
            - Endgame considerations

            Remember:
            - Focus on optimal strategic play regardless of card detection confidence
            - Base decisions on game theory and proven tactics
            - Consider both immediate trick and long-term game impact
            - Account for score situation and game phase"""
        }
            
    def decide_move(self, game_state):
        """Generate strategic move recommendation using enhanced analysis framework."""
        try:
            # Create comprehensive game state analysis
            hand_cards = [self._format_detection(card) for card in game_state.hand_cards]
            table_cards = [self._format_detection(card) for card in game_state.table_cards]
            
            prompt = f"""Current Position Analysis:

            Hand Cards: {hand_cards}
            Table Cards: {table_cards}
            Played Cards: {game_state.played_cards}
            Current Score: {game_state.score}

            Required:
            1. Analyze the current position considering hand composition and table state
            2. Recommend the optimal card to play based on strategic value
            3. Explain strategic reasoning including immediate and future implications
            4. Discuss alternative plays and their strategic tradeoffs
            5. Provide future trick planning and counter-play considerations"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    self.SYSTEM_PROMPT,
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            return self._parse_strategic_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Strategy engine error: {e}")
            return None
            
    def _parse_strategic_response(self, response):
        """Parse and format the strategic response from GPT."""
        try:
            # Split response into sections
            sections = response.split('\n\n')
            
            # Extract main recommendation and reasoning
            recommended_move = ""
            strategic_reasoning = ""
            
            for section in sections:
                if "Recommended" in section or "Play:" in section:
                    recommended_move = section.strip()
                elif "Strategic Reasoning" in section or "Reasoning:" in section:
                    strategic_reasoning = section.strip()
            
            # Format final output
            formatted_response = f"""
                Strategic Recommendation:
                {recommended_move}

                Reasoning:
                {strategic_reasoning}
                """
            return formatted_response.strip()
            
        except Exception as e:
            print(f"Error parsing strategy response: {e}")
            return response.split('\n')[0]  # Return first line as fallback # Return first line as fallback
            
    def _format_detection(self, detection):
        """Format card detection with enhanced detail."""
        boxes = detection.boxes.cpu().numpy()
        classes = detection.names
        formatted_cards = []
        
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            card_name = classes[class_id]
            formatted_cards.append(f"{card_name} ({confidence:.2f})")
            
        return formatted_cards

class SkatGame:
    def __init__(self, yolo_path, openai_key):
        print("Initializing Skat Game with enhanced strategy engine...")
        self.card_detector = YOLO(yolo_path)
        print("YOLO model loaded")
        self.strategy_engine = StrategyEngine(openai_key)
        print("Enhanced strategy engine initialized")
        self.game_state = GameState()
        self.video_source = VideoSource()

    def _display_frame(self, frame, detections):
        """Display frame with enhanced annotations."""
        annotated_frame = frame.copy()
        
        # Draw detections with enhanced information
        for det in detections:
            boxes = det.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Draw detection box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add detailed card information
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                card_name = det.names[class_id]
                label = f"{card_name} ({confidence:.2f})"
                
                # Add card location and role
                location = "Hand" if self.game_state._is_in_hand(det) else "Table"
                
                # Display labels
                cv2.putText(annotated_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, location, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add game state information to display
        state_text = [
            f"Hand Cards: {len(self.game_state.hand_cards)}",
            f"Table Cards: {len(self.game_state.table_cards)}",
            f"Score: {self.game_state.score}"
        ]
        
        # Display game state
        for i, text in enumerate(state_text):
            cv2.putText(annotated_frame, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Skat Game Analysis', annotated_frame)

def main():
    load_dotenv()
    
    YOLO_PATH = os.getenv('YOLO_MODEL_PATH')
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    
    if not OPENAI_KEY:
        raise ValueError("OpenAI API key not found in .env file")
    
    try:
        game = SkatGame(YOLO_PATH, OPENAI_KEY)
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

