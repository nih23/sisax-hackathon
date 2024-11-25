import unittest
import cv2
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
from dotenv import load_dotenv

load_dotenv()

class TestSkatGame(unittest.TestCase):
    def setUp(self):
        self.yolo_path = os.getenv('YOLO_MODEL_PATH', 'default/path/to/model.pt')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

    def create_mock_detection(self, box_data):
        class MockBox:
            def __init__(self, xyxy, cls, conf):
                self.xyxy = xyxy
                self.cls = np.array([cls]).flatten()
                self.conf = np.array([conf]) 
        
        mock_detection = Mock()
        mock_box = MockBox(
            xyxy=np.array([box_data['coords']]),
            cls=box_data['class'],
            conf=box_data['conf']
        )
        
        mock_boxes = Mock()
        mock_boxes.cpu().numpy.return_value = [mock_box]
        mock_detection.boxes = mock_boxes
        mock_detection.names = {0: "Hearts_Ace", 1: "Spades_King"}
        return mock_detection


    @patch('cv2.VideoCapture')
    def test_video_source_initialization(self, mock_cv2):
        mock_cv2.return_value.isOpened.return_value = True
        mock_cv2.return_value.read.return_value = (True, np.zeros((480, 640, 3)))
        
        from skat_game import VideoSource
        video_source = VideoSource()
        self.assertTrue(video_source.initialize())
        ret, frame = video_source.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        video_source.release()

    def test_game_state_card_classification(self):
        from skat_game import GameState
        game_state = GameState()
        
        hand_card = self.create_mock_detection({
            'coords': [100, 100, 200, 200],
            'class': 0,
            'conf': 0.95
        })
        
        table_card = self.create_mock_detection({
            'coords': [100, 400, 200, 500],
            'class': 1,
            'conf': 0.92
        })
        
        self.assertTrue(game_state._is_in_hand(hand_card))
        self.assertTrue(game_state._is_on_table(table_card))
        game_state.update_state([hand_card, table_card])
        self.assertEqual(len(game_state.hand_cards), 1)
        self.assertEqual(len(game_state.table_cards), 1)

    @patch('openai.OpenAI')
    def test_strategy_engine(self, mock_openai):
        from skat_game import StrategyEngine, GameState

        # Mock response that matches _parse_decision criteria
        mock_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [
            type('Choice', (), {'message': type('Message', (), {
                'content': "play the Hearts_Ace as your next move"
            })()})
        ]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        strategy_engine = StrategyEngine(self.openai_key)
        game_state = GameState()

        hand_card = self.create_mock_detection({
            'coords': [100, 100, 200, 200],
            'class': 0,
            'conf': 0.95
        })
        game_state.hand_cards = [hand_card]

        decision = strategy_engine.decide_move(game_state)
        self.assertIsNotNone(decision)
        # Check if decision contains both "play" and "Hearts" keywords
        self.assertTrue(
            "play" in decision.lower() and "Hearts" in decision,
            f"Decision '{decision}' should contain play recommendation"
        )

    def test_openai_key_presence(self):
        self.assertIsNotNone(self.openai_key, "OpenAI API key not found in environment")
        self.assertNotEqual(self.openai_key, "", "OpenAI API key is empty")

if __name__ == '__main__':
    unittest.main()