import gradio as gr
from skat_game import SkatGame, GameState, StrategyEngine
import cv2
import numpy as np
import os
from pathlib import Path


class SkatGameUI:
    """
    UI interface for Skat card game analysis with separate uploads for hand and table cards.
    Uses YOLO model for card detection and OpenAI for strategy suggestions.
    """

    def __init__(self):
        """Initialize game state variables."""
        self.game = None
        self.model_path = None
        self.latest_results = {"strategy": "", "state": "", "frame": None}
        self.hand_cards_detections = None
        self.table_cards_detections = None
        
    def initialize_game(self, model_file, api_key):
        """Set up game with provided YOLO model and OpenAI API key."""
        if not model_file or not api_key.strip():
            return "Please provide both YOLO model file and OpenAI API key", None, None, None
        
        try:
            temp_dir = Path("temp_models")
            temp_dir.mkdir(exist_ok=True)
            
            if isinstance(model_file, str):
                self.model_path = model_file
            else:
                save_path = temp_dir / "model.pt"
                with open(save_path, "wb") as f:
                    f.write(model_file)
                self.model_path = str(save_path)
            
            self.game = SkatGame(self.model_path, api_key)
            return "Game initialized successfully! Upload hand and table cards for analysis.", None, None, None
            
        except Exception as e:
            return f"Error initializing game: {str(e)}", None, None, None

    def process_hand_cards(self, image):
        """Process hand cards image and store detections."""
        try:
            if not self.game or image is None:
                return None, "Please initialize game first"
            
            if isinstance(image, str):
                frame = cv2.imread(image)
            else:
                frame = image
                
            detections = self.game.card_detector(frame)
            self.hand_cards_detections = detections
            
            annotated_frame = self._create_annotated_frame(frame, detections)
            detected_cards = [self.game.strategy_engine._format_detection(card) 
                            for card in detections]
            
            return annotated_frame, f"Hand cards detected: {', '.join(str(c) for c in detected_cards)}"
            
        except Exception as e:
            return None, f"Error processing hand cards: {str(e)}"

    def process_table_cards(self, image):
        """Process table cards image and store detections."""
        try:
            if not self.game or image is None:
                return None, "Please initialize game first"
            
            if isinstance(image, str):
                frame = cv2.imread(image)
            else:
                frame = image
                
            detections = self.game.card_detector(frame)
            self.table_cards_detections = detections
            
            annotated_frame = self._create_annotated_frame(frame, detections)
            detected_cards = [self.game.strategy_engine._format_detection(card) 
                            for card in detections]
            
            return annotated_frame, f"Table cards detected: {', '.join(str(c) for c in detected_cards)}"
            
        except Exception as e:
            return None, f"Error processing table cards: {str(e)}"

    def get_suggestion(self):
        """Generate strategy suggestion based on current hand and table cards."""
        try:
            if not self.game:
                return "Please initialize game first", ""
                
            if not self.hand_cards_detections:
                return "Please upload hand cards first", ""
                
            # Update game state with current detections
            self.game.game_state.hand_cards = self.hand_cards_detections or []
            self.game.game_state.table_cards = self.table_cards_detections or []
            
            # Get strategy suggestion
            strategy = self.game.strategy_engine.decide_move(self.game.game_state)
            state_info = self._format_game_state()
            
            return strategy if strategy else "Waiting for your turn...", state_info
            
        except Exception as e:
            return f"Error getting suggestion: {str(e)}", ""

    def _create_annotated_frame(self, frame, detections):
        """Create annotated frame with detection boxes and labels."""
        annotated = frame.copy()
        
        for det in detections:
            boxes = det.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                label = f"{det.names[class_id]} ({confidence:.2f})"
                cv2.putText(annotated, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return annotated
        
    def _format_game_state(self):
        """Format current game state as string for display."""
        hand_cards = [self.game.strategy_engine._format_detection(card) 
                     for card in (self.hand_cards_detections or [])]
        table_cards = [self.game.strategy_engine._format_detection(card) 
                      for card in (self.table_cards_detections or [])]
        
        return (f"Hand Cards: {', '.join(str(c) for c in hand_cards)}\n"
                f"Table Cards: {', '.join(str(c) for c in table_cards)}")

# def launch_ui():
#     """Launch Gradio interface for Skat game analysis."""
#     skat_ui = SkatGameUI()
    
#     with gr.Blocks(title="Skat Game Analysis System") as interface:
#         gr.Markdown("# Skat Game Analysis System")
#         gr.Markdown("Upload your YOLO model (easy to replace new version for Object Detection Model) and enter your OpenAI API key to start")
        
#         with gr.Row():
#             with gr.Column(scale=1):
#                 model_file = gr.File(
#                     label="YOLO Model (.pt file)",
#                     file_types=[".pt"],
#                     type="binary"
#                 )
#             with gr.Column(scale=1):
#                 api_key_input = gr.Textbox(
#                     label="OpenAI API Key",
#                     type="password",
#                     placeholder="Enter your OpenAI API key here..."
#                 )
        
#         init_button = gr.Button("Initialize Game")
#         status_message = gr.Textbox(label="Status", interactive=False)
        
#         # Card Upload Sections
#         with gr.Row():
#             # Hand Cards Section
#             with gr.Column(scale=1):
#                 gr.Markdown("### Your Hand Cards")
#                 hand_input = gr.Image(
#                     label="Upload Hand Cards Image",
#                     type="numpy"
#                 )
#                 hand_output = gr.Image(label="Detected Hand Cards")
#                 hand_status = gr.Textbox(label="Hand Cards Status")
            
#             # Table Cards Section
#             with gr.Column(scale=1):
#                 gr.Markdown("### Cards on Table")
#                 table_input = gr.Image(
#                     label="Upload Table Cards Image",
#                     type="numpy"
#                 )
#                 table_output = gr.Image(label="Detected Table Cards")
#                 table_status = gr.Textbox(label="Table Cards Status")
        
#         # Strategy Section
#         with gr.Row():
#             suggestion_button = gr.Button("Get Strategy Suggestion", size="large")
#             strategy_text = gr.Textbox(
#                 label="Strategy Suggestion",
#                 lines=2
#             )
#             game_state_text = gr.Textbox(
#                 label="Current Game State",
#                 lines=4
#             )
        
#         # Event handlers
#         init_button.click(
#             fn=skat_ui.initialize_game,
#             inputs=[model_file, api_key_input],
#             outputs=[status_message, hand_output, strategy_text, game_state_text]
#         )
        
#         hand_input.change(
#             fn=skat_ui.process_hand_cards,
#             inputs=[hand_input],
#             outputs=[hand_output, hand_status]
#         )
        
#         table_input.change(
#             fn=skat_ui.process_table_cards,
#             inputs=[table_input],
#             outputs=[table_output, table_status]
#         )
        
#         suggestion_button.click(
#             fn=skat_ui.get_suggestion,
#             inputs=None,
#             outputs=[strategy_text, game_state_text]
#         )
    
#     interface.launch(share=True)

def launch_ui():
    """Launch enhanced Gradio interface for Skat game analysis."""
    skat_ui = SkatGameUI()
    
    with gr.Blocks(title="Skat Game Analysis System", theme="soft") as interface:
        # Header Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                # 🎮 Skat Game Analysis System
                #### Your AI-Powered Skat Strategy Assistant
                
                This system helps you make optimal moves in Skat by analyzing your cards and providing strategic advice.
                """)
        
        # Setup Instructions
        with gr.Accordion("📋 Setup Instructions", open=False):
            gr.Markdown("""
            ### How to Use:
            1. Upload your YOLO model file (.pt) - This is used for card detection
            2. Enter your OpenAI API key for strategy analysis
            3. Upload images of your hand cards and table cards
            4. Get AI-powered strategy suggestions!
            
            ### Tips:
            - Ensure good lighting when taking card photos
            - Keep cards clearly visible and separated
            - Make sure all card details are readable
            """)
        
        # Model and API Setup Section
        with gr.Group():
            gr.Markdown("### 🚀 Initialize System")
            with gr.Row():
                with gr.Column(scale=1):
                    model_file = gr.File(
                        label="YOLO Model File (.pt)",
                        file_types=[".pt"],
                        type="binary"
                    )
                    gr.Markdown("*Upload your trained YOLO model for card detection*", elem_classes="small-text")
                with gr.Column(scale=1):
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="Enter your OpenAI API key here...",
                        container=False
                    )
                    gr.Markdown("*Your key is securely handled and never stored*", elem_classes="small-text")
            
            init_button = gr.Button("🎯 Initialize Game", variant="primary", size="lg")
            status_message = gr.Textbox(
                label="System Status",
                interactive=False,
                container=False
            )
        
        # Card Analysis Section
        with gr.Tabs():
            # Hand Cards Tab
            with gr.Tab("🎴 Your Hand"):
                with gr.Row():
                    with gr.Column():
                        hand_input = gr.Image(
                            label="Upload Hand Cards",
                            type="numpy",
                            elem_id="hand-input"
                        )
                        gr.Markdown("*Take a clear photo of your cards in hand*")
                    with gr.Column():
                        hand_output = gr.Image(
                            label="Detected Hand Cards",
                            elem_id="hand-output"
                        )
                        hand_status = gr.Textbox(
                            label="Detection Results",
                            elem_id="hand-status"
                        )
            
            # Table Cards Tab
            with gr.Tab("🃏 Table Cards"):
                with gr.Row():
                    with gr.Column():
                        table_input = gr.Image(
                            label="Upload Table Cards",
                            type="numpy",
                            elem_id="table-input"
                        )
                        gr.Markdown("*Take a clear photo of cards on the table*")
                    with gr.Column():
                        table_output = gr.Image(
                            label="Detected Table Cards",
                            elem_id="table-output"
                        )
                        table_status = gr.Textbox(
                            label="Detection Results",
                            elem_id="table-status"
                        )
        
        # Strategy Section
        with gr.Group():
            gr.Markdown("### 🧠 Strategic Analysis")
            with gr.Row():
                suggestion_button = gr.Button(
                    "🎲 Get Strategy Suggestion",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Row():
                with gr.Column():
                    strategy_text = gr.Textbox(
                        label="Strategic Recommendation",
                        lines=4,
                        elem_id="strategy-text"
                    )
                with gr.Column():
                    game_state_text = gr.Textbox(
                        label="Current Game State",
                        lines=4,
                        elem_id="game-state"
                    )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📝 Notes:
        - The system analyzes both hand and table cards to provide optimal strategy
        - Suggestions consider game rules, card values, and potential winning combinations
        - For best results, ensure all cards are clearly visible in the images
        
        *Built with Gradio, YOLO, and OpenAI GPT-4*
        """)
        
        # Custom CSS for better styling
        gr.Markdown("""
        <style>
        .small-text {
            font-size: 0.8em;
            color: #666;
        }
        #hand-input, #table-input {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 10px;
        }
        #strategy-text {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
        }
        .primary-button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .gradio-container {
            max-width: 1200px !important;
        }
        </style>
        """)
        
        # Event handlers
        init_button.click(
            fn=skat_ui.initialize_game,
            inputs=[model_file, api_key_input],
            outputs=[status_message, hand_output, strategy_text, game_state_text]
        )
        
        hand_input.change(
            fn=skat_ui.process_hand_cards,
            inputs=[hand_input],
            outputs=[hand_output, hand_status]
        )
        
        table_input.change(
            fn=skat_ui.process_table_cards,
            inputs=[table_input],
            outputs=[table_output, table_status]
        )
        
        suggestion_button.click(
            fn=skat_ui.get_suggestion,
            inputs=None,
            outputs=[strategy_text, game_state_text]
        )
    
    # Launch with custom configurations
    interface.launch(
        share=True
    )
if __name__ == "__main__":
    launch_ui()