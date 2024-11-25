Here’s an extended and detailed version of the `README.md` file with more sections to comprehensively document your project.

---

# Skat Game AI: YOLO Detection & GPT Strategy

This project blends advanced **AI strategies** and **object detection** to enhance gameplay in the card game **Skat**. Using a YOLO model for real-time card recognition and OpenAI's GPT for strategic recommendations, this project aims to simulate intelligent gameplay and assist players with data-driven insights.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Components](#project-components)
- [Testing](#testing)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Skat Game AI** utilizes:
1. **YOLO Object Detection** to recognize cards in real-time.
2. **OpenAI GPT Models** for crafting intelligent and strategic gameplay.
3. An **interactive interface** for both image-based and live-stream object detection.

The goal is to create a real-time system for card detection, gameplay state management, and strategic analysis using state-of-the-art AI models.

---

## Features

- **Card Recognition**: Detects cards from video streams or uploaded images.
- **AI Strategy Engine**: Provides recommendations for the best moves based on game state.
- **Real-Time Video Processing**: Processes live video input and detects cards on the fly.
- **Interactive UI**: Gradio-powered interface for demonstrations and testing.

---

## Demo

You can experience this project via two methods:
1. **Command-line Gameplay**: Real-time gameplay with card detection and AI strategies.
2. **Web UI**: Interactive interface using Gradio.

---

## Installation

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- GPU with CUDA (optional but recommended for YOLO model acceleration)

### Steps

1. Clone the repository:
   ```bash
   https://github.com/QsingularityAi/sisax-hackathon-Skat-trifft-ML.git
   cd sisax-hackathon-Skat-trifft-ML
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file from `.env.example`:
   ```env
   YOLO_MODEL_PATH=<path_to_yolo_model.pt>
   OPENAI_API_KEY=<your_openai_api_key>
   ```

4. Ensure a trained YOLO `.pt` model is available in the specified path.
   ```
   For Demo purposes, you can use `best.pt` is which trained on skat cards with Yolo11 model.
   ```

---

## Usage

### Running the Skat Game

Launch the real-time Skat game using Terminal you can get real time Object detction and AI strategy:
```bash
python skat_game.py
```
### Running the Gradio UI got AI strategy

Launch the Gradio-based interface for testing card detection upload hand and table card to get AI strategy:
```bash
python skat_game_ui.py
```


### Just for Curosity How to build custom Object Detection custom model

please Use jyupter Notobook on google colab make sure you already have own labled data sets which same feature if want use pre-train model:
```bash
Custom_skat_yolo_model.ipynb
```

### To test own model Running the Gradio UI

Launch the Gradio-based interface for testing card detection:
```bash
python yolo_prediction.py
```

---

## Project Components

### 1. **Skat Game Engine** (`skat_game.py`)

Handles the overall gameplay:
- **Game State Management**: Tracks cards in hand, on the table, and played cards.
- **AI Strategy Engine**: Suggests moves based on GPT-powered analysis.

### 2. **Card Detection with Recommendation** (`skat_game_ui.py`)

- Processes images or video streams for card detection.
- Provides a visualized output of detected cards.
- You can upload your own hand and table cards to get AI strategy.

### 3. **Tests Game Engine** (`skat_game_test.py`)

Unit tests for:
- Game state logic
- AI strategy engine
- Video source initialization

### 4. **Gradio Interface for test trained model** (`yolo_prediction.py`)

Interactive interface for:
- Image-based card detection.
- Live video stream detection.

---



## Screenshots

### Skat card with YOLO Detection
![YOLO Detection](Skat-game/Image/Card_OD2.png)

### AI Strategy Recommendation
![AI Strategy](Skat-game/Image/skat_game_recommendation.png)

---

## Example Code

Here’s an example for integrating YOLO and AI strategy:

```python
import cv2
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
yolo_path = os.getenv("YOLO_MODEL_PATH")
openai_key = os.getenv("OPENAI_API_KEY")

# Load YOLO model
model = YOLO(yolo_path)

# Initialize OpenAI strategy engine
strategy = OpenAI(api_key=openai_key)

# Load an example frame
frame = cv2.imread('example_card.jpg')

# Detect cards
results = model(frame)

# Strategy analysis
game_state = GameState()
game_state.update_state(results)
move = strategy.decide_move(game_state)

# Display results
print(f"Recommended Move: {move}")
```

---

## Future Enhancements

- **Improved Detection**: Enhance the YOLO model with custom datasets for Skat cards.
- **Strategy Refinement**: Train a reinforcement learning model for better move recommendations.
- **Multi-Player Support**: Extend functionality for analyzing multiple players’ strategies.
- **Mobile App**: Port the project to a mobile platform.

---

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

### Development Environment
- **Python 3.10+**
- Recommended IDE: PyCharm or VSCode
- Linting: `flake8`

---

## License

This project is licensed under the GPL-3.0 license. See the `LICENSE` file for details.

---

Let me know if you’d like help generating example images or any additional sections, such as dataset details or detailed troubleshooting.