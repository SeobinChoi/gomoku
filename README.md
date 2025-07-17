# Gomoku (Five in a Row) AI

A Python implementation of the Gomoku game with AI players trained using reinforcement learning.

## Features
- Play Gomoku against an AI opponent
- Train your own AI using different algorithms
- Interactive GUI for both playing and training

## Requirements
- Python 3.6+
- Required packages: (list your dependencies here, e.g., numpy, pygame, etc.)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gomoku-ai.git
   cd gomoku-ai
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Play
Run the main game:
```bash
python main.py
```

## Training the AI
To train a new AI model:
```bash
python train_gomoku.py
```

## Project Structure
- `main.py`: Main game interface
- `play_gomoku.py`: Core game logic
- `train_gomoku.py`: Training script for the AI
- `az_gomoku.py`: AlphaZero-style AI implementation
- `Q_logs/`: Directory containing training logs
- `q_table.pkl`: Pre-trained Q-table for the AI

## License
[Your chosen license]
