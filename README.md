# Angry Message Detector

A simple neural network classifier that detects angry/urgent messages vs neutral ones using PyTorch and TF-IDF features.

## Features

- Binary text classification (angry vs neutral)
- TF-IDF vectorization with uni/bi-grams
- Small feedforward neural network
- Training with backpropagation
- Easy to adapt to your own data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/angry-message-detector.git
cd angry-message-detector

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using the example data

```bash
python angry_message_detector.py
```

This will train the model on the included example data and show predictions on test messages.

### Using your own data

1. Prepare a CSV file with columns `text` and `label`:
   - `text`: your message content
   - `label`: 1 for angry/urgent, 0 for neutral

2. Place your CSV in the `data/` directory (e.g., `data/my_messages.csv`)

3. Run the training script:

```bash
python train_with_csv.py --data data/my_messages.csv --epochs 50
```

## Project Structure

```
angry-message-detector/
├── README.md
├── requirements.txt
├── angry_message_detector.py    # Main script with example data
├── train_with_csv.py            # Script for training with CSV data
├── data/
│   └── example_messages.csv     # Example training data
└── models/
    └── .gitkeep
```

## Example Data Format

```csv
text,label
"This is unacceptable, I've asked three times already!",1
"Hello, could you please check my order status?",0
"I am extremely disappointed with your service.",1
"Thanks for your help, appreciated!",0
```

## How It Works

1. **Text Vectorization**: Converts messages to TF-IDF numerical features
2. **Neural Network**: 2-layer feedforward network (input → 64 hidden units → output)
3. **Training**: Uses binary cross-entropy loss and backpropagation
4. **Prediction**: Outputs probability that a message is angry (threshold: 0.5)

## Model Architecture

```
Input (TF-IDF features) → Linear(hidden=64) → ReLU → Dropout(0.1) → Linear(1) → Sigmoid
```

## Customization

Edit hyperparameters in the script:

```python
# In angry_message_detector.py or train_with_csv.py
EPOCHS = 25           # Number of training passes
HIDDEN = 64           # Hidden layer size
LEARNING_RATE = 1e-3  # Optimizer learning rate
BATCH_SIZE = 4        # Mini-batch size
MAX_FEATURES = 300    # TF-IDF vocabulary size
```

## Performance

On the small example dataset (10 messages), the model quickly learns to distinguish angry from neutral messages. For real-world use, you'll need:

- At least 100-1000 labeled examples
- Balanced classes (similar number of angry/neutral)
- Larger `max_features` (e.g., 5000-20000)

## Saving and Loading Models

See `train_with_csv.py` for examples of:
- Saving trained models to `models/`
- Loading models for inference
- Saving the TF-IDF vectorizer

## License

MIT

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

## Acknowledgments

Built with PyTorch and scikit-learn.
