# Decadence

**Frame Interpolation AI**

Decadence is a deep learning-based frame interpolation tool designed to generate intermediate frames between existing video frames. This enhances video smoothness and enables applications such as slow-motion effects and frame rate upscaling.

## Features

- **AI-Powered Interpolation**: Utilizes neural networks to predict and generate intermediate frames.
- **Dataset Support**: Compatible with datasets like Vimeo Triplet for training and evaluation.
- **Modular Architecture**: Organized codebase with separate directories for models, data, and source code.
- **Python-Based**: Implemented entirely in Python for accessibility and ease of use.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Additional dependencies listed in `requirements.txt`

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/EvilFlame112/Decadence.git
   cd Decadence
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:

   - Download the Vimeo Triplet dataset or your dataset of choice.
   - Place it in the `data/` directory, maintaining the expected folder structure.

### Usage

#### Training

To train the model:

```bash
python src/train.py --config configs/train_config.yaml
```

*Note: Replace **`configs/train_config.yaml`** with your actual configuration file.*

#### Inference

To perform frame interpolation on a pair of images:

```bash
python src/infer.py --input_frame1 path_to_frame1 --input_frame2 path_to_frame2 --output_path path_to_output
```

*Note: Adjust the script and parameters as needed based on your implementation.*

## Project Structure

```
Decadence/
├── assets/                 # Contains sample outputs and related assets
├── data/                   # Dataset directory (e.g., Vimeo Triplet)
├── models/                 # Pretrained models and checkpoints
├── src/                    # Source code for training and inference
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/EvilFlame112/Decadence/blob/main/LICENSE) file for details.

## Acknowledgments

- Inspired by existing frame interpolation research and methodologies.
- Utilizes the Vimeo Triplet dataset for training and evaluation.

---

*For more details and updates, visit the *[*Decadence GitHub Repository*](https://github.com/EvilFlame112/Decadence)*.*

