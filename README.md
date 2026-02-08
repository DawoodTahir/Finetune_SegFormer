# Fine-Tune SegFormer

This repository contains the code, original images, segmentation results, and instructions for fine-tuning the image segmentation model on custom datasets.

## Repository Structure

```
├── finetune.py              # Main fine-tuning script
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── input_images/            # Input images for segmentation
└── README.md                # This file
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required libraries listed in `requirements.txt`

### Installation

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Fine-Tuning Script

To fine-tune the model on your custom dataset, use the following command:

```bash
python finetune.py
```

## Results

Below are examples of the original images and their corresponding segmentation results:

| Original Image | Segmentation Result |
|---|---|
| ![Original Image 1](input_images/orig1.png) | ![Segmentation Result 1](input_images/result1.png) |
| ![Original Image 2](input_images/orig2.png) | ![Segmentation Result 2](input_images/result2.png) |

## Usage

1. Prepare your custom dataset and place images in the `input_images/` directory
2. Configure the fine-tuning parameters in `finetune.py` if needed
3. Run the fine-tuning script: `python finetune.py`
4. Check the output directory for segmentation results

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.