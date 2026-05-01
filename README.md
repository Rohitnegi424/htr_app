# Multi-Language Handwritten Text Recognition

This project aims to build a Deep Learning-based system for:
- Recognizing handwritten text in multiple languages
- Translating text into other languages
- Providing accessibility output (TTS, Braille) for visually impaired users

## Project Structure
```bash
└── HTR-Accessibility/
    ├── data/  # datasets (IAM, Hindi, etc.)
    ├── notebooks/ # Jupyter notebooks for experiments
    ├── src/ # source code
    │   ├── accessibility # TTS and Braille conversion
    │   ├── models # deep learning models (CRNN, Transformers, etc.)
    │   ├── preprocessing # image preprocessing scripts
    │   ├── translation # translation module integration
    │   └── utils # helper functions
    ├── .gitignore
    ├── README.md # project overview
    └── requirements.txt # dependencies
```

## Setup
```bash
pip install -r requirements.txt
```