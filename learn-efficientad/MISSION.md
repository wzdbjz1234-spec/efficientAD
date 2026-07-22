# Mission: Learn EfficientAD for Industrial Anomaly Detection

## Why
I have a working EfficientAD pipeline for detecting product defects via template-matched ROI cropping, but I don't understand what the model actually does. I need to understand the architecture and ML concepts — from CNNs up through EfficientAD's teacher-student design — well enough to debug training issues, tune parameters, and explain results to others.

## Success looks like
- I can draw the EfficientAD architecture on a whiteboard and explain every component's role
- I can trace the code in `efficientad.py` and `common.py` and explain what each function does
- I can describe how the loss functions work and why the teacher-student design detects anomalies
- I can reason about what would happen if I changed a hyperparameter (model size, training steps, etc.)

## Constraints
- Starting from minimal CNN knowledge — need foundational concepts explained first
- Prefer learning through the concrete codebase I already have (`EfficientAD-main/`)
- Chinese is my primary language for learning complex concepts

## Out of scope
- Implementing EfficientAD from scratch
- General computer vision theory beyond what EfficientAD needs
- Other anomaly detection models (PatchCore, PaDiM, etc.)
