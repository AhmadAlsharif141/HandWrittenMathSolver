# Handwritten Math Expression Solver

A computer vision pipeline that reads a handwritten math expression from an image and computes the result — combining digit classification (CNN) with OCR-based operator detection.

Group project — BSc coursework, Machine Learning.

## Overview

The project has two parts:

1. **Digit classification** — trained and compared three models on MNIST to classify handwritten digits
2. **End-to-end expression solver** — given an image of a handwritten arithmetic expression (e.g. `7 + 3 * 2`), the pipeline segments each character, classifies digits using the trained CNN, detects operators via OCR, and computes the final result — respecting order of operations (multiplication/division before addition/subtraction).

## Part 1: Digit Classification

Trained and benchmarked three models on the MNIST dataset:

- **CNN** — Conv2D + MaxPooling layers, trained for 14 epochs
- **SVM**
- **Logistic Regression**

Evaluation included per-model accuracy, confusion matrices, and full classification reports, plus a look at misclassified examples for the CNN.

## Part 2: Expression Segmentation & Solving

- Uses **OpenCV** for grayscale conversion, thresholding, and contour detection to isolate individual characters (digits and operators) in the input image
- Sorts detected characters left-to-right to preserve expression order
- Classifies operators (`+`, `-`, `*`, `/`) using **Tesseract OCR**
- Classifies digits using the trained CNN
- Reconstructs and evaluates the full expression, applying standard order of operations

## Tech stack

Python, TensorFlow/Keras, scikit-learn, OpenCV, Tesseract OCR (pytesseract), NumPy, Matplotlib, Seaborn

## Notes

- Model comparison results (accuracy, confusion matrices) are generated in Part 1 when run against the MNIST test set
- Part 2 expects a clear image of handwritten digits/operators for reliable segmentation and OCR results

