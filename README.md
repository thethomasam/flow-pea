# Automated Plant Analysis Workflow

This Markdown document provides an overview of the automated plant analysis workflow implemented in the provided Python script.

## Overview

The provided Python script automates the process of plant analysis using a combination of image classification models. The workflow encompasses the following steps:

1. **Data Acquisition and Preprocessing:**
    - Raw drone-acquired images are processed to generate a composite image of the entire agricultural field.
    - The composite image undergoes manual annotation using GIMP, with distinct red dots marking plant locations.

2. **Model Training:**
    - Transfer learning is employed to train deep learning models efficiently.
    - Pretrained models such as ResNet50 and DeiT are utilized due to their powerful feature extraction capabilities.
    - The training process involves parameter tuning, optimization algorithm selection, and loss function formulation.

3. **Plot Labelling:**
    - An intuitive Graphical User Interface (GUI) facilitates efficient plot labelling.
    - Researchers can label plots as positive or negative examples, enhancing the dataset's diversity.
    - Positive examples are marked with visible plant growth, while negative examples enhance model robustness.

4. **Field Deployment and Analysis:**
    - The agricultural field is partitioned into subplots using a defined grid pattern.
    - Trained models analyze each subplot to identify plant presence and count.
    - Plant count predictions are consolidated to provide insights into plant distribution and density.

5. **Orchestrator Module:**
    - The orchestrator script streamlines the entire workflow for researchers.
    - Researchers can focus on input drone images, as the script automates processing, annotation, training, and analysis.
    - Orchestrator empowers efficient agricultural analysis, saving time and effort.

## Usage

The script offers various options to run the training, analysis, and labelling steps interactively. Researchers can choose to use the transformer model or ResNet for training based on their specific requirements.

## Conclusion

This automated plant analysis workflow showcases the seamless integration of data acquisition, model training, analysis, and insights generation. By combining deep learning models with efficient tools, the script addresses the challenges of plant disease identification and contributes to advancing agricultural practices.
