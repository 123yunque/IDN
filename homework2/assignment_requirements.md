# Thermal Image Classification Assignment

## Overview
In this assignment, you will implement a deep learning model for binary classification of thermal images. The task is to classify thermal images into two categories:
- **ICAS (Intracranial Aneurysm Screening)**: Images showing potential aneurysm cases
- **Non-ICAS**: Normal thermal images without aneurysm indicators

## Dataset Information
- **Dataset Path**: `dataset/datasets/thermal_classification_cropped/`
- **Classes**: 
  - `icas/` - 303 positive samples (ICAS cases)
  - `non_icas/` - 647 negative samples (Non-ICAS cases)
- **Image Format**: 512×512 RGB images (.jpg)
- **Total Images**: 950 images

## Assignment Objectives
1. **Data Loading & Preprocessing**: Implement proper data loading with train/validation/test splits
2. **Model Architecture**: Design and implement a CNN architecture for binary classification
3. **Training Pipeline**: Implement training loop with appropriate loss function and optimizer
4. **Evaluation**: Implement comprehensive evaluation metrics and visualization

## Requirements

### Part 1: Data Loading and Preprocessing (25 points)
- Implement data loading using PyTorch DataLoader
- Apply appropriate data augmentation techniques
- Split data into train (70%), validation (15%), and test (15%) sets
- Normalize images appropriately
- Handle class imbalance (647 vs 303 samples)

### Part 2: Model Architecture (25 points)
- Design a CNN architecture suitable for this classification task
- You may use:
  - Custom CNN from scratch
  - Pre-trained models (ResNet, EfficientNet, etc.) with fine-tuning
  - Transfer learning approaches
- Justify your architectural choices

### Part 3: Training and Optimization (25 points)
- Implement appropriate loss function for binary classification
- Choose and configure an optimizer
- Implement learning rate scheduling if needed
- Add regularization techniques (dropout, weight decay, etc.)
- Monitor training progress with validation metrics

### Part 4: Evaluation and Analysis (25 points)
- Implement comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC curve and AUC
  - Confusion matrix
- Visualize training progress (loss curves, accuracy curves)
- Analyze model performance and discuss results
- Provide insights about the model's strengths and limitations

## Deliverables
1. **Code**: Complete implementation with clear comments
2. **Report**: 2-3 page report including:
   - Model architecture description and justification
   - Training strategy and hyperparameter choices
   - Results analysis and interpretation
   - Discussion of challenges and potential improvements
3. **Visualizations**: Training curves, confusion matrix, ROC curve, sample predictions

## Evaluation Criteria
- **Code Quality** (20%): Clean, well-commented, reproducible code
- **Model Performance** (30%): Achieved accuracy and other metrics
- **Technical Implementation** (25%): Proper use of deep learning techniques
- **Analysis and Insights** (25%): Quality of results interpretation and discussion

## Submission Guidelines
- Submit all code files (.py or .ipynb)
- Include a requirements.txt file with dependencies
- Provide clear instructions to reproduce your results
- Submit your written report in PDF format

## Tips for Success
1. Start with a simple baseline model and gradually improve
2. Pay attention to data preprocessing and augmentation
3. Monitor for overfitting and use appropriate regularization
4. Experiment with different architectures and hyperparameters
5. Validate your results thoroughly before final submission

## Deadline
**Due Date**: [Insert your deadline here]

Good luck with your assignment!
