# SmartMove: Human Activity Recognition Using LSTM

## 1. Introduction
Human Activity Recognition (HAR) is the process of identifying physical activities using sensor data. This project uses deep learning to classify activities from smartphone motion measurements.

## 2. Problem Statement
To develop a deep learning model that recognizes human activities from sequential accelerometer and gyroscope data.

## 3. Objectives
- Understand sequential sensor data.
- Implement an LSTM neural network.
- Classify six human activities.
- Evaluate model performance using standard classification metrics.
- Build a reproducible deep learning notebook.

## 4. Dataset
The project uses the UCI Human Activity Recognition Using Smartphones dataset. It contains labeled sensor measurements collected from volunteers performing different activities.

## 5. Methodology
1. Download the dataset.
2. Load the training and testing sensor sequences.
3. Explore the data through visualizations.
4. Build an LSTM-based classification model.
5. Train the model using labeled training data.
6. Evaluate predictions on the official test set.
7. Visualize the confusion matrix and classification report.

## 6. Model Architecture
- Input: 128 time steps and 9 sensor features.
- LSTM: 64 units.
- Dropout: 30%.
- Dense hidden layer: 32 units with ReLU activation.
- Output: 6 units with softmax activation.

## 7. Tools and Technologies
Python, Google Colab, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## 8. Evaluation
The model is evaluated using test accuracy, precision, recall, F1-score, and a confusion matrix. Actual numerical results should be inserted after running the notebook.

## 9. Conclusion
This project demonstrates how an LSTM neural network can learn temporal patterns in smartphone sensor data to classify human activities.

## 10. Future Scope
Future improvements may include a web dashboard, comparison with other models, and integration with live smartphone sensor readings.

## References
UCI Machine Learning Repository — Human Activity Recognition Using Smartphones:
https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecogni-tion%2Busing%2Bsmartphones
