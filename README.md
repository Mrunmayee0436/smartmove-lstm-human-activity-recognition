# SmartMove — Human Activity Recognition Using LSTM

## Project Overview
SmartMove is a deep learning project that uses a Long Short-Term Memory (LSTM) neural network to recognize human activities from smartphone sensor data.

## Activities Recognized
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Technologies Used
- Python
- Google Colab
- TensorFlow / Keras
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn

## Dataset
The project uses the UCI Human Activity Recognition Using Smartphones dataset.

Dataset: https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecogni-tion%2Busing%2Bsmartphones

## Methodology
1. Load and explore the sensor data.
2. Prepare the sensor sequences.
3. Build and train an LSTM model.
4. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Predict activities from test samples.

## How to Run
1. Open the Jupyter notebook in Google Colab.
2. Run the cells from top to bottom.
3. Download the dataset when prompted by the notebook.
4. Train the model and review the evaluation results.

## Project Status
Academic learning project. Model performance should be measured by running the notebook.

## Acknowledgements
This project uses the UCI Human Activity Recognition Using Smartphones dataset. Please refer to the dataset documentation and any referenced code repositories.

## Installation and Setup

### Requirements
- Python 3
- Google Colab or Jupyter Notebook
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### How to Run the Project
1. Open `SmartMove_LSTM_HAR.ipynb` in Google Colab.
2. Run the notebook cells in order.
3. Download and load the UCI Human Activity Recognition dataset using the notebook.
4. Explore the sensor readings and prepare the data.
5. Train the LSTM model.
6. Evaluate the model on the test dataset.
7. View the prediction scores and confusion matrix.

## Model Architecture

The project uses an LSTM layer to learn patterns in sequential sensor measurements. A dropout layer helps reduce overfitting, followed by a dense layer and a six-class softmax output layer.

## Evaluation Metrics

The model is evaluated using:
- Test accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual evaluation results are generated when the notebook is executed.

## Project Structure

- `README.md` — Project documentation
- `SmartMove_LSTM_HAR.ipynb` — Implementation and experiments
- `requirements.txt` — Python dependencies
- `results/` — Evaluation graphs and output figures
- `PROJECT_REPORT.md` — Project methodology and conclusion

## Limitations

The current version classifies recorded smartphone sensor sequences. It does not directly recognize live activities from a phone or computer camera.

## Dataset Reference

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecogni-tion%2Busing%2Bsmartphones

## Future Enhancements

- Interactive web dashboard
- Comparison of LSTM with other machine learning models
- Live smartphone sensor integration
- Exportable prediction reports
