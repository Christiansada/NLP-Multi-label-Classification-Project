# NLP Multi-label Classification Project

This project aims to classify text data into multiple categories using various deep learning architectures like Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Bidirectional RNNs (Bi-RNN), Long Short-Term Memory (LSTM), Bidirectional LSTM (Bi-LSTM), Gated Recurrent Unit (GRU), and Bidirectional GRU (Bi-GRU). The dataset used for this project consists of text documents with multiple labels.

## Project Overview

### Data Preprocessing

- The dataset was loaded from a CSV file containing text data and corresponding labels.
- Preprocessing steps included removing duplicate entries, filtering out rows with all-zero labels, and removing punctuation and pure numerical values from the text.
- The dataset was split into training and testing sets.

Note: to get this data search for SemEval 2018 data on google 

### Tokenization and Padding

- Tokenization was performed using the Keras Tokenizer to convert text data into sequences of integers.
- Sequences were padded to ensure uniform length using the Keras pad_sequences function.

### Model Architectures

1. **Convolutional Neural Network (CNN)**
   - Utilized a CNN architecture with an embedding layer, convolutional layer, batch normalization, dropout, and dense layers.
   - Implemented with the Keras Sequential API.

2. **Recurrent Neural Network (RNN)**
   - Employed a simple RNN architecture with similar layers as CNN.
   - Configured with Keras Sequential API.

3. **Bidirectional RNN (Bi-RNN)**
   - Integrated bidirectional RNN layers to capture information from both forward and backward directions.

4. **Long Short-Term Memory (LSTM)**
   - Implemented LSTM architecture to handle long-term dependencies in text data.

5. **Bidirectional LSTM (Bi-LSTM)**
   - Enhanced LSTM with bidirectional processing for improved context understanding.

6. **Gated Recurrent Unit (GRU)**
   - Utilized GRU architecture as an alternative to LSTM for capturing long-term dependencies.

7. **Bidirectional GRU (Bi-GRU)**
   - Incorporated bidirectional processing into GRU architecture.

### Training and Evaluation

- Models were compiled with binary cross-entropy loss and Adam optimizer.
- Training was conducted with a specified number of epochs and early stopping to prevent overfitting.
- Models were evaluated on both training and testing datasets for accuracy and loss metrics.

## Results

- Each model's training and testing performance metrics were recorded and compared.
- Plots were generated to visualize the training and validation accuracy and loss over epochs.

## Conclusion

- The project demonstrates the effectiveness of various deep learning architectures for multi-label text classification tasks.
- Future work may involve experimenting with hyperparameters, exploring different architectures, or incorporating ensemble methods for further improvement.


## Dependencies

- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
