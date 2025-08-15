# Next Word Predictor with LSTM

This project implements a Next Word Prediction model using Long Short-Term Memory (LSTM) networks, trained on the [Sherlock Holmes Next-Word Prediction Corpus](https://www.kaggle.com/datasets/muhammadbilalhaneef/sherlock-holmes-next-word-prediction-corpus). The model predicts the next word in a sequence of words, leveraging the sequential nature of language.

## 🧠 Model Architecture

* **Embedding Layer**: Converts input words into dense vectors of fixed size.
* **LSTM Layer**: Captures long-range dependencies in the text.
* **Dense Layer**: Outputs a probability distribution over the vocabulary.
* **Softmax Activation**: Selects the word with the highest probability.

## 📊 Dataset

The dataset comprises unaltered text from Sherlock Holmes stories, formatted for next-word prediction tasks. It is available on Kaggle: ([Kaggle][1]).

## ⚙️ Requirements

* Python 3.x
* TensorFlow
* NumPy
* Pandas
* Matplotlib (for visualizations)

## 🚀 Setup & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/next-word-predictor.git
   cd next-word-predictor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data/` directory.

4. Preprocess the data:

   ```bash
   python preprocess.py
   ```

5. Train the model:

   ```bash
   python train.py
   ```

6. Use the model for predictions:

   ```bash
   python predict.py "Your input text here"
   ```

## 📈 Results

The model achieves an accuracy of approximately 72% on the validation set, demonstrating its ability to predict the next word in a sequence.

## 🔄 Future Improvements

* Implementing a Bidirectional LSTM to capture context from both directions.
* Incorporating attention mechanisms to focus on relevant parts of the input sequence.
* Fine-tuning hyperparameters for improved performance.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
