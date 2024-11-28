
# Sentiment Analysis for Book Reviews 📚🔍

This project is designed to analyze customer reviews of books and extract their sentiment (positive, negative, or neutral) using natural language processing (NLP) techniques. By leveraging both traditional methods (like rule-based sentiment analysis) and modern machine learning models, the project aims to provide insights into customer feedback, helping businesses and authors understand their audience better.

## 🚀 Project Overview

Sentiment analysis is a critical aspect of understanding user feedback in various industries. In this project:
- We preprocess a dataset of book reviews to clean and normalize the text data.
- Perform exploratory data analysis (EDA) to uncover patterns and distributions in customer sentiment.
- Apply multiple sentiment analysis techniques:
  - Rule-based approaches using NLTK's Vader and TextBlob.
  - Transformer-based deep learning models using Hugging Face's `transformers` library.
- Evaluate the models' performance using standard metrics such as accuracy, precision, recall, and F1-score.

## 📂 Dataset

The dataset used is `Audible_English_Books.csv`, which contains:
- **Title**: The name of the book being reviewed.
- **Reviews**: Textual customer reviews about the book.
- **Rating**: A numerical or categorical rating provided by the customer.

The dataset is preprocessed to extract and clean relevant columns for analysis, such as removing unnecessary columns and handling missing or noisy data.

---

## 🛠️ Features

### 1. **Data Preprocessing**
- **Text Cleaning**:
  - Removal of punctuation, special characters, and unnecessary spaces.
  - Tokenization (breaking text into individual words or tokens).
- **Stop-word Removal**:
  - Eliminates commonly used words (like "the", "and", "is") that do not contribute significantly to sentiment.
- **Text Normalization**:
  - Lowercasing and lemmatization (reducing words to their base forms).

### 2. **Exploratory Data Analysis (EDA)**
- Generate **word clouds** to visualize the most frequent terms in positive and negative reviews.
- Analyze sentiment distributions using bar plots and histograms.
- Understand the correlation between ratings and sentiment.

### 3. **Sentiment Analysis**
- **Rule-Based Sentiment Analysis**:
  - **Vader** (Valence Aware Dictionary and sEntiment Reasoner): A pre-trained rule-based model from NLTK for quick and efficient sentiment scoring.
  - **TextBlob**: A lightweight library for sentiment polarity scoring.
- **Transformer-Based Models**:
  - Using Hugging Face's `AutoModelForSequenceClassification` for advanced sentiment classification.
  - Pre-trained models like `BERT` are fine-tuned to classify reviews into sentiment categories.

### 4. **Model Evaluation**
- Metrics for evaluation:
  - **Accuracy**: Overall correctness of the sentiment classification.
  - **Precision**: How many selected items are relevant.
  - **Recall**: How many relevant items are selected.
  - **F1-score**: The harmonic mean of precision and recall.

---

## 🔧 Installation

To replicate this project, you'll need to set up the required environment:

### Install Dependencies
1. **Install PyTorch** (for deep learning models):
   ```bash
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Install Transformers Library**:
   ```bash
   pip install transformers
   ```

3. **Install Additional Libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk textblob
   ```

---

## 💻 Usage

To run this project:
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
   ```

2. Ensure the dataset `Audible_English_Books.csv` is available in the project directory.

3. Open the Jupyter Notebook `Sentiment_analysis_book_reviews.ipynb` in your environment:
   ```bash
   jupyter notebook Sentiment_analysis_book_reviews.ipynb
   ```

4. Execute the cells in sequence to:
   - Load and preprocess the dataset.
   - Perform exploratory analysis and generate visualizations.
   - Apply and evaluate sentiment analysis models.

---

## 📊 Outputs and Insights

### Example Visualizations
- **Word Clouds**: Highlight the most frequent terms in positive and negative reviews.
- **Sentiment Distribution Plots**: Show the breakdown of sentiments (positive, neutral, negative) in the dataset.
- **Model Performance Metrics**: Evaluate the effectiveness of each sentiment analysis technique using precision, recall, and F1-score.

### Example Use Case
If you're an author or publisher, you can:
- Identify trends in customer reviews.
- Pinpoint areas of improvement based on customer feedback.
- Enhance marketing strategies by understanding audience sentiment.

---

## 🤝 Contribution Guidelines

We welcome contributions to enhance this project! You can:
- Add new sentiment analysis techniques or models.
- Improve data visualization.
- Optimize the preprocessing pipeline.

To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them:
   ```bash
   git push origin feature-name
   ```
4. Open a pull request.

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute as per the license terms.

---

## ✨ Acknowledgments

This project uses:
- **NLTK**: For traditional NLP techniques like Vader and tokenization.
- **Hugging Face Transformers**: For state-of-the-art transformer-based sentiment analysis.
- **Matplotlib and Seaborn**: For creating insightful visualizations.

