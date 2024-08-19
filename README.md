
# SMS Spam Detection with Sentence Transformers

This project demonstrates how to classify SMS text messages as either "ham" (not spam) or "spam" using sentence embeddings and cosine similarity. The embeddings are generated using a pre-trained model from the `sentence_transformers` library.

## Project Overview

The notebook performs the following steps:

1. **Importing Necessary Libraries:**
   - `SentenceTransformer` and `util` from the `sentence_transformers` module for text embedding and similarity calculations.
   - `pandas` for handling the SMS text messages dataset.

2. **Loading the Model:**
   - The `all-MiniLM-L6-v2` model is loaded to generate embeddings for the SMS text messages.

3. **Reading the Datasets:**
   - Two datasets are used:
     - `SMS_Ham_Spam.csv`: A dataset containing SMS text messages labeled as either "ham" or "spam."
     - `unclassified_text_messages.csv`: A dataset containing SMS text messages that need to be classified.

4. **Generating Embeddings:**
   - The text messages from both datasets are converted into embeddings using the loaded model.

5. **Cosine Similarity Calculation:**
   - Cosine similarity is computed between the embeddings of the unclassified messages and the classified messages to find the most similar text messages.

6. **Ranking and Classification:**
   - The top 5 most similar classified messages are identified for each unclassified message. Based on these similarities, the unclassified messages are categorized as "ham" or "spam."

7. **Analysis:**
   - An analysis is provided, discussing the accuracy of this method in classifying the unclassified text messages, with suggestions for improving the classification results.

## Key Code Snippets

- **Importing Libraries:**
    ```python
    from sentence_transformers import SentenceTransformer, util
    import pandas as pd
    ```

- **Loading the Model and Dataset:**
    ```python
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sms_text_df = pd.read_csv("Resources/SMS_Ham_Spam.csv")
    unclassified_texts_df = pd.read_csv("Resources/unclassified_text_messages.csv")
    ```

- **Generating Embeddings:**
    ```python
    classified_message_embeddings = model.encode(classified_text_messages)
    unclassified_message_embeddings = model.encode(unclassified_text_messages)
    ```

- **Calculating Similarity:**
    ```python
    cosine_similarity_score = util.cos_sim(unclassified_message_embedding.reshape(1, -1),
                                           classified_message_embedding.reshape(1, -1))[0, 0]
    ```

- **Sorting and Ranking:**
    ```python
    classified_similarities.sort(key=lambda x: x[1], reverse=True)
    ```

## Example Output

The notebook provides output that shows the top 5 most similar classified messages for each unclassified message. Below is an example:

```text
Unclassified Message: Would your little ones like a call from Santa Xmas eve? Text Y to 9058094583 to book your time.
Top 5 Similarities:
Rank 1: Label: ham, Message: K. Did you call me just now ah? 
Similarity score: 0.3539

Rank 2: Label: ham, Message: Sorry, I'll call later in meeting.
Similarity score: 0.3094

Rank 3: Label: ham, Message: U can call me now...
Similarity score: 0.3084
...
```

## Analysis

The analysis of the results indicates that while many unclassified text messages are correctly matched with similar labeled messages, there are instances where the method might mislabel messages. It suggests that including more spam data or using a different method like LinearSVC could improve classification accuracy.

## Installation

To run this notebook, you need to have Python installed along with the necessary packages. You can install the required libraries using pip:

```bash
pip install sentence-transformers pandas
```

## Usage

Clone the repository, navigate to the project directory, and open the Jupyter notebook. Execute the cells to see the results.

```bash
git clone <repository-url>
cd <repository-folder>
jupyter notebook
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
