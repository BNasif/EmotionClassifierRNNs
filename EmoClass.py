import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import string
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

#Read data from text files
trainData = pd.read_csv("train.txt",names = ['Text', 'Emotion'], sep=';')
testData = pd.read_csv("test.txt",names = ['Text', 'Emotion'], sep=';')
valData = pd.read_csv("val.txt",names = ['Text', 'Emotion'], sep=';')

# Define function to handle punctuation and convert to lowercase
def remove_punctuation_and_lowercase(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text
trainData['Text'] = trainData['Text'].apply(remove_punctuation_and_lowercase)
testData['Text'] = testData['Text'].apply(remove_punctuation_and_lowercase)
valData['Text'] = valData['Text'].apply(remove_punctuation_and_lowercase)

# Initialize tokenizer with a defined vocabulary size
tokenizer = Tokenizer(num_words=5000)
# Fit the tokenizer on the training data
tokenizer.fit_on_texts(trainData['Text'])

# Transform the text data to sequences
train_sequences = tokenizer.texts_to_sequences(trainData['Text'])
test_sequences = tokenizer.texts_to_sequences(testData['Text'])
val_sequences = tokenizer.texts_to_sequences(valData['Text'])

# Pad the sequences
train_features = pad_sequences(train_sequences, maxlen=63)
test_features = pad_sequences(test_sequences, maxlen=63)
val_features = pad_sequences(val_sequences, maxlen=63)

# Get the size of the vocabulary
vocabulary_size = len(tokenizer.word_index)

# Initialize label encoder
le = LabelEncoder()

# Fit and transform the labels
train_labels = le.fit_transform(trainData['Emotion'])
test_labels = le.transform(testData['Emotion'])
val_labels = le.transform(valData['Emotion'])

# Convert labels to categorical
train_labels = to_categorical(train_labels, num_classes=6)
test_labels = to_categorical(test_labels, num_classes=6)
val_labels = to_categorical(val_labels, num_classes=6)

glove_model = api.load("glove-wiki-gigaword-300")

def glove_word_embedding_gensim(model, tokenizer, vocabulary_size):
    max_words = vocabulary_size + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((max_words, 300))

    for word, i in word_index.items():
        if word in model:
            if i < max_words:
                embedding_matrix[i] = model[word]

    return embedding_matrix

embedding_matrix = glove_word_embedding_gensim(glove_model, tokenizer, vocabulary_size)

#Define architecture of the model
model = Sequential()
model.add(Input(shape=(train_features.shape[1], )))
model.add(Embedding(vocabulary_size + 1, 300, weights=[embedding_matrix], trainable=False))
model.add(GRU(128, recurrent_dropout=0.3, return_sequences=False, activity_regularizer=tf.keras.regularizers.L2(0.0001)))
model.add(Dense(6,activation="softmax", activity_regularizer=tf.keras.regularizers.L2(0.0001)))

# Define the learning rate
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with categorical cross entropy loss function
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print a summary of the model
print(model.summary())

history = model.fit(train_features, train_labels, epochs = 10, validation_data=(val_features, val_labels))

# Predict probabilities of classes
test_probabilities = model.predict(test_features)

# Convert probabilities into class labels
test_predictions = np.argmax(test_probabilities, axis=1)

# Convert test labels back into integer form
test_labels_integers = np.argmax(test_labels, axis=1)

# Compute accuracy, recall, and precision
accuracy = accuracy_score(test_labels_integers, test_predictions)
recall = recall_score(test_labels_integers, test_predictions, average='micro')
precision = precision_score(test_labels_integers, test_predictions, average='micro')

# Print results
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")

# Print classification report
print("\nClassification Report:")
print(classification_report(test_labels_integers, test_predictions, target_names=le.classes_))

# model.save('my_model.h5')

# model = tf.keras.models.load_model('my_model.h5')

# Some new text to classify
new_texts = ["I am really happy today!", "I feel so sad and tired.", "What a wonderful day!"]

# Preprocess the text
new_texts = [remove_punctuation_and_lowercase(text) for text in new_texts]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_features = pad_sequences(new_sequences, maxlen=63)
predictions = model.predict(new_features)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_classes)

# Print the predictions
for text, label in zip(new_texts, predicted_labels):
    print(f'The text "{text}" is predicted to express {label} emotion.')

emotion_mapping = {
    'sentimental': 'love',
    'afraid': 'fear',
    'proud': 'joy',
    'faithful': 'love',
    'terrified': 'fear',
    'joyful': 'joy',
    'angry': 'anger',
    'sad': 'sadness',
    'jealous': 'anger',
    'grateful': 'joy',
    'prepared': 'surprise',
    'embarrassed': 'sadness',
    'excited': 'joy',
    'annoyed': 'anger',
    'lonely': 'sadness',
    'ashamed': 'sadness',
    'guilty': 'sadness',
    'surprised': 'surprise',
    'nostalgic': 'love',
    'confident': 'joy',
    'furious': 'anger',
    'disappointed': 'sadness',
    'caring': 'love',
    'trusting': 'love',
    'disgusted': 'anger',
    'anticipating': 'surprise',
    'anxious': 'fear',
    'hopeful': 'joy',
    'content': 'joy',
    'impressed': 'surprise',
    'apprehensive': 'fear',
    'devastated': 'sadness'
}

# Load the data into pandas dataframe
data = pd.read_csv('train.csv', error_bad_lines=False)  # Skip bad lines
data['context'] = data['context'].map(emotion_mapping)
unique_context_values = data['context'].unique()
data['utterance'] = data['utterance'].str.replace('_comma_', '')
# Remove punctuation
data['utterance'] = data['utterance'].str.replace('[^\w\s]', '')
# Make sure all text is lower case
data['utterance'] = data['utterance'].apply(lambda x: x.lower())
# Tokenization
data['utterance'] = data['utterance'].apply(nltk.word_tokenize)
# Recreate text from tokens
data['utterance'] = data['utterance'].apply(lambda x: ' '.join(x))

# Initialize the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['utterance'])


# Function to get the most similar response
def get_response(user_input, emotion):
    query_vec = vectorizer.transform([user_input]).toarray()

    # Select utterances that match the given emotion
    emotion_data = data[data['context'] == emotion]
    # Transform utterances into vectors
    emotion_data_vec = vectorizer.transform(emotion_data['utterance']).toarray()
    csim = cosine_similarity(emotion_data_vec, query_vec).reshape((-1,))

    # Get the index of the most similar utterance
    idx = csim.argmax()
    return emotion_data.iloc[idx + 1]['utterance']  # Return the next utterance in the dialogue

#Function to classify emotion in a single sentence
def classify_emotion(text):
    # Preprocess the text
    text = remove_punctuation_and_lowercase(text)
    new_sequences = tokenizer.texts_to_sequences([text])
    new_features = pad_sequences(new_sequences, maxlen=63)
    predictions = model.predict(new_features)  # Predict emotions in the text

    # Get the class with the highest probability for each sample
    predicted_classes = np.argmax(predictions, axis=1)

    # Convert the predicted classes to emotion labels
    predicted_labels = le.inverse_transform(predicted_classes)

    return predicted_labels[0]
# Testing the system
user_input = "I lost my pen today"
emotion = classify_emotion(user_input)
print(emotion)
print(get_response(user_input, emotion))
