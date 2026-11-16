
# ============================================================
# SIMPLE RECURRENT NEURAL NETWORK (RNN)
# ISEN5 - Data Science / Deep Learning
#
# Objective:
# Train an RNN to predict the next word in a sentence.
#
# Example:
# Input  : "I love machine"
# Output : "learning"
#
# This is a simple example of language modeling.
# ============================================================


# ------------------------------------------------------------
# 0. IMPORT LIBRARIES
# ------------------------------------------------------------

# Install TensorFlow before running this script in VS Code:
# Open the VS Code terminal and execute:
# pip install tensorflow

# NumPy is used for numerical operations.
# In this example, we use it to select the most probable word.
import sys
import numpy as np

# TensorFlow is the deep learning framework used to build and
# train the RNN.
import tensorflow as tf

# Tokenizer converts text into sequences of integer IDs.
from tensorflow.keras.preprocessing.text import Tokenizer

# pad_sequences makes all input sequences have the same length.
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sequential allows us to build a neural network layer by layer.
from tensorflow.keras.models import Sequential

# Embedding: transforms word IDs into dense numerical vectors.
# SimpleRNN: implements a basic recurrent neural network.
# Dense: implements a fully connected neural network layer.
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# ------------------------------------------------------------
# 1. TRAINING DATA
# ------------------------------------------------------------

# We define a very small text corpus.
# A corpus is a collection of text documents used for training.
#
# The model will learn statistical relationships between words.
# For example, it may learn that "learning" often follows
# "machine" in our training data.

sentences = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning"
]
    #Vocabulary: {'i': 1, 'learning': 2, 'love': 3,
    #             'machine': 4, 'deep': 5, 'study': 6}
    #Vocabulary size: 7
    #Il y a 6 mots distincts dans le corpus. La taille du vocabulaire est 7 parce que l'index 0 est réservé au padding.
    #Le tokenizer a attribué les IDs selon la fréquence des mots, et non selon leur ordre dans les phrases.

# ------------------------------------------------------------
# 2. TOKENIZATION
# ------------------------------------------------------------

# Create a Tokenizer.
# By default, it converts words to lowercase and assigns
# an integer ID to each unique word.
tokenizer = Tokenizer()

# Analyze the training sentences and build the vocabulary.
# The vocabulary contains all unique words in the corpus.
tokenizer.fit_on_texts(sentences)
    #Vocabulary: {'i': 1, 'learning': 2, 'love': 3, 'machine': 4, 'deep': 5, 'study': 6}
    #Vocabulary size: 7
    # Display the vocabulary.
    # Each word is associated with an integer ID.
print("Vocabulary:", tokenizer.word_index)

# Calculate the vocabulary size.
#
# len(tokenizer.word_index) gives the number of unique words.
# We add 1 because index 0 is reserved for padding.
#
# The padding token is used later to make sequences
# have the same length.
vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)


# ------------------------------------------------------------
# 3. CREATE INPUT-TARGET SEQUENCES
# ------------------------------------------------------------

# Our objective is to predict the next word.
#
# For the sentence:
# "I love machine learning"
#
# We create the following training examples:
#
# Input              Target
# I                  love
# I love             machine
# I love machine     learning
#
# Each input sequence is used to predict the next word.

# Initialize an empty list to store the training sequences.
sequences = []

# Process each sentence in the training corpus.
for sentence in sentences:

    # Convert the sentence into a list of integer IDs.
    #
    # Example:
    # "I love machine learning"
    # becomes:
    # [1, 2, 3, 4]
    #
    # texts_to_sequences() returns a list of sequences.
    # We use [0] to retrieve the first sequence.
    tokens = tokenizer.texts_to_sequences([sentence])[0]

    # Generate progressively longer sequences.
    #
    # len(tokens) gives the number of words in the sentence.
    #
    # For [1, 2, 3, 4], the loop generates:
    # [1, 2]
    # [1, 2, 3]
    # [1, 2, 3, 4]
    #
    # The first word is used as the initial context.
    for i in range(1, len(tokens)):

        # Add a new sequence to the training dataset.
        # tokens[:i+1] selects the words from the beginning
        # of the sentence up to the current position.
        sequences.append(tokens[:i+1])


# Display the generated sequences.
print("\nGenerated sequences:")
print(sequences)
#Generated sequences:
#[[1, 3], [1, 3, 4], [1, 3, 4, 2], [1, 3], [1, 3, 5], [1, 3, 5, 2], [1, 6], [1, 6, 4], [1, 6, 4, 2]]
#    I love machine learning
#    [1, 3]
#    [1, 3, 4]
#    [1, 3, 4, 2]

#   I love deep learning
#    [1, 3]
#    [1, 3, 5]
#    [1, 3, 5, 2]

#    I study machine learning
#    [1, 6]
#    [1, 6, 4]
#    [1, 6, 4, 2]

#sys.exit()

# ------------------------------------------------------------
# 4. PADDING
# ------------------------------------------------------------

# Neural networks generally require input sequences
# to have the same length.
#
# We calculate the length of the longest sequence.
max_length = max(len(seq) for seq in sequences)

# Apply padding to all sequences.
#
# maxlen=max_length:
# All sequences will have the same length.
#
# padding="pre":
# Zeros are added at the beginning of shorter sequences.
#
# Example:
# [1, 2] becomes [0, 1, 2, 3]
# (depending on the maximum sequence length)
#
# Zero is the padding ID, not a real word.
sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="pre"
)

# Display the padded sequences.
print("\nPadded sequences:")
print(sequences)
    #Padded sequences:
    #[[0 0 1 3]
    #[0 1 3 4]
    #[1 3 4 2]
    #[0 0 1 3]
    #[0 1 3 5]
    #[1 3 5 2]
    #[0 0 1 6]
    #[0 1 6 4]
    #[1 6 4 2]]
    #sys.exit()

# ------------------------------------------------------------
# 5. SEPARATE INPUTS (X) AND TARGETS (y)
# ------------------------------------------------------------

# X contains all words except the last word of each sequence.
#
# Example:
# Sequence: [0, 1, 2, 3]
# Input X : [0, 1, 2]
#
# The model will use X to predict the missing final word.
X = sequences[:, :-1]

# y contains the last word of each sequence.
#
# Example:
# Sequence: [0, 1, 2, 3]
# Target y: 3
#
# This is the correct word that the model must predict.
y = sequences[:, -1]

# Display the input and target datasets.
print("\nInputs X:")
print(X)

print("\nTargets y:")
print(y)
#sys.exit()
# Display the shapes of the datasets.
#
# X.shape:
# (number_of_examples, number_of_input_words)
#
# y.shape:
# (number_of_examples,)
#
# These shapes help us understand the structure
# of the training data.
print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)
#sys.exit()

# ------------------------------------------------------------
# 6. BUILD THE RNN MODEL
# ------------------------------------------------------------

# Sequential defines a neural network as a sequence of layers.
model = Sequential([

    # --------------------------------------------------------
    # EMBEDDING LAYER
    # --------------------------------------------------------
    #
    # Each word ID is transformed into a dense vector
    # containing 16 numerical values.
    #
    # vocab_size:
    # Number of words in the vocabulary, including padding.
    #
    # 16:
    # Dimension of the word embedding.
    #
    # Example:
    # Word ID 3 might be represented by a vector of 16 values.
    #
    # The embedding vectors are learned during training.
    Embedding(vocab_size, 16),

    # --------------------------------------------------------
    # SIMPLE RNN LAYER
    # --------------------------------------------------------
    #
    # The RNN processes the sequence word by word.
    #
    # 32:
    # Number of neurons in the hidden state.
    #
    # At each time step, the RNN:
    # 1. Receives the current word embedding.
    # 2. Combines it with the previous hidden state.
    # 3. Updates its internal representation.
    #
    # The hidden state acts as a memory of previous words.
    SimpleRNN(32),

    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------
    #
    # Dense connects the RNN hidden state to the output.
    #
    # vocab_size:
    # One output value for each possible word ID.
    #
    # softmax:
    # Converts output values into probabilities.
    #
    # The sum of all output probabilities is 1.
    #
    # The model predicts the next word from the vocabulary.
    Dense(vocab_size, activation="softmax")
])


# ------------------------------------------------------------
# 7. COMPILE THE MODEL
# ------------------------------------------------------------

# Compile specifies how the model will be trained.
model.compile(

    # Adam is an optimization algorithm.
    # It updates the model weights to reduce the loss.
    optimizer="adam",

    # Sparse categorical cross-entropy measures the difference
    # between the predicted probability distribution and
    # the correct target word.
    #
    # "Sparse" means that targets are integer IDs,
    # not one-hot encoded vectors.
    loss="sparse_categorical_crossentropy",

    # Accuracy measures how often the most probable word
    # corresponds to the correct target word.
    metrics=["accuracy"]
)


# Display the architecture of the neural network.
#
# This shows the layers, output shapes, and number
# of trainable parameters.
model.build(input_shape=(None, 3))
model.summary()

# ------------------------------------------------------------
# 8. TRAIN THE RNN
# ------------------------------------------------------------

# The model learns from the training examples.
#
# X:
# Input sequences.
#
# y:
# Correct next-word targets.
#
# epochs=200:
# The model processes the entire training dataset
# 200 times.
#
# verbose=0:
# Suppresses the training progress display.
#
# During training, TensorFlow:
# 1. Computes predictions.
# 2. Calculates the loss.
# 3. Computes gradients using backpropagation.
# 4. Updates the weights using Adam.
#
# This is a very small dataset, so the model can easily
# memorize it. The purpose is educational, not practical.

model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

# Display a message after training.
print("\nTraining completed!")


# ------------------------------------------------------------
# 9. PREDICT THE NEXT WORD
# ------------------------------------------------------------

# Define a function that takes a text sequence
# and predicts the next word.
def predict_next_word(text):

    # Convert the input text into integer IDs.
    #
    # Example:
    # "I love"
    # becomes:
    # [1, 2]
    sequence = tokenizer.texts_to_sequences([text])[0]

    # Apply padding to make the input length identical
    # to the length used during training.
    #
    # max_length - 1:
    # The input sequences contain one fewer word than
    # the complete training sequences.
    #
    # padding="pre":
    # Add zeros at the beginning if necessary.
    sequence = pad_sequences(
        [sequence],
        maxlen=max_length - 1,
        padding="pre"
    )

    # Ask the trained model to predict the next word.
    #
    # The model returns a probability for every word
    # in the vocabulary.
    #
    # verbose=0:
    # Do not display the prediction progress.
    #
    # [0]:
    # Retrieve the probability vector for our first input.
    probabilities = model.predict(
        sequence,
        verbose=0
    )[0]

    # Select the ID corresponding to the highest probability.
    #
    # argmax() returns the position of the maximum value.
    #
    # Example:
    # [0.05, 0.10, 0.15, 0.70]
    # The predicted ID is 3.
    predicted_id = np.argmax(probabilities)

    # Convert the predicted integer ID back into a word.
    #
    # tokenizer.word_index contains:
    # word -> integer ID
    #
    # We search for the word associated with predicted_id.
    for word, index in tokenizer.word_index.items():

        # If the current word has the predicted ID,
        # return that word as the prediction.
        if index == predicted_id:
            return word

    # If the predicted ID does not correspond to a word
    # in the vocabulary, return "Unknown".
    return "Unknown"


# ------------------------------------------------------------
# 10. TEST THE TRAINED MODEL
# ------------------------------------------------------------

# Example 1:
# The model receives "I love".
# It tries to predict the next word.
print("\nInput: I love")
print("Predicted next word:", predict_next_word("I love"))


# Example 2:
# The model receives "I love machine".
# It should learn to predict "learning".
print("\nInput: I love machine")
print("Predicted next word:", predict_next_word("I love machine"))


# Example 3:
# The model receives "I study machine".
# It should learn to predict "learning".
print("\nInput: I study machine")
print("Predicted next word:", predict_next_word("I study machine"))


# ============================================================
# END OF THE RNN EXAMPLE
# ============================================================
