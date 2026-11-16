import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

sentences = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning",
    "I enjoy learning neural networks",
    "Deep learning uses neural networks",
    "I study data science",
    "Machine learning is useful",
    "Neural networks learn patterns"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

print("Vocabulary:", tokenizer.word_index)

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)

sequences = []

for sentence in sentences:
    tokens = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokens)):
        sequences.append(tokens[:i+1])

print("\nGenerated sequences:")
print(sequences)

max_length = max(len(seq) for seq in sequences)

sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="pre"
)

print("\nPadded sequences:")
print(sequences)

X = sequences[:, :-1]
y = sequences[:, -1]

print("\nInputs X:")
print(X)

print("\nTargets y:")
print(y)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

model = Sequential([
    Embedding(vocab_size, 16, mask_zero=True),
    SimpleRNN(32),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, X.shape[1]))

model.summary()

model.fit(
    X,
    y,
    epochs=300,
    verbose=0
)

print("\nTraining completed!")

def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_length - 1,
        padding="pre"
    )

    probabilities = model.predict(
        sequence,
        verbose=0
    )[0]

    predicted_id = np.argmax(probabilities)

    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            return word

    return "Unknown"

print("\nInput: I love")
print("Predicted next word:", predict_next_word("I love"))

print("\nInput: I love machine")
print("Predicted next word:", predict_next_word("I love machine"))

print("\nInput: I study machine")
print("Predicted next word:", predict_next_word("I study machine"))

print("\nInput: Deep learning uses")
print("Predicted next word:", predict_next_word("Deep learning uses"))

print("\nInput: Neural networks learn")
print("Predicted next word:", predict_next_word("Neural networks learn"))