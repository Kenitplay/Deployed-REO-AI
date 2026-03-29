import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model once
model = tf.keras.models.load_model('research_rnn_model.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

max_len = 20

def predict_title(title):
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred)

    confidence = float(pred[0][idx] * 100)
    result = le.inverse_transform([idx])[0]

    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "title": title
    }
