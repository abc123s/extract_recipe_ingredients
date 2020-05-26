from tensorflow import keras

def build_model(
    architecture,
    embedding_units,
    recurrent_units,
    vocab_size,
    tag_size
):
    if architecture == 'rnn':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
            keras.layers.SimpleRNN(recurrent_units, return_sequences=True),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'brnn':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
            keras.layers.Bidirectional(keras.layers.SimpleRNN(recurrent_units, return_sequences=True)),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'gru':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
            keras.layers.GRU(recurrent_units, return_sequences=True),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'bgru':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
            keras.layers.Bidirectional(keras.layers.GRU(recurrent_units, return_sequences=True)),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'lstm':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
            keras.layers.LSTM(recurrent_units, return_sequences=True),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'blstm':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
            keras.layers.Bidirectional(keras.layers.LSTM(recurrent_units, return_sequences=True)),
            keras.layers.Dense(tag_size),
        ])
    
    
