from tensorflow import keras

def build_model(architecture, vocab_size, tag_size):
    if architecture == 'rnn':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, 64, mask_zero = True),
            keras.layers.SimpleRNN(64, return_sequences=True),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'brnn':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, 64, mask_zero = True),
            keras.layers.Bidirectional(keras.layers.SimpleRNN(64, return_sequences=True)),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'gru':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, 64, mask_zero = True),
            keras.layers.GRU(64, return_sequences=True),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'bgru':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, 64, mask_zero = True),
            keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True)),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'lstm':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, 64, mask_zero = True),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dense(tag_size),
        ])

    if architecture == 'blstm':
        return keras.Sequential([
            keras.layers.Embedding(vocab_size, 64, mask_zero = True),
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            keras.layers.Dense(tag_size),
        ])
    
    
