from tensorflow import keras

def build_recurrent_layers(architecture, num_layers, units, dropout_rate, recurrent_dropout_rate):
    layers = []

    if architecture == 'rnn':
        for _ in range(num_layers):
            layers.append(
                keras.layers.SimpleRNN(
                    units,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                )
            )

    if architecture == 'brnn':
        for _ in range(num_layers):
            layers.append(
                keras.layers.Bidirectional(keras.layers.SimpleRNN(
                    units,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                ))
            )

    if architecture == 'gru':
        for _ in range(num_layers):
            layers.append(
                keras.layers.GRU(
                    units,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                )
            )

    if architecture == 'bgru':
        for _ in range(num_layers):
            layers.append(
                keras.layers.Bidirectional(keras.layers.GRU(
                    units,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                ))
            )

    if architecture == 'lstm':
        for _ in range(num_layers):
            layers.append(
                keras.layers.LSTM(
                    units,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                )
            )

    if architecture == 'blstm':
        for _ in range(num_layers):
            layers.append(
                keras.layers.Bidirectional(keras.layers.LSTM(
                    units,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                ))
            )
    
    return layers

def build_model(
    architecture,
    embedding_units,
    num_recurrent_layers,
    recurrent_units,
    dropout_rate,
    recurrent_dropout_rate,
    vocab_size,
    tag_size
):
    recurrent_layers = build_recurrent_layers(
        architecture,
        num_recurrent_layers,
        recurrent_units,
        dropout_rate,
        recurrent_dropout_rate,
    )

    return keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
        *recurrent_layers,
        keras.layers.Dense(tag_size),
    ])