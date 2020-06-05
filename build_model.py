from tensorflow import keras

def build_recurrent_layers(
    architecture,
    num_layers,
    units,
    regularizer,
    regularization_factor,
    dropout_rate,
    recurrent_dropout_rate
):
    # construct regularizer
    if regularizer == 'l2':
        regularizer = keras.regularizers.l2(regularization_factor)
    elif regularizer == 'l1':
        regularizer = keras.regularizers.l1(regularization_factor)
    elif regularizer == 'l1_l2':
        regularizer = keras.regularizers.l1_l2(regularization_factor)
    else:
        regularizer = None

    # construct model layers
    layers = []
    if architecture == 'rnn':
        for _ in range(num_layers):
            layers.append(
                keras.layers.SimpleRNN(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
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
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
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
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
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
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
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
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
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
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
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
    regularizer,
    regularization_factor,
    dropout_rate,
    recurrent_dropout_rate,
    vocab_size,
    tag_size
):
    recurrent_layers = build_recurrent_layers(
        architecture,
        num_recurrent_layers,
        recurrent_units,
        regularizer,
        regularization_factor,
        dropout_rate,
        recurrent_dropout_rate,
    )

    return keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_units, mask_zero = True),
        *recurrent_layers,
        keras.layers.Dense(tag_size),
    ])