import tensorflow as tf

import numpy as np
import sys
import os

import converter

# HYPERPARAMS
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
BUFFER_SIZE = 100
EMBEDDING_LAYER_SIZE = 256
RNN_LAYER_SIZE = 512
EPOCHS = 80
def OBJECTIVE(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

beats_to_generate = 128

file = open('combined_out.csv', 'r')
raw_data = file.read()
file.close()

# 0 represents holding a note, 1 represents a rest
data = np.genfromtxt('combined_out.csv', delimiter=',').astype(int)

# First 32 sequences of 8 notes withheld, approx 10% of the data
holdoutSize = 40
training_data = data[holdoutSize:]

dataset = tf.data.Dataset.from_tensor_slices(training_data)

sequences = dataset.batch(SEQUENCE_LENGTH + 1, drop_remainder=True)


def split_input_target(chunk):
    input = chunk[:-1]
    target = chunk[1:]
    return input, target


dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

feature_value_range = training_data.max() + 1


def build_model(range, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(range, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(range)])
    return model

if __name__ == '__main__':
    model = None
    #if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == 'train'):
    model = build_model(feature_value_range, EMBEDDING_LAYER_SIZE, RNN_LAYER_SIZE, BATCH_SIZE)
    model.summary()
    model.compile(optimizer='adam', loss=OBJECTIVE)

    checkpoint_dir = './checkpoint'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    #if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == 'predict'):
    # Rebuild model for a new batch size

    model = build_model(range=feature_value_range, embedding_dim=EMBEDDING_LAYER_SIZE, rnn_units=RNN_LAYER_SIZE,
                        batch_size=1)
    print(tf.train.latest_checkpoint(checkpoint_dir))
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))


    if len(sys.argv) > 2:
        beats_to_generate = int(sys.argv[2])
    if model is None:
        print("Error: model has not been initialized.")
        exit(1)


    initial_sequence = training_data[:SEQUENCE_LENGTH]
    initial_sequence = tf.expand_dims(initial_sequence, 0)
    sequence_generated = []

    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(beats_to_generate):
        predictions = model(initial_sequence)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        initial_sequence = tf.expand_dims([predicted_id], 0)

        sequence_generated.append(predicted_id)

    transcription = ''
    for t in sequence_generated:
        transcription += str(t) + '\n'

    output_file = open('out.csv', 'w')
    output_file.write(transcription)
    output_file.close()

    converter.convert_to_midi(transcription_path='out.csv')

    # Testing

    performance_csv = 'prediction,actual\n'
    successCount = 0
    #For every sequence of SEQUENCE_LENGTH in the holdout data:
    for i in range(0, holdoutSize - SEQUENCE_LENGTH):
        testSequence = training_data[i:(i+SEQUENCE_LENGTH)]
        testSequence = tf.expand_dims(testSequence, 0)
        prediction = model(testSequence)

        prediction = tf.squeeze(prediction, 0)

        predicted_pitch = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()
        actual_pitch = training_data[i + SEQUENCE_LENGTH]
        performance_csv += str(predicted_id) + ','
        performance_csv += str(actual_pitch) + '\n'
        if predicted_pitch == actual_pitch:
            successCount += 1

    performance_file = open('performance.csv', 'w')
    performance_file.write(performance_csv)
    performance_file.close()
    print(successCount / (holdoutSize - SEQUENCE_LENGTH))




