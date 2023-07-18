# -*- coding: utf-8 -*-
"""DpTC1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aZJXny0HIMMU8fzU9wmp8QVpoCq1avLC

train and show result
"""

import tensorflow as tf
from keras.models import load_model
import numpy as np
import pickle

chordfile = 'data/all_progressions'
with open(chordfile) as file:
  chord_progression = file.read()
chord_progression = chord_progression.replace('\n', ' \n')
print(chord_progression)

chord = chord_progression.split(' ')
chord = list(map(lambda x:x.replace('\n', ''), chord))
chord_set = set(chord)
sorted_chord = sorted(chord_set)
print(chord_set)

# mapping table between id and chord
ids_from_chord = {ch:i for i, ch in enumerate(sorted_chord)}
chord_array = np.array(sorted_chord)
chord_from_ids = np.array([ids_from_chord[ch] for ch in sorted_chord], dtype=np.int32)

print(ids_from_chord)
print()
print(chord_from_ids)
print()
print(type(chord_array))
print(chord_array)
print(chord[:10], 'Encoding->', chord_from_ids[:10])
print(chord_from_ids[15:21], 'dedoding->', chord_array[15:21])

chord_dataset = tf.data.Dataset.from_tensor_slices(chord_from_ids)

print(chord_dataset)
for ex in chord_dataset.take(5):
  print(ex)
  print('{} -> {}'.format(ex.numpy(), chord_array[ex.numpy()]))

seq_length = 10
chunk_size = seq_length+1
chord_chunks = chord_dataset.batch(chunk_size, drop_remainder=True)

for seq in chord_chunks.take(1):
  input_seq = seq[:seq_length].numpy()
  target = seq[seq_length].numpy()
  print(type(input_seq))
  print(input_seq)
  print(target)
  print(input_seq, '->', target)
  print(repr(chord_array[input_seq]),
        '->',
        repr(chord_array[target]))

def split_input_target(chunk):
  input_seq = chunk[:-1]
  target_seq = chunk[1:]
  return input_seq, target_seq
dataset_sequences = chord_chunks.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
tf.random.set_seed(1)
dataset = dataset_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_model(vocab_size, embedding_dim, rnn_units):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim),
      tf.keras.layers.LSTM(
          rnn_units, return_sequences=True),
          tf.keras.layers.Dense(vocab_size)])
  return model

charset_size = len(chord_array)
embedding_dim = 256
rnn_units = 512
tf.random.set_seed(1)

model = build_model(
    vocab_size = charset_size,
    embedding_dim = embedding_dim,
    rnn_units = rnn_units)

model.summary

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ))

model.fit(dataset, epochs=100)
model.save('model/model.h5')
with open('instances/ids_from_chord.bin', 'wb') as p:
    pickle.dump(ids_from_chord, p)

with open('instances/chord_array.bin', 'wb') as p:
    pickle.dump(chord_array, p)

def predict(model, starting_str, len_generated_text=19, max_input_length=3, scale_factor=1.0):
  encoded_input = [ids_from_chord[s]for s in starting_str]
  encoded_input = tf.reshape(encoded_input, (1, -1))
  generated_str = starting_str
  model.reset_states()
  generated_chord = []

  for i in range(len_generated_text):
    logits = model(encoded_input)
    logits = tf.squeeze(logits, 0)
    scaled_logits = logits * scale_factor
    new_char_indx = tf.random.categorical(scaled_logits, num_samples=1)
    new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()
    generated_chord.append(chord_array[new_char_indx])

    new_char_indx = tf.expand_dims([new_char_indx], 0)

    encoded_input = tf.concat([encoded_input, new_char_indx], axis=1)
    encoded_input = encoded_input[:,-max_input_length:]
  return generated_chord

tf.random.set_seed(1)

starting_str=['C', 'F'] #ここを色々変える
# number of input
for max_input_length in range(3):
  # change scale factor on each iterate
  for scale_factor in range(5):
    print("max_input_length: {}, scale_factor:{}".format(max_input_length+4, scale_factor*0.5))
    print(predict(model, starting_str=starting_str,
                 len_generated_text=10,
                 max_input_length=max_input_length+2,
                 scale_factor = scale_factor*0.5))
    print()