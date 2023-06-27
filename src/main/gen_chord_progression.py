import tensorflow as tf
from keras.models import load_model
import numpy as np
import pickle

model = load_model('model/model.h5')
with open('instances/ids_from_chord.bin', 'rb') as p:
    ids_from_chord = pickle.load(p)

with open('instances/chord_array.bin', 'rb') as p:
    chord_array = pickle.load(p)

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