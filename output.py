import csv
import os
import librosa
import tensorflow as tf
import tensorflow_io as tfio
import h5py
from itertools import groupby
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam


# Define your model architecture and any custom layers/objects here
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(2491, 257, 1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt =Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# Load model from a saved checkpoint
checkpoint_path = "Saved checkpoints\96%"
model.load_weights(checkpoint_path)

def load_mp3_16k_mono(filename):
    # Load MP3 with librosa (at original sample rate)
    y, sr = librosa.load(filename, sr=None, mono=True)

    # Convert to TensorFlow tensor
    wav = tf.convert_to_tensor(y, dtype=tf.float32)

    # Resample to 16kHz using TensorFlow
    wav = tfio.audio.resample(wav, sr, 16000)

    return wav

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

audio_folder = "Y_N_Boat\y"
results = {}
for file in os.listdir(audio_folder):
    if file.endswith(".wav"):  # Assuming the audio files are in wav format
        FILEPATH = os.path.abspath(os.path.join(audio_folder, file))

        try:
            wav = load_mp3_16k_mono(FILEPATH)

            # Dynamically set the sequence_length and sequence_stride
            sequence_length = min(80000, len(wav) // 2)
            sequence_stride = sequence_length // 2

            if sequence_length < 1 or sequence_stride < 1:
                print(f"Skipping file {file} as it's too short to be processed.")
                continue

            audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                wav, 
                wav, 
                sequence_length=sequence_length, 
                sequence_stride=sequence_stride, 
                batch_size=1
            )
            audio_slices = audio_slices.map(preprocess_mp3)
            audio_slices = audio_slices.batch(64)

            yhat = model.predict(audio_slices)
            results[file] = yhat

        except ValueError as e:
            print(f"Error processing file {file}: {e}")

class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds

postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
