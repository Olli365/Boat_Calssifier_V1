import os
import tensorflow as tf 
import tensorflow_io as tfio
import librosa
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

Yes_Boat = os.path.join('Y_N_Boat', 'y', 'y_20220601_140500.wav')
No_Boat = os.path.join('Y_N_Boat', 'n', 'n_CCC_106.20220628_144100.wav')

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def load_wav_16k_mono_librosa(filename):
    # Load WAV with librosa (at original sample rate)
    y, sr = librosa.load(filename, sr=None, mono=True)
    
    # Resample the audio to 16kHz using librosa
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    # Convert to TensorFlow tensor
    wav = tf.convert_to_tensor(y_resampled, dtype=tf.float32)

    return wav


wave = load_wav_16k_mono(Yes_Boat)
nwave = load_wav_16k_mono(No_Boat)



POS = os.path.join('Y_N_Boat', 'y')
NEG = os.path.join('Y_N_Boat', 'n')


num_files = len(os.listdir(NEG))    
pos = tf.data.Dataset.list_files(POS+'/*.wav')
neg = tf.data.Dataset.list_files(NEG+'/*.wav').take(num_files)


positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)


lengths = []
for file in os.listdir(os.path.join('Y_N_Boat', 'y')):
    tensor_wave = load_wav_16k_mono_librosa(os.path.join('Y_N_Boat', 'y', file))
    lengths.append(len(tensor_wave))



tf.math.reduce_mean(lengths)

tf.math.reduce_min(lengths)

tf.math.reduce_max(lengths)


def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label



filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)



data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)



train = data.take(57)
test = data.skip(57).take(25)

samples, labels = train.as_numpy_iterator().next()

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(2491, 257, 1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


opt =Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.summary()


checkpoint_path = "checkpoint/YN.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



hist = model.fit(train, epochs=4, validation_data=test, callbacks=[callback])


os.listdir(checkpoint_dir)


PATH = os.path.join('Y_N_Boat', 'test')


test = tf.data.Dataset.list_files(PATH+'/*.wav')
data = tf.data.Dataset.zip((test, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


filepath, label = data.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)


data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)
test = data


def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav



def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([80000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram



results = {}
for file in os.listdir(os.path.join('Y_N_Boat', 'test')):
    try:
        FILEPATH = os.path.join('Y_N_Boat', 'test', file)
        
        wav = load_mp3_16k_mono(FILEPATH)
        audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=80000, sequence_stride=80000, batch_size=1)
        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(64)
        
        yhat = model.predict(audio_slices)
        
        results[file] = yhat

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        # Optionally, you can continue to the next file or handle the error as needed.
        continue




class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]




from itertools import groupby
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()


import csv
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'boat'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])