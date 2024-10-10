import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augemented_signal = signal + noise *  noise_factor
    return augemented_signal

def time_stretch(signal, stretch_rate):
    # Stretch the signal
    stretched_signal = librosa.effects.time_stretch(signal, rate=stretch_rate)
    
    original_length = len(signal)
    stretched_length = len(stretched_signal)
    
    if stretched_length > original_length:
        # If the stretched signal is longer, clip it to the original length
        return stretched_signal[:original_length]
    elif stretched_length < original_length:
        # If the stretched signal is shorter, pad it to the original length
        padding = original_length - stretched_length
        return np.pad(stretched_signal, (0, padding), 'constant', constant_values=(0, 0))
    else:
        # If the lengths are equal, return the stretched signal as is
        return stretched_signal
    
def pitch_scale(signal, sr, num_semimtones):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=num_semimtones)

def brightness(signal, max_delta):
    return tf.image.random_brightness(signal, max_delta)

def random_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = np.random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor





def load_wav_16k_mono(filename):

    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

"""
def augment(filename):
    wav, sr = librosa.load(filename)
    choice = tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32)
    if choice == 0:
        aug_wav = add_white_noise(wav, 0.3)
    elif choice == 1:
        aug_wav = time_stretch(wav, 1.3)
    elif choice == 2:
        aug_wav = pitch_scale(wav, sr, 5)
    elif choice == 3:
        aug_wav = random_gain(wav, 1, 2)
    aug_wav = librosa.resample(aug_wav, orig_sr=sr, target_sr=16000 )
    aug_wav = tf.convert_to_tensor(aug_wav)
    return aug_wav
        


def preprocess(file_path, label):
    
    choice = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    if choice == 0:
        wav = tf.py_function(augment(file_path), [tf.float32, tf.int32])
    else:
        wav = load_wav_16k_mono(file_path)
        
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    # Normalization
    mean = tf.math.reduce_mean(spectrogram)
    std = tf.math.reduce_std(spectrogram)
    spectrogram = (spectrogram - mean) / std
    choice = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    if choice != 0:
        choice2 = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        if choice2 == 0:
            spectrogram = brightness(spectrogram, 0.05)

    return spectrogram, label
"""
def augment(filename):
    def _augment_py(filename):
        # Decode filename if it's encoded as bytes in a numpy array
        if isinstance(filename, np.ndarray) and filename.dtype.type is np.bytes_:
            filename = filename.item().decode('utf-8')  # Convert numpy bytes to string
        elif isinstance(filename, np.ndarray):
            filename = filename.item()  # Assuming it's already a string within a numpy object array
        
        wav, sr = librosa.load(filename)
        choice = np.random.randint(0, 4)
        if choice == 0:
            aug_wav = add_white_noise(wav, 0.3)
        elif choice == 1:
            aug_wav = time_stretch(wav, 1.3)
        elif choice == 2:
            aug_wav = pitch_scale(wav, sr, 5)
        elif choice == 3:
            aug_wav = random_gain(wav, 1, 2)
        aug_wav = librosa.resample(aug_wav, orig_sr=sr, target_sr=16000)
        return aug_wav.astype(np.float32)
    
    # Use tf.numpy_function with the correct signature - it expects the function, inputs, and output types
    aug_wav = tf.numpy_function(_augment_py, [filename], tf.float32)
    aug_wav.set_shape([None])  # Adjust this shape based on your data
    return aug_wav


def preprocess(file_path, label):
    def _preprocess_py(file_path, label):
        choice = np.random.randint(0, 4)
        if choice == 0:
            wav = augment(file_path)
        else:
            wav = load_wav_16k_mono(file_path)
        wav = wav[:48000]
        zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav], 0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        # Normalization
        mean = tf.math.reduce_mean(spectrogram)
        std = tf.math.reduce_std(spectrogram)
        spectrogram = (spectrogram - mean) / std
        return spectrogram, label
    return tf.numpy_function(_preprocess_py, [file_path, label], [tf.float32, tf.float32])



POS = os.path.join('Y_N_Boat', 'y')
NEG = os.path.join('Y_N_Boat', 'n')

num_files = 250
pos = tf.data.Dataset.list_files(POS + '/*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*.wav').take(num_files)

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

#data = data.map(preprocess)
data = data.map(lambda x, y: preprocess(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(32)
test = data.skip(32).take(14)
samples, labels = train.as_numpy_iterator().next()


model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.summary()

checkpoint_path = "checkpoint/YN.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

hist = model.fit(train, epochs=5, validation_data=test, callbacks=[callback])

print(os.listdir(checkpoint_dir))
