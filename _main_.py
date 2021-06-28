import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr
from keras.models import load_model
import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
import librosa
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model=load_model('C:\\Users\\ASUS\\aud\\best_model1.hdf5')

#prediction function
classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def predict(audio):
    audio=np.asarray(audio)
    dim_1 = np.shape(audio)[1]
    dim_2 = np.shape(audio)[2]
    channels = 1
    audio = audio.reshape((audio.shape[0], dim_1, dim_2, channels))
    prob=model.predict(audio)
    index=np.argmax(prob[0])
    return classes[index]

while True:
    # record sound(3 seconds)
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 16000
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()

    print('Recording for 3 seconds')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    p.terminate()
    print('Finished recording')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    # load prediction sound file
    os.listdir('\\')
    filepath = 'output.wav'

    # break the sound file into two digit form
    samples, sample_rate = librosa.load(filepath, sr=16000)
    ma_ord = 1000
    zr1 = np.abs(samples)

    l1, r1, l2, r2 = utils.bound(utils.filt_amp(utils.MA(utils.MA(np.asarray(zr1), ma_ord), ma_ord))) + ma_ord

    l1 = l1 - ma_ord*3
    r1 = r1 + ma_ord*3
    l2 = l2 - ma_ord*3
    r2 = r2 + ma_ord*3
    samp1 = samples[l1:r1]
    samp2 = samples[l2:r2]

    if np.size(samp1)<10:
        print("You Are not Audible(speak Again)")
        os.remove("output.wav")
        while True:
            answer = input('Run again? (y/n): ')
            if answer in ('y', 'n'):
                break
            print('Invalid input.')
        if answer == 'y':
            continue
        else:
            break
    if np.size(samp2)<10:
        print("You Are not Audible(speak Again)")
        os.remove("output.wav")
        while True:
            answer = input('Run again? (y/n): ')
            if answer in ('y', 'n'):
                break
            print('Invalid input.')
        if answer == 'y':
            continue
        else:
            break

    # extract MFCC features from both the sound samples
    #####
    mfcc1 = librosa.feature.mfcc(samp1, sr=8000)
    pad_width = 40 - mfcc1.shape[1]

    if pad_width<0:
        print("You Are not Audible(speak Again)")
        os.remove("output.wav")
        while True:
            answer = input('Run again? (y/n): ')
            if answer in ('y', 'n'):
                break
            print('Invalid input.')
        if answer == 'y':
            continue
        else:
            break

    mfcc1 = np.pad(mfcc1, pad_width=((0, 0), (0, pad_width)), mode='mean')
    sam1 = []
    sam1.append(mfcc1)
    #####
    mfcc2 = librosa.feature.mfcc(samp2, sr=8000)
    pad_width = 40 - mfcc2.shape[1]

    if pad_width < 0:
        print("You Are not Audible(speak Again)")
        os.remove("output.wav")
        while True:
            answer = input('Run again? (y/n): ')
            if answer in ('y', 'n'):
                break
            print('Invalid input.')
        if answer == 'y':
            continue
        else:
            break

    mfcc2 = np.pad(mfcc2, pad_width=((0, 0), (0, pad_width)), mode='mean')
    sam2 = []
    sam2.append(mfcc2)
    #####

    # predict both of them
    d1 = predict(sam1)
    d2 = predict(sam2)
    print(d1 + "" + d2)
    plt.axvline(x=l1, color='r')
    plt.axvline(x=r1, color='r')
    plt.axvline(x=l2, color='r')
    plt.axvline(x=r2, color='r')
    plt.text(l1 + 1000, np.max(samples) / 1.2, d1, fontsize=15, color='g')
    plt.text(l2 + 1000, np.max(samples) / 1.2, d2, fontsize=15, color='g')
    plt.plot(np.asarray(samples))

    plt.show()

    # delete the file to test again
    os.remove("output.wav")
    while True:
        answer = input('Run again? (y/n): ')
        if answer in ('y', 'n'):
            break
        print('Invalid input.')
    if answer == 'y':
        continue
    else:
        break