import numpy as np
import math
import base64

from sklearn.svm import SVC
from statistics import mode
from sklearn.metrics import accuracy_score
from scipy.signal import correlate
from joblib import dump, load
from io import BytesIO
from pathlib import Path
import soundfile as sf
import librosa
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


class RIAModel:
    """Given an cough audio sample, assess symptomatic risk of having a respiratory illness"""    
    def __init__(self):

        # Defining the window paramters
        self.frame_size = 0.020     #20 ms
        self.frame_shift = 0.010    #10 ms

        # Loading the model
        path = Path(__file__).resolve().parent/'model/model.joblib'
        print(str(path))
        self.clf = load(str(path))
        

    def predict(self, base64_string):

        data, fs = sf.read(BytesIO(base64.b64decode(base64_string)))

        #If we have multichannel input
        if(data.ndim == 2):
            data = (data[:,0] + data[:,1])/2

        # Convert from seconds to samples
        frame_length, frame_step = self.frame_size * fs, self.frame_shift * fs 
        
        signal_length = len(data)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        
        # Make sure that we have at least 1 frame
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        
        # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        pad_signal = np.append(data, z) 

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        
        prediction = []
        for i in range(len(indices)):
            feature_input = []
            frame = data[indices[i][0]:indices[i][frame_length-1]]
            feature_mfcc = mfcc(frame, samplerate=fs, winlen=self.frame_size, winstep=self.frame_shift, numcep=13)
            feature_fbank = logfbank(frame, samplerate=fs, winlen=self.frame_size, winstep=self.frame_shift, nfilt=13)
            feature_init = np.concatenate((feature_mfcc,feature_fbank), axis=1)
            feature_energy = math.sqrt(np.mean(frame*frame))
            feature = np.append(feature_init,feature_energy)
            feature = feature.tolist()
            feature_input.append(feature)
            prediction.append(self.clf.predict(feature_input)[0])

        detected_label = mode(prediction)
        frames_with_detected_label = prediction.count(detected_label)
        confidence = round(float( frames_with_detected_label/len(prediction)), 2)

        if(detected_label == 1):
            return {
            "prediction": 'COVID',
            "confidence": confidence
            }
    
        else:
            return {
            "prediction": 'NORMAL',
            "confidence": confidence
            }
    
