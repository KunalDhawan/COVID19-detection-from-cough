from pathlib import Path
from joblib import load
import math, os, sys
import numpy as np
import librosa, scipy, json, math
import soundfile as sf
import base64
from io import BytesIO
from python_speech_features import mfcc
from python_speech_features import logfbank

class ValidationModel:
    """ Given audio recording, validate if cough sample is present and is of good quality """

    def __init__(self):

        # Loading the model
        path = Path(__file__).resolve().parent/'model_validation/model.joblib'
        self.clf = load(str(path))

        #initializing the paramters
        self.frame_size = 0.030
        self.frame_shift = 0.005
        self.alpha = 0.02

    def feature_extraction(self, frame, frame_size, frame_shift, fs):

        feature_fbank = logfbank(frame, samplerate=fs, winlen=frame_size, winstep=frame_shift, nfilt=21)
        feature = np.hstack([feature_fbank])

        return feature

    def silence_removal(self, data, frame_length, alpha):
        '''
        Code: Silence removal code: This function removes
            the silence present in the given speech signal at the begin and end

        Input: data = input speec
            frame_length = frame size*fs
            alpha = percentage as (10%=0.1, 20%=0.2, ...)  for thresholding the energy envelope

        Output: output = silence removed signal

        Example: frame_size=0.30
                frame_length=int(frame_size*fs)
                data, fs = librosa.load('normalvoice.wav', sr = 8000, mono=True)
                out=silence_removal(data, frame_length, 0.02)
        '''

        signal_energy=scipy.signal.convolve(data**2, np.ones(frame_length))
        signal_energy=signal_energy[frame_length//2:-frame_length//2 + 1]
        threshold = max(signal_energy)*alpha
        for i in range(len(signal_energy)): #finding begin index
            if signal_energy[i]>threshold:
                index_begin=i
                break
            else:
                pass
            
        for i in range(len(signal_energy)-1, -1, -1): #finding end index
            if signal_energy[i]>threshold:
                index_end=i
                break
            else:
                pass    
        
        output=data[index_begin:index_end]
        
        return output  

    def predict(self, base64_string):

        data_orig, fs = sf.read(BytesIO(base64.b64decode(base64_string)))
        
        #If we have multichannel input
        if(data_orig.ndim == 2):
            data_orig = (data_orig[:,0] + data_orig[:,1])/2
        
        data_8k = librosa.resample(data_orig, fs, 8000)
        fs = 8000

        #Calculate the frame length and frame step values
        frame_length, frame_step = int(round(self.frame_size * fs)), int(round(self.frame_shift * fs))

        #Remove the leading and trailing silences  
        data=self.silence_removal(data_8k, frame_length, self.alpha)

        #Divide the data into frames
        frames = librosa.util.frame(data, frame_length, frame_step)

        #Start the frame level prediction
        prediction = []
        for i in range(frames.shape[1]):
            frame = frames[:,i]   
            feature = self.feature_extraction(frame, self.frame_size, self.frame_shift, fs)
            prediction.append(self.clf.predict(feature)[0])

        num_cough_frames = sum(np.asarray(prediction)==1)
        num_non_cough_frames = sum(np.asarray(prediction)==0)

        #Make the decision
        if ( (3*num_non_cough_frames) <= num_cough_frames):
            confidence =  num_cough_frames*100 / (num_cough_frames + num_non_cough_frames) 
            return {
            "status": 'VALID',
            "confidence": float("{:.2f}".format(confidence))
            }

        else:
            confidence =  num_non_cough_frames*100 / (num_cough_frames + num_non_cough_frames) 
            return {
            "status": 'INVALID',
            "confidence": float("{:.2f}".format(confidence))
            }
