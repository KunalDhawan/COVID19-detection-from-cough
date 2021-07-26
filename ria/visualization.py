import sys, librosa, numpy, scipy, copy, math
import base64
from io import BytesIO
import soundfile as sf
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class OutputVisualiztion:
    """ Plot a figure to provide visualization for the given signal with some features """

    def interpolateLinear(
        self,
        y1, #
        y2, #
        x # weighting [0..1]. 0 would be 100 % y1, 1 would be 100 % y2
    ):
        '''
        simple linear interpolation between two variables
        Parameters
        y1	
        y2	
        x	weighting [0..1]: 0 would be 100 % y1, 1 would be 100 % y2
        
        Returns
        the interpolated value
        '''
        return y1 * (1.0 - x) + y2 * x

    def interpolateParabolic(
        self,
        alpha, 
        beta, 
        gamma, 
        x # relative position of read offset [-1..1]
    ):
        '''
        parabolic interpolation between three equally spaced values
        Parameters
        alpha	first value
        beta	second value
        gamma	third value
        x	relative position of read offset [-1..1]
        
        Returns
        the interpolated value
        '''

        if (x == 0): return beta
        
        #we want all numbers above zero ...
        offset = alpha
        if (beta < offset): offset = beta
        if (gamma < offset): offset = gamma
        offset = math.fabs(offset) + 1
        
        alpha += offset
        beta += offset
        gamma += offset
        
        a = b = c = 0
        a = (alpha - 2.0 * beta + gamma) / 2.0
        if (a == 0):
            if (x > 1):
                return self.interpolateLinear(beta, gamma, x) - offset
            else:
                return self.interpolateLinear(alpha, beta, x + 1) - offset
        else:
            c = (alpha - gamma) / (4.0 * alpha)
            b = beta - a * c * c
            return (a * (x - c) * (x - c) + b) - offset
    
    def findArrayMaximum(
        self,
        data, 
        offsetLeft = 0, 
        offsetRight = -1, # if -1, the array size will be used
        doInterpolate = True, # increase accuracy by performing a 
                              # parabolic interpolation
    ):
        '''
        Parameters
        data	a numpy array
        offsetLeft	the index position at which analysis will commence
        offsetRight	the terminating index position. if -1, the array size will be used
        doInterpolate	if True: increase accuracy by performing a parabolic interpolation within the results
        
        Returns
        a list containing the index and the value of the maximum
        '''

        
        objType = type(data).__name__.strip()
        if objType != "ndarray":
            raise Exception('data argument is no instance of numpy.array')
        size = len(data)
        if (size < 1):
            raise Exception('data array is empty')
        xOfMax = -1
        valMax = min(data)
        if offsetRight == -1:
            offsetRight = size
        for i in range(offsetLeft + 1, offsetRight - 1):
            if data[i] >= data[i-1] and data[i] >= data[i + 1]:
                if data[i] > valMax:
                    valMax = data[i]
                    xOfMax = i
        if doInterpolate:
            if xOfMax > 0 and xOfMax < size - 1:
                # use parabolic interpolation to increase accuracty of result
                alpha = data[xOfMax - 1]
                beta = data[xOfMax]
                gamma = data[xOfMax + 1]
                xTmp = (alpha - gamma) / (alpha - beta * 2 + gamma) / 2.0
                xOfMax = xTmp + xOfMax
                valMax = self.interpolateParabolic(alpha, beta, gamma, xTmp)
        if xOfMax == -1:
            raise Exception("no maximum found")
        return [xOfMax, valMax]

    def createLookupTable(self, size, type = 3):
        '''
        creates a lookup table covering the range of [0..1]
        Parameters
        size	number of data values that are distributed over the range [0..1] the type of the lookup table. To date these types are supported:
        LOOKUP_TABLE_NONE: a rectangular window
        LOOKUP_TABLE_SINE: a sine function
        LOOKUP_TABLE_COSINE: a cosine function
        LOOKUP_TABLE_HAMMING: a Hamming window
        LOOKUP_TABLE_HANN: a Hann window
        '''

        LOOKUP_TABLE_NONE = 0
        LOOKUP_TABLE_SINE = 1
        LOOKUP_TABLE_COSINE = 2
        LOOKUP_TABLE_HAMMING = 3
        LOOKUP_TABLE_HANN = 4


        data = numpy.zeros(size)
        for i in range(size):
            xrel = float(i) / float(size)
            if type == LOOKUP_TABLE_NONE:
                tmp = 1
            elif type == LOOKUP_TABLE_SINE:
                tmp = math.sin (xrel * math.pi * 2)
            elif type == LOOKUP_TABLE_COSINE:
                tmp = math.cos (xrel * math.pi * 2)
            elif type == LOOKUP_TABLE_HAMMING:
                tmp = 0.54 - 0.46 * math.cos(2 * math.pi * xrel)
            elif type == LOOKUP_TABLE_HANN:
                tmp = 0.5 - 0.5 * math.cos(2 * math.pi * xrel)
            #elif type == LOOKUP_TABLE_GAUSSIAN:
            #   // y = exp(1) .^ ( - ((x-size./2).*pi ./ (size ./ 2)) .^ 2 ./ 2);
            #   tmp = pow((double)exp(1.0), (double)(( - pow ((double)(((FLOAT)x-table_size / 2.0) * math.pi / (table_size / 2.0)) , (double)2.0)) / 2.0));
            else:
                raise Exception('type ' + str(type) + ' not recognized')
            data[i] = tmp
        return data

    def calculateF0once(
        self,
        data, 
        fs, 
        Fmin = 50,
        Fmax = 3000,
        voicingThreshold = 0.3,
        applyWindow = False
    ):
        '''
        calculates the fundamental frequency of a given signal.
        In this analysis the signal is treated as a monolithic data block, so this function, albeit being faster in execution, is only useful for stationary data. See calculateF0() for calculation of the time-varying fundemental frequency.
        Parameters
        data	a numpy array or a list if floats
        fs	sampling frequency [Hz]
        Fmin	lowest possible fundamental frequency [Hz]
        Fmax	highest possible fundamental frequency [Hz]
        voicingThreshold	threshold of the maximum in the autocorrelation function - similar to Praat's "Voicing threshold" parameter
        applyWindow	if True, a Hann window is applied to the FFT data during analysis
        
        Returns
        the estimated fundamental frequency [Hz], or 0 if none is found.
        '''

        LOOKUP_TABLE_HANN = 4

        dataTmp = copy.deepcopy(data)
        
        # apply window
        if applyWindow:
            fftWindow = self.createLookupTable(len(dataTmp), LOOKUP_TABLE_HANN)
            dataTmp *= fftWindow
        
        # autocorrelation
        result = numpy.correlate(dataTmp, dataTmp, mode = 'full')
        r = result[result.size//2:] / float(len(data))
        
        # find peak in AC
        freq = numpy.nan
        try:
            xOfMax, valMax = self.findArrayMaximum(r,
                int(round(float(fs) / Fmax)),
                int(round(float(fs) / Fmin)))
            valMax /= max(r)
            freq = float(fs) / xOfMax
        except Exception as e:
            pass
        return freq

    def lpAnalysis(self, s, fs, p=12):
        '''
        For the given signal estimate:
            LP Coefficients of the given signal using Burg's method,
            LP Residual
            Pitch using autocorrelation of LP Residual
            LP Spectrum
            Formants
        INPUT:
        s     =   signal upon which lp analysis is to be performed
        fs    =   sampling frequency of the signal (preferably 8kHz)
        p     =   lp order
        
        OUTPUT:
        lpres   =   LP Residual
        pitch   =   Pitch value in milli-seconds
        spectrum =  LP Spectrum
        formants    =   Formants in the spectrum (if any)
        '''

        if not s.flags['F_CONTIGUOUS']:  # to avoid memomy management based error
            s = numpy.asfortranarray(s)

        s = s*numpy.hamming(len(s)) # this step is necessary to destroy any discontinuites present at beginning/end of frame

        lp_coeff = librosa.lpc(s, p) # returns lp filter denom polynomial
        

        lp_res_org = scipy.signal.lfilter(lp_coeff, [1], s) # lp residual
        lp_res = lp_res_org/max(abs(lp_res_org))

        # Calculate pitch
        pitch = self.calculateF0once(lp_res, fs)


        # Calculate Formats
        frequency, spectrum = scipy.signal.freqz(1, lp_coeff)
        spectrum = abs(spectrum) # frequency spectrum
        spectrum = 20*numpy.log10(spectrum) # this is an optional step. This will only highlight the peaks in spectrum
        frequency = frequency*fs/(2*numpy.pi) # convert frequency to Hertz

        peaks, _ = scipy.signal.find_peaks(spectrum) # indices of peaks
        formants = frequency[peaks] # value is in Hertz

        return lp_res, pitch, spectrum, formants

    def plot(self, base64_string):

        data_orig, fs = sf.read(BytesIO(base64.b64decode(base64_string)))
        
        #If we have multichannel input
        if(data_orig.ndim == 2):
            data_orig = (data_orig[:,0] + data_orig[:,1])/2
        
        s = librosa.resample(data_orig, fs, 8000)
        fs = 8000

        # framing parameters
        frame_length = int( 0.030*fs ) # 30 ms 
        hop_length = 1
        NFFT = 1024

        # Spectrogram
        f, t, S = scipy.signal.spectrogram(s, fs=fs, window= scipy.signal.blackman(frame_length), nperseg=frame_length, noverlap=frame_length - hop_length, nfft=NFFT, mode='magnitude')


        # Energy Contour
        energy_contour = scipy.signal.convolve(s**2,numpy.ones(frame_length))
        energy_contour = copy.deepcopy( energy_contour[frame_length//2:-frame_length//2 + 1] )

        # Pitch Contour
        s_frames = librosa.util.frame(s, frame_length, hop_length)

        pitch_contour = []
        for i in range(0,numpy.shape(s_frames)[1]):
            
            _, pitch, _, _ = self.lpAnalysis( s_frames[:,i], fs, p=12 )

            pitch_contour.append(pitch)

        spectral_flatness = librosa.feature.spectral_flatness(S=S)
        spectral_flatness = spectral_flatness[0]


        zero_crossing_rate = librosa.feature.zero_crossing_rate(s, frame_length=frame_length, hop_length=hop_length, center=False)
        zero_crossing_rate = zero_crossing_rate[0]


        # some adjustments
        signal = s[0:len(t)]
        t = t*1000


        fig = make_subplots(
            rows=4, cols=2, 
            specs=[[{}, {}],
                [{"rowspan":3}, {}],
                [None, {}],
                [None, {}]], subplot_titles=("Signal", "Pitch Contour", "Spectrogram", "Energy Contour", "Spectral Flatness", "Zero Crossing Rate"),
            print_grid=True)

        fig.add_trace(
            go.Scatter(y=signal, x=t), 
            row=1, col=1
            )


        fig.add_trace(
            go.Scatter(y=pitch_contour, x=t, mode='markers', marker_size = 2), 
            row=1, col=2
            )

        fig.add_trace(
            go.Heatmap(
            x= t,
            y= f,
            z= 10*numpy.log10(S),
            colorscale='Jet',
            showscale=False
            ),
            row=2, col=1
            )

        fig.add_trace(
            go.Scatter(y=energy_contour, x=t), 
            row=2, col=2
            )


        fig.add_trace(
            go.Scatter(y=spectral_flatness, x=t), 
            row=3, col=2
            )


        fig.add_trace(
            go.Scatter(y=zero_crossing_rate, x=t), 
            row=4, col=2
            )

        fig.update_xaxes(matches='x', range=(min(t),max(t)))


        # Update xaxis properties
        fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
        fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
        fig.update_xaxes(title_text="Time (ms)", row=3, col=2)
        fig.update_xaxes(title_text="Time (ms)", row=4, col=2)


        # Update yaxis properties
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Energy", row=2, col=2)
        fig.update_yaxes(title_text="Flatness", row=3, col=2)
        fig.update_yaxes(title_text="Rate", row=4, col=2)

        # Update title and height
        fig.update_layout(height=900,title_text="Visualization Demo", title_font_size =24 ,showlegend=False, template="ggplot2")

        html_div = plotly.io.to_html(fig, full_html=True)

        # fig.write_html(html_filename)
        # fig.show(renderer='browser')

        return html_div
