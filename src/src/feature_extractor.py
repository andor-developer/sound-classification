import numpy as np
import soundimporter as sr
""" 
Features Extraction, gives us the data to store things like the zcr, waveformdata, rms, etc into a feature object. 
type: map
Sound File Storage: Useful for sound storage. Try to only store meta data if possible. All these are stores in a Features Objects. 
soundfiles[id] 
      -> label
      -> sampleRate
      -> fileLocation
      -> zcr --very useful
      -> waveFormData // This has to be stored in memory. Need to adjust moving forward. 
      -> rms
      -> spectralcentroid
      -> spectralrolloff
      -> spetral flux
"""
class FeatureExtractor(object):

    def __init__(self, features):
        self.features = features
        
    def extractFeatures(self):
        waveform, sampleRate = sr.readWavFileAllChannels(self.features.fileLocation)
        self.features.waveform = waveform
        self.features.sampleRate = sampleRate
        
        blocklength = 2048
       # self.features.zcr = zero_crossing_rate(waveform, blocklength, sampleRate)
       # self.features.rms = root_mean_square(waveform, blocklength, sampleRate)
       # self.features.spectral_centroid = spectral_centroid(waveform, blocklength, sampleRate)
       # self.features.spectral_rolloff = spectral_rolloff(waveform, blocklength, sampleRate)
      #  self.features.spectral_flux = spectral_flux(waveform, blocklength, sampleRate)

        return self.features

""" 

Short time forier transform. The results of the short form fourier transform are three fold:

1. Phase Histogram
2. Frequency Histogram
3. Set of bins
4. Above with alternative frequency representations

The fast forier transform uses a divide and conquer algorithm to efficiently computer the discrete Fourier transform
Rather than creating our own function, we are using numpy's fft algorithm out of the box. Using Numpy's fft method, 

Returns: A complex data array: The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified. If n is even, the length of the transformed axis is (n/2)+1. If n is odd, the length is (n+1)/2. 

Uses a hanning window. The Hanning window is a taper formed by using a weighted cosine.
https://en.wikipedia.org/wiki/Hann_function
}
"""
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    
    # get the frame size and the hopsize
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = np.lib.stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    #use numpys computes the one-dimensional discrete Fourier Transform for real input.
    return np.fft.rfft(frames)    


""" scale frequency axis logarithmically. Not sure exactly how this function works. Need to revisit in due time. """    
def logscale_spec(spec, sr=44100, factor=20.): # TODO: The sample rate needs to be adjusted to the sample rate of the music
    timebins, freqbins = np.shape(spec) #Get the shape of the spec
    print("TimeBins are " , timebins , " Frequency Bins are " , freqbins)
    scale = np.linspace(0, 1, freqbins) ** factor  # what is the factor?
    scale *= (freqbins-1)/max(scale) #Scale values for bins. 
    scale = np.unique(np.round(scale)) #Round the interger values
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)])) #real and complex number
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1) 
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
            
    
    return newspec, freqs


"""The following method is numerical more elegant and computationally efficient. It uses the optimized linear algebraic functions of the Numerical Python (numpy) package.
The method further computes the Zero Crossing Rate for a sequence of blocks (also called frames or windows).

\[ zcr = \frac{1}{N-1} \sum_{i=1}^{N-1} | sign|x(i)| - sign|x(i-1)| | \]

This particular implementation takes the zero crossing rate over blocks of specified size (will try 2048). 

"""
def zero_crossing_rate(wavedata, block_length, sample_rate):
    
    # how many blocks have to be processed and when do they start?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))
    
    zcr = []
    
    for i in range(0,num_blocks-1):
        
        # Get the zero crossing rate for a block of data using numpy's diff and sign package.
        # Multiply it by .5 at the end to get the rate. May have to change .5. 
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    
    return np.asarray(zcr), np.asarray(timestamps)

""" Root mean squared calculates the relative energy. It uses the amplitude, but calculates the instantaneous energy. """
def root_mean_square(wavedata, block_length, sample_rate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))
    
    rms = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        rms_seg = np.sqrt(np.mean(wavedata[start:stop]**2))
        rms.append(rms_seg)
    
    return np.asarray(rms), np.asarray(timestamps)

""" Determines where the concentration of energy occurs. """
def spectral_centroid(wavedata, window_size, sample_rate):
    
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sc = []

    for t in range(timebins-1):
        power_spectrum = np.abs(magnitude_spectrum[t])**2      
        sc_t = np.sum(power_spectrum * np.arange(1,freqbins+1)) / np.sum(power_spectrum)
        sc.append(sc_t)
    
    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)
    
    return sc, np.asarray(timestamps)


""" Computes the spectral rolloff which measures the skewness of the spectral shape. """
def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = np.abs(magnitude_spectrum)**2
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sr = []
    spectralSum    = np.sum(power_spectrum, axis=1)
    
    for t in range(timebins-1):
        
        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]
        
        sr.append(sr_t)
        
    sr = np.asarray(sr).astype(float)
    
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)
    
    return sr, np.asarray(timestamps)

""" Measures the rate of local change"""
def spectral_flux(wavedata, window_size, sample_rate):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum))**2, axis=1)) / freqbins
    
    return sf[1:], np.asarray(timestamps)

""" Returns the average waveform across two channels """
def averageWaveform(sample):
    return np.mean(sample,axis=1)

"""Simple check to make sure the feature extrator is working. Should be run only once."""
def testFeatureExtraction():
    print("Testing Feature Extractor")
    fileloc = '/Users/andorkesselman/Documents/rnd/sound/src/datasets/base/clean_audio/1_clean.wav'
    wav = readWavFile(fileloc)
    features = Features()
    features.fileLocation = fileloc
    features = FeatureExtractor(features).extractFeatures()
    pprint(vars(features))
    print("---------FEATURE EXTRATOR PASSED----------")

def testFileViewer():
    samples, freq = readWavFileAllChannels(clean_dir_base + "/1_clean.wav")
    show_stereo_waveform(samples)
    
#Run 
def testFouriuerViewer():
    samples, freq = readWavFileAllChannels(clean_dir_base + "/1_clean.wav")
    plotstft(samples, freq)

    
def testZeroCrossingRate():
    samples, freq = readWavFileAllChannels(clean_dir_base + "/1_clean.wav")
    zcr, ts = zero_crossing_rate(samples, 2048, freq);
    print('TS is ', ts)
    show_feature_superimposed(samples[:,0], freq, zcr, ts);
    
def testSounds():
    #testFileViewer()
    #testFouriuerViewer()
    testZeroCrossingRate()
