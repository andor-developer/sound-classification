
# coding: utf-8

# # Detecting Sound Interference with Tensorflow for Hangout Sessions
# 
# In this analysis, we look at sound interference from multiple hangout streams using google hangout. We train and test the data using a neural network.
# The goal of this network is to give Google the functionality to enable a "mute" action is multiple audio streams occur on the hangouts at the same time. Frequently, when my colleagues and myself use hangouts, we log in at the same time with our computer. The problem is that multiple microphones on at the same time create an issue with loud feedback loops that greatly disrupt a meeting. Furthermore, often it is difficult to determine which of the incoming channels is responsible for disturbing the audio system. 
# We are not audio processing experts, but hope that this simple neural network may provide enough of a baseline to accurately detect multiple audio feedback loops. It would be our hope that it would enable an "action" on Google's side, to mute the interefering audio system and prompt a warning. 

# Which type of Classifier do we use? SVM or NN?

# In[111]:

from pylab import*
from scipy.io import wavfile
import matplotlib
import matplotlib.pyplot as plt
import random


# In[102]:

sampleFreq, sample = wavfile.read('/Users/andorkesselman/Desktop/sample.wav')
print('Sample Frequency is' , sampleFreq, 'and sample type is', sample.dtype)
sample = sample / (2.**15) #normalize and center
print(sample.shape)
print('The duration of the audio file is' , (len(sample) / sampleFreq),  'ms') # find the duration 
ch1 = sample[:,0] #take one channel. There are two channels in this scenario


# In[96]:

timeArray = arange(0, len(sample), 1) # with steps of 1000
timeArray = timeArray / sampFreq 
print(timeArray[0:5,]) #Divide by the sample frequency to give the correct itme in miliseconds
timeArray = timeArray * 1000  #scale to seconds


# In[103]:

plt.plot(timeArray, ch1, color='r') #x=time y=ch1 which is channel 1


# In[105]:

#So we've plotted the values. Now we neeed to do some form of classification. 
#To generate this, we create our own training and test set. 
#First, we generate a single interference instance using google hangouts for ~5 mintues. 
#We then then randomly sample 10 seconds from the 5 minutes.
#Each sample then input a random noise variant using one of three different methods. 


# In[ ]:

# Generate Training Data introducing Random Noise Factor
def generateTrainingData():
    
    print('Generating Training Data')

    basedir='basedir'
    basefile = readBaseTrainingFiles(basedir)
    
    count = 0
    iterations = 5 # the amount of sample files to be generate from each base file
    
    while (count < iterations):
        sample = sampleFile(basefile)
        sample = randomNoiseGenerator(sample)
        
        outdir = basedir + str(count)
        writeSample(sample, outdir)

def readBaseTrainingFiles(basedir):
    print('reading basedir training files')

def sampleFile(basefile):
    print('Randomly Sampling File from BaseFile')
    return 0

def writeSample(sample, outdir):
    print('Writing Sample to ', outdir)
    


# In[129]:

#Generate White Noise
def noise1(array): 
    mean = 0
    std = 1 
    num_samples = 1000
    samples = numpy.random.normal(mean, std, size=num_samples)
    print('Adding white noise to the sample set')
    return array

#Generating Noise Using http://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def noise2(array):
    return band_limitd_noise(array)

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def randomNoiseGenerator(sample):
    r = random.randint(0, 3)
    if r == 0:
        return noise1(sample)
    if r == 1:
        return band_limited_noise(sample)
    else:
        return sample


# In[ ]:

generateTrainingData()


# In[ ]:



