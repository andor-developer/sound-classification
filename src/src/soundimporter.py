from scipy.io import wavfile


# Read Wav File from Location Specified in Method Call
def readWavFile(wavfile_location):
    sampleFreq, sample = wavfile.read(wavfile_location)
    sample = sample / (2.**15) #normalize and center
    ch1 = sample[:,0] #take one channel. There are two channels in this scenario
    return ch1, sampleFreq

# Read Wav File from Location Specified in Method Call
def readWavFileAllChannels(wavfile_location):
    sampleFreq, sample = wavfile.read(wavfile_location)
    sample = sample / (2.**15) #normalize and center
    ch1 = sample[:,0] #take one channel. There are two channels in this scenario
    return sample, sampleFreq
 
# Write Audio Sample To File
def writeSample(sample, outdir):
    wavfile.write(outdir, 44100, sample)

""" Randomly Returns a Sample of a File. TODO: Improve Sampling Method.  
Must Remain a Sequence in this Case because Audio is time dependent. Randomly Sampling would be BAD. """
def sampleFile(wav):
    #take random 
    size = int(len(wav) / 5) # take 1/5 of the full file size. TODO: What's the best implemenation?
    i = int(len(wav) - size - 1)
    r = random.randint(0, i) # make sure that we get a full set. Hence the - size
    return wav[r:(r+size), ]