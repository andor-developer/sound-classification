""" type: map
Sound File Storage: Useful for sound storage. Try to only store meta data if possible. All these are stores in a Features Objects. 
soundfiles[id] 
      -> label
      -> sampleRate
      -> fileLocation
      -> zcr
      -> waveFormData // This has to be stored in memory. Need to adjust moving forward. 
      -> rms
      -> spectralcentroid
      -> spectralrolloff
      -> spetral flux
"""
import config
import helpers
import feature_extractor as fex

soundfiles = {}

class Features(object):   
    def __init__(self):
        pass 
"""
Label Storage: Useful for quick lookup. 
labelmap[label] -> list<id's with label>
"""
labelmap = {}

"""
Internal Mapping:
labelmap[id] -> soundfiles[id] 
"""

def generateSamples(base_dir, out_dir, suffix):
    
    print('Generating Training Data')
    audio_files = listFilesInDirectory(base_dir)
    count = 0
    iterations_per_file = 5 # the amount of sample files to be generate from each base file
    
    for file in audio_files:
        wav, sampleFreq = readWavFile(file)
        while (count < iterations_per_file):
            sample = sampleFile(wav)
            sample = randomNoiseGenerator(sample, sampleFreq)
            writeSample(sample, join(out_dir + str(count) + "_" + suffix + ".wav")) #write the sample and after adding noise. Naming convention malleable. 
            count+=1
            
            
""" Toss in the generated datasets. These are the final datasets. """
def readSamples(dataset_dir):

    # Store classification in folders
    for subdir in helpers.listSubDirectories(dataset_dir):
        spldir = subdir.split("/")
        classifier = spldir[len(spldir) - 1]
        print("Working on " + classifier  + " classifier right now ... Please wait")
        for file in helpers.listFilesInDirectory(subdir):
           
            # Check for collisions and create UUID
            uuid = helpers.generateUUID() 
            if uuid in soundfiles: #upon collision create new UUID. Assumes no second collision. 
                uuid = helpers.generateUUID() 
          
            #Add Features
            features = Features()
            features.label = classifier 
            features.fileLocation = file
            features.uuid = uuid
            fe = fex.FeatureExtractor(features) 
            features = fe.extractFeatures()
            
            # Add Mapping for Easy LookUP of Classification to ID
            if features.label not in labelmap:
                idlist = []
                idlist.append(uuid)
                labelmap[str(features.label)] = idlist
            else:
                idlist = labelmap[str(features.label)]
                idlist.append(uuid) #add uuid
                labelmap[str(features.label)] = idlist
                
            soundfiles[uuid] = features  


def main():
    c = config.Config()
    readSamples(c.basedir)

if __name__== "__main__":
    main()