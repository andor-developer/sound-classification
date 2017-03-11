from os import listdir
from os.path import isfile, join
import os
import uuid

""" List the files in the directory """
def listFilesInDirectory(directory):
    ret = []
    for file in listdir(directory):
            ret.append(join(directory, file))
    return ret

""" List the directories in a parent directory. Returns a list of directories. """
def listSubDirectories(directory):
    ret = []
    for subdirectory in listdir(directory):
            path = join(directory, subdirectory)
            if(os.path.isdir(path)):
                ret.append(path)
    return ret

""" Generate Random UUID """
def generateUUID():
    return uuid.uuid4() 

def testFileHelpers():
    print("Listing Files in Directory")
    print(listFilesInDirectory(inter_dir_gen))
    print("Listing SubDirectories")
    print(listSubDirectories(dataset_dir))