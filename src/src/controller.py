import data_loader as dl
import config

def main():
    c = config.Config()
    dl.readSamples(c.basedir)

if __name__== "__main__":
    main()