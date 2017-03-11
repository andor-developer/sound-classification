import loader as dl
import config
import logging

def main():
    c = config.Config()
    logging.warning('Starting controller.py')  # will print a message to the console
    dl.readSamples(c.basedir)

if __name__== "__main__":
    main()