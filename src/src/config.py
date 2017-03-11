class Config():
	
    basedir = ""
    dataset_dir = ""
    inter_dir_base = ""
    clean_dir_base = ""
    inter_dir_gen = ""
    base_dataset = ""
    clean_dir_gen = ""    
    
    def __init__(self):	
        self.basedir = '/Users/andor/workspace/sound-classification/src/data/instruments/'
        self.dataset_dir = self.basedir + 'datasets/'
        self.inter_dir_base = self.dataset_dir + 'base/interference_audio'
        self.clean_dir_base = self.dataset_dir + 'base/clean_audio'
        self.generated_dir = self.dataset_dir + 'generated/'
        self.base_dataset = self.dataset_dir + 'base/'
        self.inter_dir_gen = self.dataset_dir + 'generated/interference_audio/'
        self.clean_dir_gen = self.dataset_dir + 'generated/clean_audio/'
        self.PLOT_WIDTH  = 15
        self.PLOT_HEIGHT = 3.5    

