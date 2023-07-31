import os


class Config:
    """
    Handle routing through folders.
    In case of any troubles, please replace PATH parameter with your path to the project folder.
    Also replace filename if it does not match.
    """
    # PATH can be replaced
    PATH = os.chdir(os.path.join(os.getcwd(), "..")) if ("notebook" or "src") in os.getcwd() else os.getcwd()
    # FNAME can be replaced
    FNAME = 'all_ATV_new.csv'
    #
    PATH2RAW_DATA = os.path.join(PATH, "data/raw")
    PATH2PROCESSED_DATA = os.path.join(PATH, "data/processed")
    PATH2EXTERNAL_DATA = os.path.join(PATH, "data/external")
    PATH2MODELS = os.path.join(PATH, "models")
    PATH2FIGS = os.path.join(PATH, "reports/figures")
    PATH2TALBES = os.path.join(PATH, "reports/tables")
    PATH2REPORT = os.path.join(PATH, 'reports')
    
    @classmethod
    def get_path2file(cls, fname, rawdata=None, processeddata=None, external=None, models=None, report=None, figures=None, tables=None):
        path = cls.PATH
        if rawdata is not None:
            path = cls.PATH2RAW_DATA
        if processeddata is not None:
            path = cls.PATH2PROCESSED_DATA
        if external is not None:
            path = cls.PATH2EXTERNAL_DATA
        if models is not None:
            path = cls.PATH2MODELS
        if figures is not None:
            path = cls.PATH2FIGS
        if tables is not None:
            path = cls.PATH2TALBES
        if report is not None:
            path = cls.PATH2REPORT
        return os.path.join(path, fname)