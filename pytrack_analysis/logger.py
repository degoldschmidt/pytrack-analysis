import logging
import logging.config
import sys
from .profile import get_log

class Logger(object):
    """
    This class creates an object for the main entry point of logging
    """
    def __init__(self, profile, scriptname):
        """ Defines formatting and filehandler for logging """
        self.profile = profile
        self.scriptname = scriptname

        # logfilename from profile
        self.file_name = get_log(profile)
        self.fh = logging.FileHandler(self.file_name)

        # Set logger name
        self.log = logging.getLogger(scriptname)
        self.log.setLevel(logging.DEBUG)

        # Log messages are formatted as below
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.log.addHandler(self.fh)

        # Beginning of each log start
        self.log.info("==================================================")
        self.log.info("===* STARTING SCRIPT: {:} *===".format(scriptname))
        self.log.info("Part of project {:} (current user: {:})".format(profile['active'], profile['activeuser']))
        self.log.info("Hosted @ {:} (OS: {:})".format(profile[profile['active']]['systems'], OS))
        self.log.info("Python version: {:}".format(sys.version))

    def close(self):
        """ Closing of log """
        logger = logging.getLogger(self.scriptname)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.fh)
        logger.info("===*  ENDING SCRIPT  *===")
        logger.info("==================================================")
