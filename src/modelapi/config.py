from configparser import ConfigParser
import os

config = ConfigParser()
config.read("settings.ini")

sensitive_config = ConfigParser()
sensitive_config.read("sensitive_settings.ini")

if not os.path.isdir(config['paths']['temp']):
    os.makedirs(os.path.dirname(config['paths']['temp']), exist_ok=True)

config['paths']['saved_models'] = config['paths']['temp'] + '/saved_models/'

if not os.path.isdir(config['paths']['saved_models']):
    os.makedirs(os.path.dirname(config['paths']['saved_models']), exist_ok=True)
