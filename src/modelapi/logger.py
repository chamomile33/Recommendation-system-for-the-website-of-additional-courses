import logging
import logging.handlers
from modelapi.config import config

rfh = logging.handlers.RotatingFileHandler(
    filename=config['logger']['filename'],
    mode='a',
    maxBytes=int(config['logger']['maxBytes']),
    backupCount=int(config['logger']['backupCount'])
)

logging.basicConfig(
    level=config['logger']['level'],
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    handlers=[
        rfh
    ]
)