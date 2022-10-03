import logging

logging.basicConfig(
    filename='output.log',
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)

logger = logging.getLogger('ai_log')