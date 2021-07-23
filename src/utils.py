from datetime import datetime

# Dateime Parsing String
DATETIME_PSTR = '%Y_%m_%d'

def _decode_time_str(time):
    return datetime.strptime(time, DATETIME_PSTR)