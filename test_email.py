import datetime
from lib.helper import parse_config, zip_dir
import lib.emailClass

time = datetime.datetime.today().strftime("%d-%m-%Y %H:%M")

email = lib.emailClass.EmailModule(configFile='config.ini')
email.setProperties(subject='No Mask Report at {}'.format(time),
    body_message='Hi, please kindly check the attached file for your reference.')
email.sendEmail()
