import datetime
import mimetypes
import os, smtplib, zipfile
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from helper import parse_config

def zip_dir(dir_path, outFullName):
    """
    Compress the specified folder
    :param dir_path: target folder path
    :param outFullName: Compressed file save path+XXXX.zip
    :return:
    """
    testcase_zip = zipfile.ZipFile(outFullName, 'w', zipfile.ZIP_DEFLATED)
    for path, dir_names, file_names in os.walk(dir_path):
        for filename in file_names:
            testcase_zip.write(os.path.join(path, filename))
    testcase_zip.close()
    print("Packed successfully")

zip_dir('lib/tesla', 'lib/output.zip')

parser = parse_config('config.ini')

Sender_Email = parser['STRING']['SENDER_EMAIL']
Receiver_Emails = parser['STRING']['RECEIVER_EMAILS']
Password = parser['STRING']['PASSWORD']

filepath = "lib/tesla.zip"
attachment_name = "Test report.zip"

time = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
msg = MIMEMultipart()
# Message body
msg.attach(MIMEText("See the attachment for the test report of {}".format(time),'plain','utf-8'))
msg['From'] = Sender_Email
msg['To'] = Receiver_Emails[0]+','+Receiver_Emails[1]
subject = "{} Test Report".format(time)
msg['Subject'] = subject

data = open(filepath, 'rb')
ctype, encoding = mimetypes.guess_type(filepath)
if ctype is None or encoding is not None:
    ctype = 'application/octet-stream'
maintype, subtype = ctype.split('/', 1)
file_msg = MIMEBase(maintype, subtype)
file_msg.set_payload(data.read())
data.close()
encoders.encode_base64(file_msg)  # Encode the attachment
file_msg.add_header('Content-Disposition', 'attachment', filename=attachment_name)  # Modify email header
msg.attach(file_msg)
try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(Sender_Email,Password)
    server.sendmail(Sender_Email,Receiver_Emails,msg.as_string())
    server.quit()
    print("Sent successfully")
except Exception as err:
    print("Failed to send")
    print(err)


