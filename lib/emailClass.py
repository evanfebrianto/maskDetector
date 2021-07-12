import datetime
import mimetypes
import os, smtplib
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from lib.helper import parse_config, zip_dir, get_latest_folder

class EmailModule():
    def __init__(self, configFile='config.ini') -> None:
        parser = parse_config(configFile)

        self.sender = parser['STRING']['SENDER_EMAIL']
        self.receivers = parser['STRING']['RECEIVER_EMAILS']
        self.password = parser['STRING']['PASSWORD']
        self.log_dir = parser['STRING']['LOG_DIR']
        self.msg = MIMEMultipart()
        self.last_folder_sent = None
    
    def setProperties(self, subject, body_message='') -> None:
        # Remove existing zip file
        if os.path.exists(os.path.join(self.log_dir,"to_send.zip")):
            os.remove(os.path.join(self.log_dir,"to_send.zip"))
        
        # Create a new zip file from latest folder
        latest_folder = get_latest_folder(dir_path=self.log_dir)
        zip_dir(dir_path=os.path.join(self.log_dir, latest_folder), 
                outFullName=os.path.join(self.log_dir,"to_send.zip"))
        filepath = os.path.join(self.log_dir,"to_send.zip")

        # Set attachment name in email
        time = datetime.datetime.today().strftime("%d-%m-%Y %H:%M:%S")
        attachment_name = "No Mask Report {}.zip".format(time)
        
        # format multireceiver
        receiver = ''
        for e in self.receivers:
            receiver += e + ','
        receiver[:-1] # delete last comma

        # Message body
        self.msg.attach(MIMEText(body_message,'plain','utf-8'))
        self.msg['From'] = self.sender
        self.msg['To'] = receiver 
        self.msg['Subject'] = subject

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
        self.msg.attach(file_msg)


    def sendEmail(self) -> None:
        latest_folder = get_latest_folder(dir_path=self.log_dir)
        print('last_sent: {}\nlatest_folder: {}'.format(self.last_folder_sent, latest_folder))
        if self.last_folder_sent != latest_folder:
            try:
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(self.sender,self.password)
                server.sendmail(self.sender,self.receivers,self.msg.as_string())
                server.quit()
                self.last_folder_sent = latest_folder
                print("Email sent successfully")
            except Exception as err:
                print("Failed to send emails")
                print(err)