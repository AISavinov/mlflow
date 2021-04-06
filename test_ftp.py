import os
from ftplib import FTP

ftp = FTP()
ftp.debugging = 1
with open(local_file, "rb") as f:
    base_name = os.path.basename(local_file)
    print(base_name)
    #ftp.nlst()
    ftp.set_pasv(False)
    ftp.retrlines('LIST')
    # ftp.cwd(artifact_dir)
    ftp.storbinary("STOR " + base_name, f)
ftp.close()
