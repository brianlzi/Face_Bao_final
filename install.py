import os

os.system("pip install -r requirements.txt")
os.system("conda install -c conda-forge opencv")
os.system("cd .. && git clone https://github.com/tzutalin/labelImg.git")
os.system("conda install pyqt=5")
os.system("cd labelImg && pyrcc5 -o libs/resources.py resources.qrc")