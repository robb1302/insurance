import os
import shutil
import sys

ROOT_FOLDER = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0])) + '/'
ROOT_FOLDER = ROOT_FOLDER.replace('\\', '/')

# Creating tmp folder
TEMP_FOLDER = ROOT_FOLDER + 'tmp/'
PNG_FOLDER = TEMP_FOLDER + 'png/'

if os.path.exists(TEMP_FOLDER):
    print('Temporary folder exists. Deleting and recreating...')
    shutil.rmtree(TEMP_FOLDER)



os.mkdir(TEMP_FOLDER)
os.mkdir(PNG_FOLDER)

# Creating output folder
OUT_FOLDER = ROOT_FOLDER + 'out/'
PDF_FOLDER = OUT_FOLDER + 'pdf/'
JSON_FOLDER = OUT_FOLDER + 'json/'
PIC_FOLDER = OUT_FOLDER + 'pic/'
MODEL_FOLDER = OUT_FOLDER + 'model/'


if os.path.exists(OUT_FOLDER):
    print('Output folder exists. Deleting and recreating...')
    shutil.rmtree(OUT_FOLDER)

os.mkdir(OUT_FOLDER)
os.mkdir(PDF_FOLDER)
os.mkdir(JSON_FOLDER)
os.mkdir(PIC_FOLDER)
os.mkdir(MODEL_FOLDER)

WEIGHTS_FOLDER = OUT_FOLDER + 'weights/'

DATA_FOLDER = ROOT_FOLDER + 'data/'

# Settings-Folder
SETTINGS_FOLDER = ROOT_FOLDER + 'src/settings/'
MODEL_SETTINGS_FOLDER = SETTINGS_FOLDER + 'calculateModel/model_settings/'


