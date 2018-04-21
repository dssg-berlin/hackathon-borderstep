"""EXTRACT ZIP FILE FROM data/input TO data/processed."""

import zipfile

zip_file = 'data/input/DSSG.zip'
target_dir = 'data/processed/'

zip_ref = zipfile.ZipFile(zip_file, 'r')
zip_ref.extractall(target_dir)
zip_ref.close()
