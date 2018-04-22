import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
import re


BASE_DATA_PATH = 'data/processed/'
BASE_DATA_PATH_RESULTS = 'data/processed/DSSG/'


def get_data(file_path):
    if 'xlsx' in file_path:
        data = pd.read_excel(BASE_DATA_PATH + file_path)

    elif 'pkl' in file_path:
        #data = pd.read_pickle('data/processed/word_count_full.pkl')
        data = pd.read_pickle(BASE_DATA_PATH + file_path)

    return data


def extract_title_from_html(ex):

    if ex:
        soup = BeautifulSoup(ex, 'html.parser')
        return ' '.join(header.get_text().translate(str.maketrans("\n\t\r\xa0", "    ")).strip() for header in soup.find_all(['title','h1','h2','h3','h4','h5','h6']))


def extract_text_from_title(pages):
    titles = joblib.Parallel(n_jobs=5)(joblib.delayed(extract_title_from_html)(page) for page in tqdm(pages.values))
    return titles


def read_files_from_data_folder(filenames):
    for filename in filenames:
        domain_info = get_data(filename)
        domain_info['titles'] = np.nan
        print('extracting titles')
        domain_info.loc[domain_info.html.notnull(), 'titles'] = extract_text_from_title(domain_info['html'].dropna())
        print('extracted title')
        end = filename.find('.pkl')
        filename = filename[:end]
        filepath = BASE_DATA_PATH_RESULTS+filename+'_with_titles.pkl.gz'
        print('saving text')
        domain_info.to_pickle(filepath, compression='infer', protocol=4)
        print('saved text')

if __name__ == '__main__':
    filenames = ['word_count_full.pkl']
    read_files_from_data_folder(filenames)

