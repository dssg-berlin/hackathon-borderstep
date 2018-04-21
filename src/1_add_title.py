import pandas as pd
import numpy as np
import html


# /Users/jpapaioannou/Documents/Repos/private/dssg2018/hackathon-borderstep/
BASE_DATA_PATH = 'data/processed/'
# 'word_count_2015_2016.pkl'
BASE_DATA_PATH_RESULTS = 'data/processed/DSSG/'


def get_data(file_path):
    if 'xlsx' in file_path:
        data = pd.read_excel(BASE_DATA_PATH + file_path)

    elif 'pkl' in file_path:
        data = pd.read_pickle(BASE_DATA_PATH + file_path)

    return data


def extract_title_from_html(ex):
    from bs4 import BeautifulSoup
    if ex:
        soup = BeautifulSoup(ex, 'html.parser')
        for header in soup.find_all('h1'):
            if header:
                yield header.get_text().translate(str.maketrans("\n\t\r\xa0", "    ")).strip()


"""
def read_files_from_data_folder(filenames):
    for filename in filenames:
        domain_info = get_data(filename)
        for idx, row in domain_info[['html']].dropna().iterrows():
            html_text = row.values.item()
            headers = ' '.join(extract_title_from_html(html_text))
            domain_info.loc[idx, 'titles'] = headers
        end = filename.find('.pkl.gz')
        filename = filename[:end]
        filepath = BASE_DATA_PATH_RESULTS+filename+'_with_titles.pkl.gz'
        domain_info.to_pickle(filepath, compression='infer', protocol=4)
"""


def extract_text_from_title(x):
    html_text = x.values.item()
    headers = ' '.join(extract_title_from_html(html_text))
    return headers


def read_files_from_data_folder(filenames):
    for filename in filenames:
        domain_info = get_data(filename)
        domain_info['titles'] = \
            domain_info[['html']].dropna().apply(lambda x: extract_text_from_title(x),
                                                 axis=1)
        end = filename.find('.pkl.gz')
        filename = filename[:end]
        filepath = BASE_DATA_PATH_RESULTS+filename+'_with_titles.pkl.gz'
        domain_info.to_pickle(filepath, compression='infer', protocol=4)


if __name__ == '__main__':
    #filenames = ['GEMO-GrüneUN2006-2013.pkl.gz', 'GEMO_2015.pkl.gz', 'GEMO_2016.pkl.gz']
    filenames = ['word_count_2015_2016.pkl']
    read_files_from_data_folder(filenames)

