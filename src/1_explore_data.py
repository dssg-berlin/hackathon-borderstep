import pandas as pd

BASE_DATA_PATH = 'data/processed/DSSG/'

def get_data(file_path):

    if 'xlsx' in file_path:
        data = pd.read_excel(BASE_DATA_PATH + file_path)

    elif 'pkl' in file_path:
        data = pd.read_pickle(BASE_DATA_PATH + file_path)

    return data


if __name__ == '__main__':

    gemo2015 = get_data("GEMO2015_ 625 gültige Datensätze_171019.xlsx")
    gemo2016 = get_data("GEMO2016_ 625 gültige Datensätze_171019.xlsx")
    print(gemo2016.shape)
