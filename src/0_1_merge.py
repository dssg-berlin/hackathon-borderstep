"""TODO: description of script here."""

import pandas as pd
import numpy as np

def merge_data(df_excel, df_web):
    df_excel.drop_duplicates(subset="domain", keep="first", inplace=True)
    df_merged = pd.merge(df_excel, df_web, on="domain")
    return df_merged

col_names = ["noname", "name_company", "PLZ", "town", "Bundesland", "url_2",
             "wz_2008_main_activity_code", "wz_2008_main_activity_code_description", "description",
             "code_green", "hints", "hints_business_model", "product_cat",
             "CreMA_13A", "CReMA_13B", "CReMA_11A", "CReMA_12", "CReMA_12_add",
             "CReMA_10", "CReMA_11B", "CReMA_13C", "CReMA_14", "CReMA_15", "CReMA_16",
             "CEPA_2", "CEPA_3", "CEPA_3_add", "CEPA_1", "CEPA_4", "CEPA_5", "CEPA_7", "CEPA_8", "CEPA_6", "CEPA_9",
             "sustainabile_agriculture", 'sustainable_mobility',
             'green_IT', 'green_finance', 'green_services', 'date_founded',
             'legal_form_today', 'legal_form_today_since', 'crefo_no',
             'commercial_register_no', 'capital_in_euro',
             'no_principals',
             'MAN_function', 'MAN_name', 'MAN_date_founded',
             'MAN_legal_form', 'MAN_company',
             'WZ_2008_main_code_1',
             'WZ_2008_main_code_1_desc',
             'WZ_2008_code',
             'WZ_2008_code_desc',
             'WZ_2008_secondary_code',
             'WZ_2008_secondary_description',
             'ONACE_2008_main_code',
             'ONACE_2008_main_description', 'stock_value',
             'quote_import', 'quote_export', 'revenue_last_year',
             'revenue_last',
             'revenue_year_before_last',
             'renenue_last_in_tsd_euro',
             'revenue_year_lag_one', 'revenue_year_lag_two',
             'no_emploies_last_year', 'no_emploies_last',
             'no_emploies_last_available',
             'no_emploies_last_availabe_2',
             'no_emploies_lag_one',
             'no_emploies_lag_two', "hr_first_entry_date",
             "hr_latest_entry_date", "reason_last_entry", "hi_entry_date", "hi_reason", "legal_form_first_date", "legal_form_first_type",
             "hir_legal_form", "hir_commercial_register", "hiu_old_fimierung", "email", "Tel"]

col_names_2006_2013 = ["noname", "name_company", "PLZ","town","empty","url_2", "date_founded",
             "wz_2008_main_activity_code", "wz_2008_main_activity_code_description", "code_green",
             "hints", "hints_leistung", "fourth_leistung", 
             "CreMA_13A", "CReMA_13B", "CReMA_11A", "CReMA_12", "CReMA_12_add",
             "CReMA_10", "CReMA_11B", "CReMA_13C","CReMA_14","CReMA_15","CReMA_16",
             "CEPA_2","CEPA_3","CEPA_3_add","CEPA_1","CEPA_4","CEPA_5","CEPA_7", "CEPA_8","CEPA_6","CEPA_9", 
             "Bundesland", 'renenue_last_in_tsd_euro',
             'revenue_year_lag_one', 'revenue_year_lag_two', 'revenue_year_lag_three', 'revenue_year_lag_four',
             'revenue_year_lag_five', 'revenue_year_lag_six', 'revenue_2014', 'revenue_2013', 'revenue_year_2012',
             'revenue_year_2011', 'revenue_year_2010', 'revenue_year_2009', 'revenue_year_2008',
             'umgruendung', 'vorheriger_name', 'stock_market_listed', 'quote_import', 'quote_export',
             'ONACE_2008_main_code',
             'ONACE_2008_main_description',
             'ONACE_2008_secondary_code',
             'ONACE_2008_haupt_secondary_code',
             'ONACE_2008_neben_description',
             'ONACE_2008_haupt_secondary_description',
             'ONACE_2008_main_code_1',
             'ONACE_2008_main_description_1',
             'ONACE_2008_main_code_3',
             'ONACE_2008_main_description_3',
             'ONACE_2008_main_code_4',
             'ONACE_2008_main_description_4',
             'no_emploies_last_available',
             'no_emploies_lag_one',
             'no_emploies_lag_two',
             'no_emploies_lag_three',
             'no_emploies_lag_four',
             'no_emploies_lag_five',
             'no_emploies_lag_six',
             'no_emploies_2014',
             'no_emploies_2013',
             'no_emploies_2012',
             'no_emploies_2011',
             'no_emploies_2010',
             'no_emploies_2009',
             'no_emploies_2008',
             'revenue_last_year',
             'no_emploies_last_year',
             'crefo_no',
             'commercial_register_no',
             'WZ_2008_code',
             'WZ_2008_code_desc',
             'WZ_2008_secondary_code',
             'WZ_2008_secondary_description',
             'WZ_2008_main_code_1',
             'WZ_2008_main_code_1_desc',
             'WZ_2008_main_code_3',
             'WZ_2008_main_code_3_desc',
             'WZ_2008_main_code_4_desc',
             'WZ_2008_main_code_4'
             ]

GEMO2015 = pd.read_csv(
    "data/processed/DSSG/GEMO2015_ 625 gültige Datensätze_171019.csv")
GEMO2015.columns = col_names
GEMO2016 = pd.read_csv(
    "data/processed/DSSG/GEMO2016_ 625gültige_Datensätze_171019.csv")
GEMO2016.columns = col_names

# 2006 - 2013 haben ein anderes Format
GEMO2006_13 = pd.read_csv("data/processed/DSSG/GEMO-Grüne UN 2006-2013_angepasstes sample für DSSG.csv")
GEMO2006_13.columns = col_names_2006_2013


# load and set index to domain:
GEMO2015['domain'] = GEMO2015['url_2'].str.replace('http://|https://|www.', '')
GEMO2015.index = GEMO2015['domain']
GEMO2016['domain'] = GEMO2016['url_2'].str.replace('http://|https://|www.', '')
GEMO2016.index = GEMO2016['domain']

GEMO2006_13['domain'] = GEMO2006_13['url_2'].str.replace('http://|https://|www.', '')
GEMO2006_13.index = GEMO2006_13['domain']


GEMO2016_html = pd.read_pickle("data/processed/DSSG/GEMO_2016.pkl.gz")
GEMO2015_html = pd.read_pickle("data/processed/DSSG/GEMO_2015.pkl.gz")
GEMO2006_2013_html = pd.read_pickle("data/processed/DSSG/GEMO-GrüneUN2006-2013.pkl.gz")

df = merge_data(GEMO2015, GEMO2015_html).append(merge_data(GEMO2016, GEMO2016_html))

df2 = merge_data(GEMO2006_13, GEMO2006_2013_html)

col_names_both = np.intersect1d(list(df.columns), list(df2.columns))

df3 = (df[col_names_both]).append(df2[col_names_both])
df3.to_pickle('data/processed/merge_full.pkl')
