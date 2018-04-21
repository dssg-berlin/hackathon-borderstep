import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
col_names = ["noname", "name_company", "PLZ","town","Bundesland","url",
             "wz_2008_main_activity_code", "wz_2008_main_activity_code_description", "description",
             "code_green", "hints", "hints_business_model", "product_cat", 
             "CreMA_13A", "CReMA_13B", "CReMA_11A", "CReMA_12", "CReMA_12_add",
             "CReMA_10", "CReMA_11B", "CReMA_13C","CReMA_14","CReMA_15","CReMA_16",
             "CEPA_2","CEPA_3","CEPA_3_add","CEPA_1","CEPA_4","CEPA_5","CEPA_7", "CEPA_8","CEPA_6","CEPA_9",
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
        "hr_latest_entry_date", "reason_last_entry","hi_entry_date", "hi_reason", "legal_form_first_date", "legal_form_first_type", 
        "hir_legal_form", "hir_commercial_register", "hiu_old_fimierung", "email", "Tel"]

GEMO2015 = pd.read_csv("data/processed/DSSG/GEMO2015_ 625 gültige Datensätze_171019.csv")
GEMO2015.columns = col_names
GEMO2016 = pd.read_csv("data/processed/DSSG/GEMO2016_ 625gültige_Datensätze_171019.csv")
GEMO2016.columns = col_names

# 2006 - 2013 haben ein anderes Format
#GEMO2006_13 = pd.read_csv("GEMO-Grüne UN 2006-2013_angepasstes sample für DSSG.csv")
#GEMO2006_13.columns = col_names


# load and set index to domain:
GEMO2015['domain']= GEMO2015['url'].str.replace('http://|https://|www.', '')
GEMO2015.index = GEMO2015['domain']
GEMO2016['domain'] = GEMO2016['url'].str.replace('http://|https://|www.', '')
GEMO2016.index  = GEMO2015['domain'] 
GEMO2015["domain"].is_unique # ?


# remove duplicates
GEMO2015.drop_duplicates(subset="domain", keep="first", inplace=True)
GEMO2016.drop_duplicates(subset="domain", keep="first", inplace=True)


GEMO2016_html = pd.read_pickle("data/processed/DSSG/GEMO_2016.pkl.gz")
GEMO2015_html = pd.read_pickle("data/processed/DSSG/GEMO_2015.pkl.gz")


GEMO2015_html["domain"].value_counts()[0:10]

df = pd.merge(GEMO2015, GEMO2015_html, on="domain")
df = df.append(pd.merge(GEMO2016, GEMO2016_html, on="domain"))



print(GEMO2015.shape, GEMO2015_html.shape)
print(GEMO2016.shape, GEMO2016_html.shape)
print(df.shape)

#function for later

def joined_data(Gemo2015, Gemo2015_web, Gemo2016, Gemo2016_web):

    def merge_data(df_excel, df_web):
        df_excel['domain']= df_excel['url'].str.replace('http://|https://|www.', '')
        df_excel.drop_duplicates(subset="domain", keep="first", inplace=True)
        df_merged = pd.merge(df_excel, df_web, on="domain")
        return df_merged

    df_full = merge_data(Gemo2015, Gemo2015_web).append(merge_data(Gemo2016, Gemo2016_web))
    return df_full


df.to_pickle('data/processed/merge_2015_2016.pkl')
