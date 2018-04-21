"""PREDICT IF COMPANY IS GREEN OR NOT."""

import pandas as pd
import h2o
from h2o.estimators import H2OXGBoostEstimator
from h2o.automl import H2OAutoML
from matplotlib import pyplot
import re

DATASET = 'data/processed/DSSG/GEMO2015_ 625 gültige Datensätze_171019.csv'
TARGET_VARIABLE = 'code_green'

# READ DATA
new_names = [
    "noname", "name_company", "PLZ", "town", "Bundesland", "url",
    "wz_2008_main_activity_code", "wz_2008_main_activity_code", "description",
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
    "hir_legal_form", "hir_commercial_register", "hiu_old_fimierung", "email", "Tel"
    ]

df = pd.read_csv(DATASET, nrow=3, names=new_names)

# START h2o
h2o.init()  # h2o.shutdown()
hf = h2o.H2OFrame(df)

# PARTITION DATA
train, test = hf.split_frame(ratios=[.8])

# RUN AUTOML
aml = H2OAutoML(max_runtime_secs=30)
aml.train(y='class',
          training_frame=train,
          validation_frame=test)

# EXAMING BEST MODELS
aml.leaderboard
aml.leader
