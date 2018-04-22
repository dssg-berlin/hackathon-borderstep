domain_info.loc[domain_info.WZ_2008_main_code_1.isnull(),
                'WZ_2008_main_code_1'] = '-1'
feat = domain_info.WZ_2008_main_code_1
enc = OneHotEncoder()
le = LabelEncoder()
label_encoded_feat = le.fit_transform(feat)
one_hot = enc.fit_transform(label_encoded_feat.reshape(-1,1))