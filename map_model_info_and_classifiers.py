# coding: utf-8
import json
model_info = json.load(open('models_and_best_model.json', 'r'))
import pandas as pd
df = pd.read_csv('best_hyper_parameters.csv')
model_types = df['models'].to_list()
classifiers_info = json.load(open('model_shortcut_to_fullform.json', 'r'))

classifiers = []
classifiers_short = []
for m in model_types:
    m_type = m.split('-')[0]
    mod = '-'.join(m.split('-')[1:])
    if m_type == 'rdkDes':
        temp = model_info['rdkitDescriptors'][mod]
    elif m_type == 'tpatf':
        temp = model_info['pharmacophore'][mod]
    elif m_type == 'volsurf':
        temp = model_info['volsurf'][mod]
    else:
        temp = model_info['fingerprints'][mod]

    classifiers_short.append(temp)
    classifier_name = classifiers_info[temp]    
    classifiers.append(classifier_name)

df.insert(1, 'classifiers', classifiers)
df.insert(2, 'Abbrev.', classifiers_short)
df.to_csv('best_hyper_parameters_with_classifiers.csv', index=False)
