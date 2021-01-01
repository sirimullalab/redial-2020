# coding: utf-8
import glob
import os, sys
import pickle
import json
dictn = {}
for m in glob.glob('./redial-2020-notebook-work/models_tuned_best/*.pkl'):
    name,_=os.path.splitext(os.path.basename(m))
    fp_name = name.split('-')[0]
    name = name[:-31]
    name = name.split('-')[1]
    if fp_name == 'tpatf':
        fp_name2 = 'pharmacophore'
    elif fp_name == 'rdkDes':
        fp_name2 = 'rdkit descriptors'
    elif fp_name == 'volsurf':
        fp_name2 = 'volsurf'
    else:
        fp_name2 = 'fingerprint'

    with open(m, 'rb') as mod:
        model = pickle.load(mod)
        try:
            dictn[name+'-'+fp_name2]=model.get_best_params()
        except:
            print(name)
            pass
            
with open('best_hyper_parameters.json', 'w') as f:
    json.dump(dictn, f)
