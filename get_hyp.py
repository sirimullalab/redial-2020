# coding: utf-8
import glob
import os, sys
import pickle
import json
dictn = {}
for m in glob.glob('./saved_models/*.pkl'):
    name,_=os.path.splitext(os.path.basename(m))
    name = name.split('-')[0]
    with open(m, 'rb') as mod:
        model = pickle.load(mod)
        try:
            dictn[name]=model.get_best_params()
        except:
            print(name)
            pass
            
with open('hyper_parameters_new.json', 'w') as f:
    json.dump(dictn, f)
