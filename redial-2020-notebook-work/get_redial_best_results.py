# coding: utf-8
import pandas as pd, os, glob, sys, json

fp_list1 = ['ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6', 'lecfp4', 'lecfp6',\
        'lfcfp4', 'lfcfp6', 'maccs', 'hashap', 'hashtt', 'avalon', 'laval', 'rdk5', 'rdk6',\
        'rdk7']
fp_list2 = ['rdkDes']
fp_list3 = ['tpatf']
fp_list4 = ['volsurf']
fp_list = [fp_list1, fp_list2, fp_list3, fp_list4]
fp_type_name = ['fingerprints', 'rdkitDescriptors', 'pharmacophore', 'volsurf']
model_types = ['CoV1-PPE','CoV1-PPE_cs', 'MERS-PPE', 'MERS-PPE_cs', 'hCYTOX', '3CL', 'AlphaLISA', 'ACE2', 'CPE', 'TruHit', 'cytotox']

models = []

for f in glob.glob('reports_default/*.json'):
   file  = json.load(open(f, 'r'))
   models.extend(list(file.keys()))

final_results = {}
for k in range(len(fp_list)):
    for mt in model_types:
        max_f1 = 0
        max_model = ""
        fp_name = ""
        for fp in fp_list[k]:
            file = json.load(open('reports_default/'+fp+'-'+mt+'-'+'balanced_randomsplit7_70_15_15_te_results.json', 'r'))
            for m in models:
                if file[m]['valid']=='NaN':
                    continue
                else:
                    f1 = file[m]['valid']['F1_Score']
                    if f1>max_f1:
                        max_f1 = f1
                        max_model = m
                        fp_name = fp
        best_file = json.load(open('reports_default/'+fp_name+'-'+mt+'-'+'balanced_randomsplit7_70_15_15_te_results.json', 'r'))
        temp = best_file[max_model]
        temp['best_model']=max_model
        temp['best_features']=fp_name
        final_results[mt]= temp
        print(mt, fp_name, max_f1, max_model)
    
    if not os.path.isdir('best_default_results'):
        os.mkdir('best_default_results')
    with open('best_default_results/best_defaults_results_'+fp_type_name[k]+'.json', 'w') as re:
        json.dump(final_results, re)

