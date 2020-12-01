######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

from rdkit.Chem import MolStandardize
import pandas as pd
uncharged_smi = pd.read_csv('neutralized_charges_smi_all_filter3.csv',index_col=0)['uncharged_smi'].tolist()

def multiprocess_tautomer(_ref_smi):
    new_dict = {}
    taut_smi = MolStandardize.canonicalize_tautomer_smiles(_ref_smi)
    new_dict[_ref_smi] = taut_smi
    print(new_dict)
    return new_dict


import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
test = uncharged_smi[0:10]
# query_ref_iterable = pool.starmap(multiprocess_find_similarity, [(query_fp, smi_all_dict[ref_smi]['features']['lecfp6'][0], ref_smi) for ref_smi in ref_smi_list])
query_ref_iterable = pool.map(multiprocess_tautomer, [(ref_smi) for ref_smi in test])

final_query_ref_dict = {}
[final_query_ref_dict.update(c) for c in query_ref_iterable]

import json

with open('taut.json','w') as f:
    json.dump(final_query_ref_dict,f,indent=4)

print('DONE!!')