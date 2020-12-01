######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from collections import OrderedDict
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
import pickle
from glob import glob
import numpy as np
from rdkit import Chem
import multiprocessing as mp
from time import time
from time import sleep
from requests import get
from random import randint
import json
from datetime import datetime
import argparse
from urllib import parse
import os, sys
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from pubchempy import Compound, get_compounds, get_synonyms
import tqdm
from tqdm import tqdm
from tqdm import tqdm_notebook
import time

s = time.time()

# Loading SMILES
print('Loading SMILES')

# Read appropriate csv file below
SMILES = pd.read_csv('drug_central_drugs.csv',index_col=0)['SMILES'].tolist()


def multi_preprocess_smi(smi):
    
    new_dict = {}
    
    try:
        # Filter 1- Convert to Canonical Smiles
        mol = Chem.MolFromSmiles(smi)
        can_smi = Chem.MolToSmiles(mol, True)

        # Filter 2- Remove salt
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(can_smi)
        res, deleted = remover.StripMolWithDeleted(mol, dontRemoveEverything=True)
        removed_salt_smi = Chem.MolToSmiles(res)

        # Filter 3- Remove Charge
        uncharger = rdMolStandardize.Uncharger()
        m = Chem.MolFromSmiles(removed_salt_smi)
        p = uncharger.uncharge(m)
        uncharged_smi = Chem.MolToSmiles(p)

        # Filter 4 - Standardize the tautomer
        clean_smi = MolStandardize.canonicalize_tautomer_smiles(uncharged_smi)
    
        new_dict[smi] = clean_smi

    except:
        new_dict[smi] = '-'
    
    return new_dict


test = SMILES[0:10]

CPU = mp.cpu_count()
pool = mp.Pool(CPU)

# First replace SMILES with test
query_ref_iterable = pool.map(multi_preprocess_smi, [(ref_smi) for ref_smi in test])

final_query_ref_dict = {}
[final_query_ref_dict.update(c) for c in query_ref_iterable]

e = time.time()

tot_time = e - s

print('Time Taken (in seconds): {}'.format(round(tot_time, 3)))

import json

with open('taut_clean_smi.json','w') as f:
    json.dump(final_query_ref_dict, f, indent=4)

print('DONE!!')


'''
For filtering smiles in parallel.
Usage = $python apply_all_filters.py
-> Read appropriate csv file
-> Change test to SMILES
'''

