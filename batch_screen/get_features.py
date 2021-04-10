import sys,os,glob
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
import pickle
import tempfile, os
import shutil
from sklearn.preprocessing import LabelEncoder
from config import fpFunc_dict
import argparse
from sklearn.impute import SimpleImputer

class FeaturesGeneration:
    def __init__(self):
        self.fingerprints = []
    def get_fingerprints(self, df, model, fp_name, split, numpy_folder):

        smiles_list = df['SMILES_stand'].to_list()
        
        not_found = []
        for smi in smiles_list:
            try: 
                m = Chem.MolFromSmiles(smi)
            
                can_smi = Chem.MolToSmiles(m, True)
                fp = fpFunc_dict[fp_name](m)
                bit_array = np.asarray(fp)
                self.fingerprints.append(bit_array)
            except:
                not_found.append(smi)
                
                if fp_name == 'tpatf':
                    add = [np.nan for i in range(self.fingerprints[0].shape[1])]
                elif fp_name == 'rdkDes':
                    add = [np.nan for i in range(len(self.fingerprints[0]))]
                else:
                    add = [np.nan for i in range(len(self.fingerprints[0]))]
                tpatf_arr = np.array(add, dtype=np.float32)
                self.fingerprints.append(tpatf_arr) 
                
                pass
        
        if fp_name == 'rdkDes':
            X = np.array(self.fingerprints)
            ndf = pd.DataFrame.from_records(X)
            ndf.isnull().sum().sum()
            r, _ = np.where(df.isna())
            ndf.isnull().sum().sum()

            for col in ndf.columns:
                ndf[col].fillna(ndf[col].mean(), inplace=True)
            ndf.isnull().sum().sum()
            X = ndf.iloc[:,0:].values
            fp_array = ( np.asarray((X), dtype=object) )
            X = X.astype(np.float32)
            X = np.nan_to_num(X)
            rdkDes_scaler = pickle.load(open('../scalers/'+model+'-rdkDes_scaler.pkl', 'rb'))
            X = rdkDes_scaler.transform(X)

        else:
            fp_array = ( np.asarray((self.fingerprints), dtype=object) )
            X = np.vstack(fp_array).astype(np.float32)
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median.fit(X)  
            X = imp_median.transform(X)
        
#        Y = df['Label'].values
#        Y = Y.reshape(Y.shape[0],1)
#        Y = np.vstack(Y).astype(np.float32)
        final_array = X #np.concatenate((X, Y), axis=1)
        self.fingerprints = []
        return final_array
