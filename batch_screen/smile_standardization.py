import os,sys,re,time,argparse,logging
import tempfile, shutil
import rdkit
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import pickle
from glob import glob
import numpy as np
from rdkit import Chem
import argparse
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pandas as pd
from rdkit.Chem import SmilesMolSupplier, SDMolSupplier, SDWriter, SmilesWriter, MolStandardize, MolToSmiles, MolFromSmiles

class StandardSmiles:

    def Standardize(self, stdzr, remove_isomerism, molReader, molWriter):
        n_mol=0; 
        for mol in molReader:
            n_mol+=1
            molname = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
            logging.debug('%d. %s:'%(n_mol, molname))
            mol2 = self.StdMol(stdzr, mol, remove_isomerism)
            output = rdkit.Chem.MolToSmiles(mol2, isomericSmiles=True) if mol2 else None
            return output
    #############################################################################
    def MyNorms(self):
        norms = list(MolStandardize.normalize.NORMALIZATIONS)
        for i in range(len(norms)-1, 0, -1):
            norm = norms[i]
            if norm.name == "Sulfoxide to -S+(O-)-":
                del(norms[i])
        norms.append(MolStandardize.normalize.Normalization("[S+]-[O-] to S=O",
        "[S+:1]([O-:2])>>[S+0:1](=[O-0:2])"))
        logging.info("Normalizations: {}".format(len(norms)))
        return(norms)

    #############################################################################
    def MyStandardizer(self, norms):
        stdzr = MolStandardize.Standardizer(
            normalizations = norms,
            max_restarts = MolStandardize.normalize.MAX_RESTARTS,
            prefer_organic = MolStandardize.fragment.PREFER_ORGANIC,
            acid_base_pairs = MolStandardize.charge.ACID_BASE_PAIRS,
            charge_corrections = MolStandardize.charge.CHARGE_CORRECTIONS,
            tautomer_transforms = MolStandardize.tautomer.TAUTOMER_TRANSFORMS,
            tautomer_scores = MolStandardize.tautomer.TAUTOMER_SCORES,
            max_tautomers = MolStandardize.tautomer.MAX_TAUTOMERS
            )
        return(stdzr)

    #############################################################################
    def StdMol(self, stdzr, mol, remove_isomerism=False):
        smi = MolToSmiles(mol, isomericSmiles=(not remove_isomerism)) if mol else None
        mol_std = stdzr.standardize(mol) if mol else None
        smi_std = MolToSmiles(mol_std, isomericSmiles=(not remove_isomerism)) if mol_std else None
        logging.debug(f"{smi:>28s} >> {smi_std}")
        return(mol_std)

    #############################################################################
    def preprocess_smi(self, smi):
        norms = MolStandardize.normalize.NORMALIZATIONS
        test_smiles = [smi]
        test_label = [1] # dummy list
        temp_dir = tempfile.mkdtemp()
        df = pd.DataFrame(zip(test_smiles, test_label), columns=['SMILES', 'Label'])

        df.to_csv(temp_dir+'/temp_file.csv', index=False)

        try:
            molReader = SmilesMolSupplier(temp_dir+'/temp_file.csv', delimiter=',', smilesColumn=0, nameColumn=1, titleLine=True, sanitize=True)
            molWriter = SmilesWriter(temp_dir+'/temp_outfile.csv', delimiter=',', nameHeader='Name', includeHeader=True, isomericSmiles = (True), kekuleSmiles=False)
            stdzr = self.MyStandardizer(norms)
            stand_smiles = self.Standardize(stdzr, True, molReader, molWriter)
            shutil.rmtree(temp_dir)
            
            return stand_smiles
        except:
            print('No')
            return '' 
sd = StandardSmiles()
print(sd.preprocess_smi("CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@](C#N)([C@H](O)[C@@H]1O)C1=CC=C2N1N=CN=C2N)OC1=CC=CC=C1"))
