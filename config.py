######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import rdMolDescriptors
import tempfile, os
import shutil
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

# Function for generating TPATF features, using Mayachem tools
def get_tpatf(m):

    # Creates a temp folder
    temp_dir = tempfile.mkdtemp()

    # Compute 2D coordinates
    AllChem.Compute2DCoords(m)

    # Save sdf file
    w = Chem.SDWriter(os.path.join(temp_dir, "temp.sdf"))
    w.write(m)
    w.flush()

    try:
        # Path to perl script
        script_path = 'mayachemtools/bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl'
        command = "perl " + script_path + " -r " + os.path.join(temp_dir,"temp") + " --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(temp_dir, "temp.sdf")
        os.system(command)

        with open(os.path.join(temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"', '')
                    features = [int(i) for i in line.split(" ")]
    except:
        features = None

    # Delete the temporary directory
    shutil.rmtree(temp_dir)

    tpatf_arr = np.array(features, dtype=np.float32)
    tpatf_arr = tpatf_arr.reshape(1, tpatf_arr.shape[0])
    return tpatf_arr

LocInfo_dict =[
{
    "ToxDes": {
    "dataset_size": 1662,
    "actives": 831,
    "inactives": 831,
    "cohen_k_test": 0.36,
    "roc_auc": 0.68,
    "f1_score": 0.679,
    "Recall": 0.68,
    "accuracy": 0.68,
    "Precision": 0.682
    },
    "ToxFP": {
    "dataset_size": 1662,
    "actives": 831,
    "inactives": 831,
    "cohen_k_test": 0.392,
    "roc_auc": 0.696,
    "f1_score": 0.696,
    "Recall": 0.696,
    "accuracy": 0.696,
    "Precision": 0.698
    },
    "ToxTopo": {
    "dataset_size": 1662,
    "actives": 831,
    "inactives": 831,
    "cohen_k_test": 0.368,
    "roc_auc": 0.684,
    "f1_score": 0.684,
    "Recall": 0.684,
    "accuracy": 0.684,
    "Precision": 0.684
    },
    "ActFP": {
    "dataset_size": 736,
    "actives": 368,
    "inactives": 368,
    "cohen_k_test": 0.392,
    "roc_auc": 0.696,
    "f1_score": 0.695,
    "Recall": 0.696,
    "accuracy": 0.696,
    "Precision": 0.698
    },
    "ActDes": {
    "dataset_size": 680,
    "actives": 340,
    "inactives": 340,
    "cohen_k_test": 0.216,
    "roc_auc": 0.608,
    "f1_score": 0.606,
    "Recall": 0.608,
    "accuracy": 0.608,
    "Precision": 0.609
    },
    "ActTopo": {
    "dataset_size": 680,
    "actives": 340,
    "inactives": 340,
    "cohen_k_test": 0.294,
    "roc_auc": 0.647,
    "f1_score": 0.647,
    "Recall": 0.647,
    "accuracy": 0.647,
    "Precision": 0.647
    }
}
]

nbits = 1024
longbits = 16384

# dictionary
fpFunc_dict = {}
fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['tpatf'] = lambda m: get_tpatf(m)
fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)

long_fps = {'laval', 'lecfp4', 'lecfp6', 'lfcfp4', 'lfcfp6'}
fps_to_generate = ['lecfp4', 'lfcfp4', 'rdkDes', 'tpatf', 'rdk5', 'hashtt', 'avalon', 'laval', 'rdk7', 'ecfp4', 'hashap', 'lecfp6', 'maccs']

ModFileName_LoadedModel_dict = {}

