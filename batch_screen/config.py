#######################################################################
# Author: Srijan Verma                                                #
# Department of Pharmacy                                              #
# Birla Institute of Technology and Science, Pilani, India            #
# Last modified: 13/08/2020                                           #
#######################################################################

from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import rdMolDescriptors
import tempfile, os
import shutil
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import LabelEncoder

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
        script_path = '../../redial-2020/mayachemtools/bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl'
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
