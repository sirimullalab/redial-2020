######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

import rdkit
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from collections import OrderedDict
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import pickle
from glob import glob
import numpy as np
from rdkit import Chem
from config import LocInfo_dict, fpFunc_dict, long_fps, fps_to_generate, ModFileName_LoadedModel_dict
import multiprocessing as mp
from time import time
from time import sleep
from requests import get
from random import randint
import json
from datetime import datetime
import argparse
from urllib import parse
import os
import subprocess
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from pubchempy import Compound, get_compounds, get_synonyms
from func_timeout import func_timeout, FunctionTimedOut
import os,sys,re,time,argparse,logging
import pandas as pd
import rdkit, shutil
#import rdkit.Chem.AllChem
from rdkit.Chem import SmilesMolSupplier, SDMolSupplier, SDWriter, SmilesWriter, MolStandardize, MolToSmiles, MolFromSmiles
import tempfile

pubchem_time_limit = 30  # in seconds

ochem_api_time_limit = 20 # in seconds

def Standardize(stdzr, remove_isomerism, molReader, molWriter):
  n_mol=0; 
  for mol in molReader:
    n_mol+=1
    molname = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
    logging.debug('%d. %s:'%(n_mol, molname))
    mol2 = StdMol(stdzr, mol, remove_isomerism)
    output = rdkit.Chem.MolToSmiles(mol2, isomericSmiles=True) if mol2 else None
    return output
#############################################################################
def MyNorms():
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
def MyStandardizer(norms):
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
def StdMol(stdzr, mol, remove_isomerism=False):
  smi = MolToSmiles(mol, isomericSmiles=(not remove_isomerism)) if mol else None
  mol_std = stdzr.standardize(mol) if mol else None
  smi_std = MolToSmiles(mol_std, isomericSmiles=(not remove_isomerism)) if mol_std else None
  logging.debug(f"{smi:>28s} >> {smi_std}")
  return(mol_std)

#############################################################################
def preprocess_smi(smi):
    norms = MolStandardize.normalize.NORMALIZATIONS

    test_smiles = [smi]
    test_label = [1] # dummy list
    temp_dir = tempfile.mkdtemp()
    df = pd.DataFrame(zip(test_smiles, test_label), columns=['SMILES', 'Label'])

    df.to_csv(temp_dir+'/temp_file.csv', index=False)

    try:
        molReader = SmilesMolSupplier(temp_dir+'/temp_file.csv', delimiter=',', smilesColumn=0, nameColumn=1, titleLine=True, sanitize=True)

        molWriter = SmilesWriter(temp_dir+'/temp_outfile.csv', delimiter=',', nameHeader='Name',
        includeHeader=True, isomericSmiles = (True), kekuleSmiles=False)
        stdzr = MyStandardizer(norms)
        stand_smiles = Standardize(stdzr, True, molReader, molWriter)
        shutil.rmtree(temp_dir)
        
        return stand_smiles
    except:
        return None

class Similarity:

    def calculate_fp(self, fp_name, smiles):
        m = Chem.MolFromSmiles(smiles)
        return fpFunc_dict[fp_name](m)

    def load_dict(self, path):
        with open(path, 'rb') as file:
            _dict = pickle.load(file)
        return _dict

    def multiprocess_find_similarity(self, _query_fp, _ref_fp, _ref_smi):

        new_dict = {}
        tanimoto = round(TanimotoSimilarity(_query_fp, _ref_fp), 3)
        new_dict[_ref_smi] = {'tanimoto': tanimoto}
        return new_dict

    def get_top_values(self, data, _value='tanimoto', n=10, order=False):
        """Get top n similarities.

        Returns a dictionary or an `OrderedDict` if `order` is true.
        """
        top = sorted(data.items(), key=lambda x: float(x[1][_value]), reverse=True)[:n]
        if order:
            return OrderedDict(top)
        return dict(top)

    ##########################----INITIALIZE THE MODEL----#####################################
    def model_initialization(self, smi, three_cl=False):

        similarity_dict = {}
        # Use all CORES
        pool = mp.Pool(mp.cpu_count())

        # Calculate FP for query smi. NOTE: smi = query smiles
        try:
            query_fp = self.calculate_fp('ecfp4', smi)

        except:
            query_fp = None

        # If fingerprint could not be calculated, because of invalid smiles string
        if query_fp == None:
            similarity_dict[smi] = None
            pool.close()
            return similarity_dict

        # Else, fp is calculated. continue -->
        else:

            smi_all_dict = self.load_dict('smi_all_dict_updated_new_cleaning_with_3cl.pkl')

            final_query_ref_dict = {}
            ref_smi_list = list(smi_all_dict.keys())
            query_ref_iterable = pool.starmap(self.multiprocess_find_similarity,
                                              [(query_fp, smi_all_dict[ref_smi]['features']['ecfp4'][0], ref_smi) for
                                               ref_smi in ref_smi_list])

            [final_query_ref_dict.update(c) for c in query_ref_iterable]

            # get_top_values is a func, to get top 10 smi & tanimoto scores
            final_query_ref_dict = self.get_top_values(final_query_ref_dict, _value='tanimoto', n=10, order=True)

            # get additional info for top_ref_smi
            all_location_iter = []
            for _smi in final_query_ref_dict:
                final_query_ref_dict[_smi].update(smi_all_dict[_smi])

            # Removing 'features' keys, as it is no more required
            [final_query_ref_dict[_smi].pop('features', None) for _smi in final_query_ref_dict]

            # deleting redundant variables from memory
            del smi_all_dict, ref_smi_list, all_location_iter

            similarity_dict[smi] = final_query_ref_dict
            del final_query_ref_dict

            pool.close()

            return similarity_dict


class Predict:
    ##############################<TEST THE MODEL>#################################################
    def model_testing(self, opt, X_test, mod, target):

        scaler_loaded = '-'
        if target[-3:] == 'Des':

            scaler_loaded = target

            # Loading rdkDes scaler
            # print("SCALER LOADED: ", target)
            rdkDes_scaler = pickle.load(open('scalers/' + target + '-rdkDes_scaler.pkl', 'rb'))

            X = rdkDes_scaler.transform(X_test)

            # Replace nan, posinf, neginf with mean of row
            X_test = np.nan_to_num(X, nan=np.nanmean(X), posinf=np.nanmean(X), neginf=np.nanmean(X))
            #####################

        test_predictions = opt.predict(X_test)[0]

        if mod == 'PassiveAggressiveClassifier' or mod == 'SGDClassifier' or mod == 'LinearSVC':
            if test_predictions == 0.0:
                return 'inactive', None, scaler_loaded

            else:
                return 'active', None, scaler_loaded

        else:
            test_predictions_prob = opt.predict_proba(X_test)

            if test_predictions == 0.0:
                return 'inactive', str(round(test_predictions_prob[0][0], 2)), scaler_loaded

            else:
                return 'active', str(round(test_predictions_prob[0][1], 2)), scaler_loaded

    ##############################################################################################################################

    ##########################----LOAD THE MODEL----#####################################
    def load_model(self, model_file):

        target_fp_mod = os.path.splitext(os.path.basename(model_file))[0][0:-5]

        with open(model_file, 'rb') as file:
            opt = pickle.load(file)

        ModFileName_LoadedModel_dict[target_fp_mod] = opt

    ######################################################################################################

    ##########################----CALCULATE FEATURE----#####################################

    def CalculateFP(self, fp_name, smiles):

        m = Chem.MolFromSmiles(smiles)
        return fpFunc_dict[fp_name](m)

    ##########################################################################################

    ########################----MULTI-PROCESS FOR PREDICTION----######################################
    def multi_process(self, loaded_model, arr):

        output_dict = {}
        target = loaded_model.split('-')[0]
        fp_name = loaded_model.split('-')[1]
        mod = loaded_model.split('-')[2]

        if arr is None:
            output_dict[target] = {
                'prediction': '-',
                'probability': '-',
                'model': '-',
                'no_of_actives': '-',
                'feature': '-',
                'cohen_k_test': '-'
            }
        else:

            # Get the model
            opt = ModFileName_LoadedModel_dict[loaded_model]

            # Get predictions
            test_pred, test_pred_proba, scaler_loaded = self.model_testing(opt, arr, mod, target)

            output_dict[target] = {
                'prediction': test_pred,
                'probability': test_pred_proba,
                'model': mod,
                'feature': fp_name,
                'scaler_loaded': scaler_loaded
            }
            # Get additional info using below code line
            # output_dict[target].update(LocInfo_dict[0][target])

        return output_dict

    ##############################----MULTI-PROCESS_FPs----##################################
    def multi_process_fp(self, _smi, _fp):

        fpName_array_dict = {}

        if _fp in long_fps:
            _dtype = np.float16

        else:
            _dtype = np.float32

        try:

            # Below if for tpatf / volsurf
            if _fp == 'tpatf' or _fp == 'volsurf':
                X = self.CalculateFP(_fp, _smi)

            # Below if rdkDescriptor
            elif 'rdkDes' in _fp:

                # Hard Coded fp name below
                fp = self.CalculateFP('rdkDes', _smi)
                fp = np.asarray(fp)
                fp = fp.reshape(1, 200)
                X = np.array(fp)
                X = np.vstack(X).astype(_dtype)

            else:
                fp = self.CalculateFP(_fp, _smi)
                bits = fp.ToBitString()
                bits = [bits]
                X = np.array([(np.fromstring(fp, 'u1') - ord('0')) for fp in (bits)], dtype=_dtype)

        except:
            X = None
            pass

        fpName_array_dict[_fp] = X
        return fpName_array_dict

    ##########################----INITIALIZE THE MODEL----#####################################

    def model_initialization(self, smi_list):
        dict_all = {}

        all_mod_files = sorted(glob('saved_models/*.pkl'))

        # Loop over and Load all models in memory and store in a dict--> key = model_file_name ; value = model
        for i in range(len(all_mod_files)):
            self.load_model(all_mod_files[i])

        # Use all CORES
        pool = mp.Pool(mp.cpu_count())

        # Loop over list of SMILES
        for j in range(len(smi_list)):

            # If empty string, then save None
            if smi_list[j] == '':
                dict_all[smi_list[j]] = None
                continue

            # Multi processing for generation of 2 features (for j th smiles)
            final_result_fp = {}
            result_fp = pool.starmap(self.multi_process_fp, [(smi_list[j], k) for k in fps_to_generate])

            # final_result_fp is a dict, for j th smiles, having 2 FPs--> key = fp_name ; value = array
            for e in result_fp:
                final_result_fp.update(e)

            # If all features are not none, for j th smiles, then predict
            if any(x is not None for x in final_result_fp.values()):
                result = pool.starmap(self.multi_process, [(k, final_result_fp[k.split('-')[1]]) for k in
                                                           list(ModFileName_LoadedModel_dict.keys())])

                final_result = {}
                for d in result:
                    final_result.update(d)

                dict_all[smi_list[j]] = final_result

            # If all features are None, then save None for j th smiles
            else:
                dict_all[smi_list[j]] = None
                continue

        pool.close()
        return dict_all


#########################----GET OCHEM API RESULTS-------###################################
class OchemAPIResults:
    # Ochem URL
    '''
    http://rest.ochem.eu/
    http://rest.ochem.eu/predict?MODELID=536&SMILES=Cc1ccccc1
    '''

    def get_ochem_model_results(self, smiles, model_id):
        try:
            d = func_timeout(ochem_api_time_limit, self.fetch_ochem, args=(smiles, model_id))
            if d[smiles]['response_code'] == 200:
                if model_id == 535:  # logp
                    _val = str(d[smiles]['results']['logPow']['value'])
                    return _val
                elif model_id == 536:  # logs
                    _val = str(d[smiles]['results']['Aqueous Solubility']['value'])
                    return _val

            else:
                return '-'
        except:
            return '-'

    def save_file(self, smi_dict, model_id, save_dir, i, res_code):
        save_path = save_dir + '/smi_' + str(i + 1) + '-response_code_' + str(res_code) \
                    + '-model_id_' + str(model_id) + '.json'

        with open(save_path, 'w') as f:
            json.dump(smi_dict, f, indent=4)

    def fetch_ochem(self, smiles, model_id, save_dir=None):

        # datetime object containing current date and time
        now = datetime.now()
        error_codes = [401, 400, 404]
        requests = 0
        start_time = time()
        total_runtime = datetime.now()
        smi_dict = {}

        s_time = time()
        # SMILES needs to be of HTML format! That's why below line exists-->
        url_smi = parse.quote(smiles)
        smi_dict[smiles] = {'results': -1, 'response_code': -1, 'time_taken': -1,
                            'model_id': -1, 'short_error': -1, 'long_error': -1}

        #         if i % 10 == 0:
        #             print('sleeping for 1 min....')
        #             sleep(randint(60, 80))

        try:

            #######<GET RESPONSE/>#######
            response = get("http://rest.ochem.eu/predict?MODELID={0}&SMILES={1}".format(model_id, url_smi))

            # Monitor the frequency of requests
            requests += 1

            # Pauses the loop between 2 - 4 seconds and marks the elapsed time
            sleep(randint(2, 4))
            current_time = time()
            elapsed_time = current_time - start_time
            print("===================<OchemAPI_RESPONSE>========================")
            print("Total Request:{}; Frequency: {} request/s; Total Run Time: {}".format(requests,
                                                                                         requests / elapsed_time,
                                                                                         datetime.now() - total_runtime))
            #             clear_output(wait=True)

            print("Response Code: ", response.status_code)

            # Throw a warning for non-200 status codes
            if response.status_code in error_codes:
                smi_dict[smiles].update({'results': json.loads(response.text),
                                         'response_code': int(response.status_code),
                                         'time_taken': round((time() - s_time), 3),
                                         'model_id': model_id, 'short_error': 'ERROR',
                                         'long_error': str(response.text)})

                return smi_dict
                # save_file(smi_dict, model_id, save_dir, i, response.status_code)

            if response.status_code == 206 or response.status_code == 200:
                while (response.status_code == 206):
                    response = get("http://rest.ochem.eu/predict?MODELID={0}&SMILES={1}".format(model_id, url_smi))

                    # Pauses the loop between 1 - 2 seconds
                    sleep(randint(1, 2))

                    # If results are not ready, then continue
                    if response.text == 'not yet ready':
                        print('ochem api results --> not yet ready')
                        continue

                    # If error in results, then break
                    if response.status_code in error_codes:
                        break

                if response.status_code == 200:
                    err_code = None
                else:
                    err_code = 'ERROR'

                smi_dict[smiles].update({'results': json.loads(response.text),
                                         'response_code': int(response.status_code),
                                         'time_taken': round((time() - s_time), 3),
                                         'model_id': model_id, 'short_error': err_code})
                return smi_dict

        except Exception as e:
            smi_dict[smiles].update({'short_error': str(e.__class__.__name__),
                                     'long_error': str(e),
                                     'time_taken': round((time() - s_time), 3)})
            return smi_dict

#########<OCHEM ALOGPS CALCULATIONS [NOTE: can only be EXECUTED FROM VIA A LINUX MACHINE!]>#########
# USE DOCKER FOR BELOW TASK -->
class OchemToolALOGPS:
    def calculate_alogps(self, smi):

        cmd = ['./alogps-linux','--smiles', smi]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, error = p.communicate()
        return out, error


###############<FETCH PHYSICO CHEMICAL PROPERTIES>###########################
# Takes input as smiles
class FetchPhysicoProperty:

    # Get molecular weight of smiles string
    def get_molecular_wt(self, smi):

        try:
            m = Chem.MolFromSmiles(smi)
            return round(Descriptors.MolWt(m), 2)

        except:
            return '-'

    # Get molecular formula of smiles string
    def get_molecular_formula(self, smi):

        try:
            m = Chem.MolFromSmiles(smi)
            return CalcMolFormula(m)

        except:
            '-'
################################################################################

###############<FETCH ATTRIBUTES FROM CHEMICAL DATABASES, USING APIS / OTHER>###########################
# Takes input as smiles
class FetchChemoDB:

    # Convert to Canonical Smiles
    def get_canonical(self, smi):

        try:
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol, True)
            return can_smi

        except:
            return None

    # Fetch Pubchem results
    def fetch_pubchem(self, smi):

        can_smi = self.get_canonical(smi)
        if can_smi == None:
            return '-', '-'

        try:
            # func_timeout runs for a certain time period. If results not returned in that time, it breaks
            # refer - https://pypi.org/project/func-timeout/
            r = func_timeout(pubchem_time_limit, get_compounds, args=(smi, 'smiles'))
            # r = get_compounds(smi, 'smiles')
            _cid = r[0].cid
            return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + str(_cid), _cid

        except:
            return '-', '-'

    # Fetch DrugCentral results
    def fetch_drug_central(self, smi, _input):

        can_smi = self.get_canonical(smi)
        if can_smi == None:
            return '-', '-'

        # Read csv
        #df = pd.read_csv('drug_central_drugs.csv')
        df = pd.read_csv('drug_central_drugs-stand.csv')
        ### added by GK ###
        dc_dictn = dict(zip(df.ID, df.INN_cleaned))
        dc_dictn_inn = dict(zip(df.INN_cleaned, df.Canonical_Smiles))
        ##################-----------################

        try:
            # Check if query canonical smi matches with canonical smi in drugCentral db
            dc_id = df[df.Canonical_Smiles == can_smi]['ID'].values[0]
            dc_name = dc_dictn[dc_id] # added by gK
            dc_smiles_stand = dc_dictn_inn[dc_name]
            return 'http://drugcentral.org/drugcard/' + str(dc_id), dc_id, dc_name, dc_smiles_stand # dc_name added by gk

        except:
            try:
                # Convert to string
                _input = str(_input)

                # Convert to lowercase
                _input = _input.lower()

                # Remove leading and trailing spaces
                _input = _input.strip()

                # Matching query drug_name with that present in drugCentral db
                dc_id = df[df.INN_cleaned == _input]['ID'].values[0]
                dc_name = dc_dictn[dc_id] # added by Gk
                dc_smiles_stand = dc_dictn_inn[dc_name]
                return 'http://drugcentral.org/drugcard/' + str(dc_id), dc_id, dc_name, dc_smiles_stand # dc_name, dc_smiles_stand added by gk

            except:
                return '-', '-', '-', '-' # added '-' by gk
################################################################################

###############<CHECK INPUT TYPE>###########################
class CheckInput:

    # Convert to Canonical Smiles
    def get_canonical(self, smi):

        try:
            if len(smi) == 0:
                return None

            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol, True)
            return can_smi

        except:
            return None

    def check_input(self, _input):

        smi_flag = False
        drug_name_flag = False
        pubchem_cid_flag = False

        # First, check if canonical
        can_smi = self.get_canonical(_input)

        if can_smi != None:
            smi_flag = True
            try:
                drug_name = func_timeout(pubchem_time_limit, get_synonyms, args=(can_smi, 'smiles'))

                if len(drug_name[0]['Synonym']) == 1:
                    drug_name = str(drug_name[0]['Synonym'][0])
                else:
                    drug_name = str(' | '.join(drug_name[0]['Synonym'][0:2]))
            except:
                drug_name = '-'
            return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

        else:

            # Convert to string
            _input = str(_input)

            # Convert to lowercase
            _input = _input.lower()

            # Remove leading and trailing spaces
            _input = _input.strip()

            ######<CHECK IF PUBCHEM CID>######
            try:
                # Check if it is a PubChem CID
                r = func_timeout(pubchem_time_limit, get_compounds, args=(_input, 'cid'))

                # Get canonical smiles
                can_smi = r[0].canonical_smiles
                can_smi = self.get_canonical(can_smi)

                if can_smi != None:
                    pubchem_cid_flag = True

                    try:
                        drug_name = func_timeout(pubchem_time_limit, get_synonyms, args=(can_smi, 'smiles'))
                        if len(drug_name[0]['Synonym']) == 1:
                            drug_name = str(drug_name[0]['Synonym'][0])
                        else:
                            drug_name = str(' | '.join(drug_name[0]['Synonym'][0:2]))
                    except:
                        drug_name = '-'
                    return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

                else:
                    smi_flag = False
                    pubchem_cid_flag = False
                    drug_name = '-'

            except:
                smi_flag = False
                pubchem_cid_flag = False
                can_smi = None
                drug_name = '-'

                ######<CHECK IF DRUG NAME>######
            try:

                # Remove multiple spaces from between words
                _input = " ".join(_input.split())

                # Check if name present in pubchem
                r = func_timeout(pubchem_time_limit, get_compounds, args=(_input, 'name'))

                # Get canonical smiles
                can_smi = r[0].canonical_smiles
                can_smi = self.get_canonical(can_smi)

                if can_smi != None:
                    drug_name_flag = True

                    try:
                        drug_name = func_timeout(pubchem_time_limit, get_synonyms, args=(can_smi, 'smiles'))
                        if len(drug_name[0]['Synonym']) == 1:
                            drug_name = str(drug_name[0]['Synonym'][0])
                        else:
                            drug_name = str(' | '.join(drug_name[0]['Synonym'][0:2]))
                    except:
                        drug_name = '-'
                    return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

                else:
                    smi_flag = False
                    drug_name_flag = False
                    drug_name = '-'
                    return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

            except:
                smi_flag = False
                drug_name_flag = False
                can_smi = None
                drug_name = '-'
                return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

################################################################################

#########################----MAIN FUNCTION BELOW-------###################################
def main():

    # Calculate start time
    start_time = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', action='store', dest='smiles', required=False, type=str, help='SMILES string')
    args = parser.parse_args()

    if not (args.smiles) or (args.smiles == ''):
        parser.error('No input is given, add --smiles')

    if args.smiles:

        smi = preprocess_smi(args.smiles)
        if smi is not None:
            p = Predict()
            s = Similarity()
            o = OchemResults()

        else:
            parser.error('Invalid Smiles, add correct SMILES format')


    # Calculating end time
    end_minus_start_time = ((time() - start_time))
    print("RUNTIME: {:.3f} seconds".format(end_minus_start_time))  # Calculating end time

if __name__ == "__main__":
    main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################

'''
    1. Example command: python3 run_script.py --smiles  "CCCCO"
    2. Takes one argument (SMILES string) and returns a top_n_smi_similarity dict and prediction_dict
    3. current runtime for this script, on 4 cores , intel i5 cpu = ~1.5sec
'''
