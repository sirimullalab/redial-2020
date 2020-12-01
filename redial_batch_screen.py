######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

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
import argparse
import os
import pandas as pd
import json
import os
from datetime import datetime
from time import time
import argparse
from urllib import parse
from requests import get
from time import sleep
from random import randint
import json
from func_timeout import func_timeout, FunctionTimedOut

pubchem_time_limit = 10  # in seconds
ochem_api_time_limit = 20 # in seconds

#
def preprocess_smi(smi):

    # Filter 1- Convert to Canonical Smiles
    try:
        mol = Chem.MolFromSmiles(smi)
        can_smi = Chem.MolToSmiles(mol, True)
    except:
        return None

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

    return clean_smi

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

class Predict:
    ##############################<TEST THE MODEL>#################################################
    def model_testing(self, opt, X_test, mod):

        test_predictions = opt.predict(X_test)[0]

        if mod == 'PassiveAggressiveClassifier' or mod == 'SGDClassifier' or mod == 'LinearSVC':
            if test_predictions == 0.0:
                return 'inactive', None

            else:
                return 'active', None

        else:
            test_predictions_prob = opt.predict_proba(X_test)

            if test_predictions == 0.0:
                return 'inactive', str(round(test_predictions_prob[0][0], 2))

            else:
                return 'active', str(round(test_predictions_prob[0][1], 2))

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
                'feature': '-'
            }
        else:

            # Get the model
            opt = ModFileName_LoadedModel_dict[loaded_model]

            # Get predictions
            test_pred, test_pred_proba = self.model_testing(opt, arr, mod)

            output_dict[target] = {
                'prediction': test_pred,
                'probability': test_pred_proba,
                'model': mod,
                'feature': fp_name
            }
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
            elif _fp == 'rdkDes':

                fp = self.CalculateFP(_fp, _smi)
                fp = np.asarray(fp)
                fp = fp.reshape(1, 200)
                X = np.array(fp)
                X = np.vstack(X).astype(_dtype)

                # Loading rdkDes scaler
                rdkDes_scaler = pickle.load(open('rdkDes_scaler.pkl', 'rb'))
                # rdkDes_scaler = joblib.load('rdkDes_scaler.save')

                X = rdkDes_scaler.transform(X)

                # Replace nan, posinf, neginf with mean of row
                X = np.nan_to_num(X, nan=np.nanmean(X), posinf=np.nanmean(X), neginf=np.nanmean(X))

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

    def model_initialization(self, smi_list, USE_OCHEM_API=False):
        dict_all = {}

        all_mod_files = sorted(glob('saved_models/*.pkl'))

        # Loop over and Load all models in memory and store in a dict--> key = model_file_name ; value = model
        for i in range(len(all_mod_files)):
            self.load_model(all_mod_files[i])

        # Use all CORES
        #       mp.cpu_count() = total CPUs to use
        #        pool = mp.Pool(mp.cpu_count())

        # Custom no. of CORES
        CORES_TO_USE = 4 # OR mp.cpu_count()
        pool = mp.Pool(CORES_TO_USE)

        # Loop over list of SMILES
        for j in range(len(smi_list)):

            # If empty string, then save None
            if smi_list[j] == '':
                dict_all[smi_list[j]] = None
                continue

            # Preprocessing the smiles
            processed_smiles = preprocess_smi(smi_list[j])

            # Multi processing for generation of 2 features (for j th smiles)
            final_result_fp = {}
            result_fp = pool.starmap(self.multi_process_fp, [(processed_smiles, k) for k in fps_to_generate])

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

            ############<CONSENSUS MODEL CODE BELOW>########################
            try:
                consensus_prediction_results = {}
                consensus_prediction_results[processed_smiles] = {}
                act_labels = []
                act_proba_1 = []
                act_proba_0 = []
                tox_labels = []
                tox_proba_1 = []
                tox_proba_0 = []
                truhit_labels = []
                truhit_proba_1 = []
                truhit_proba_0 = []
                alpha_lisa_and_truhit_labels = []
                alpha_lisa_and_truhit_proba_1 = []
                alpha_lisa_and_truhit_proba_0 = []
                alpha_lisa_labels = []
                alpha_lisa_proba_1 = []
                alpha_lisa_proba_0 = []
                ace2_enzymatic_labels = []
                ace2_enzymatic_proba_1 = []
                ace2_enzymatic_proba_0 = []
                three_cl_enzymatic_labels = []
                three_cl_enzymatic_proba_1 = []
                three_cl_enzymatic_proba_0 = []

                for key, val in dict_all[smi_list[j]].items():
                    if dict_all[smi_list[j]][key]['prediction'] == 'active' and 'Act' in key:
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        act_proba_1.append(proba_1)
                        act_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        act_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and 'Act' in key:
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        act_proba_1.append(proba_1)
                        act_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        act_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'active' and 'Tox' in key:
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        tox_proba_1.append(proba_1)
                        tox_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        tox_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and 'Tox' in key:
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        tox_proba_1.append(proba_1)
                        tox_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        tox_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    #######################<HACK BENEATH, ALERT!>############################
                    elif dict_all[smi_list[j]][key]['prediction'] == 'active' and (key == 'TruHitDes' or key == 'TruHitTopo' or key == 'TruHitFP'):
                        # FLIPPING THE below 2 lines
                        proba_1 = round(1 - proba_1, 2)
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        #######################################

                        truhit_proba_1.append(proba_1)
                        truhit_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})

                        # Flipping PREDICTION, below 2 lines
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        truhit_labels.append('INACTIVE')
                        #######################################
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and (key == 'TruHitDes' or key == 'TruHitTopo' or key == 'TruHitFP'):
                        # FLIPPING THE below 2 lines
                        proba_0 = round(1 - proba_0, 2)
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        #######################################

                        truhit_proba_1.append(proba_1)
                        truhit_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})

                        # Flipping PREDICTION, below 2 lines
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        truhit_labels.append('ACTIVE')
                        #######################################
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'active' and (key == 'AlphaLisaAndTruHitDes' or key == 'AlphaLisaAndTruHitTopo' or key == 'AlphaLisaAndTruHitFP'):
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        alpha_lisa_and_truhit_proba_1.append(proba_1)
                        alpha_lisa_and_truhit_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        alpha_lisa_and_truhit_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and (key == 'AlphaLisaAndTruHitDes' or key == 'AlphaLisaAndTruHitTopo' or key == 'AlphaLisaAndTruHitFP'):
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        alpha_lisa_and_truhit_proba_1.append(proba_1)
                        alpha_lisa_and_truhit_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        alpha_lisa_and_truhit_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'active' and (key == 'AlphaLisaDes' or key == 'AlphaLisaTopo' or key == 'AlphaLisaFP'):
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        alpha_lisa_proba_1.append(proba_1)
                        alpha_lisa_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        alpha_lisa_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and (key == 'AlphaLisaDes' or key == 'AlphaLisaTopo' or key == 'AlphaLisaFP'):
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        alpha_lisa_proba_1.append(proba_1)
                        alpha_lisa_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        alpha_lisa_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'active' and (key == 'ACE2EnzymaticDes' or key == 'ACE2EnzymaticTopo' or key == 'ACE2EnzymaticFP'):
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        ace2_enzymatic_proba_1.append(proba_1)
                        ace2_enzymatic_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        ace2_enzymatic_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and (key == 'ACE2EnzymaticDes' or key == 'ACE2EnzymaticTopo' or key == 'ACE2EnzymaticFP'):
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        ace2_enzymatic_proba_1.append(proba_1)
                        ace2_enzymatic_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        ace2_enzymatic_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'active' and (key == '3CLEnzymaticDes' or key == '3CLEnzymaticTopo' or key == '3CLEnzymaticFP'):
                        proba_1 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_0 = round(1 - proba_1, 2)
                        three_cl_enzymatic_proba_1.append(proba_1)
                        three_cl_enzymatic_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'ACTIVE'})
                        three_cl_enzymatic_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                    elif dict_all[smi_list[j]][key]['prediction'] == 'inactive' and (key == '3CLEnzymaticDes' or key == '3CLEnzymaticTopo' or key == '3CLEnzymaticFP'):
                        proba_0 = float(dict_all[smi_list[j]][key]['probability'])
                        proba_1 = round(1 - proba_0, 2)
                        three_cl_enzymatic_proba_1.append(proba_1)
                        three_cl_enzymatic_proba_0.append(proba_0)
                        dict_all[smi_list[j]][key].update({'predict_proba_0': str(proba_0)})
                        dict_all[smi_list[j]][key].update({'predict_proba_1': str(proba_1)})
                        dict_all[smi_list[j]][key].update({'prediction': 'INACTIVE'})
                        three_cl_enzymatic_labels.append(dict_all[smi_list[j]][key]['prediction'])
                        continue

                consensus_prediction_results[processed_smiles]['Activity Model'] = {'prediction': max(act_labels, key=act_labels.count)}
                consensus_prediction_results[processed_smiles]['Toxicity Model'] = {'prediction': max(tox_labels, key=tox_labels.count)}
                consensus_prediction_results[processed_smiles]['TruHit Model'] = {'prediction': max(truhit_labels, key=truhit_labels.count)}
                consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model'] = {'prediction': max(alpha_lisa_and_truhit_labels, key=alpha_lisa_and_truhit_labels.count)}
                consensus_prediction_results[processed_smiles]['AlphaLisa Model'] = {'prediction': max(alpha_lisa_labels, key=alpha_lisa_labels.count)}
                consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'] = {'prediction': max(ace2_enzymatic_labels, key=ace2_enzymatic_labels.count)}
                consensus_prediction_results[processed_smiles]['3CL Enzymatic Model'] = {'prediction': max(three_cl_enzymatic_labels, key=three_cl_enzymatic_labels.count)}

                if consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['Toxicity Model'].update({'probability': str(round(sum(tox_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['Toxicity Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['Toxicity Model'].update({'probability': str(round(sum(tox_proba_1) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['Activity Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['Activity Model'].update({'probability': str(round(sum(act_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['Activity Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['Activity Model'].update({'probability': str(round(sum(act_proba_1) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['TruHit Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['TruHit Model'].update({'probability': str(round(sum(truhit_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['TruHit Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['TruHit Model'].update({'probability': str(round(sum(truhit_proba_1) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model'].update({'probability': str(round(sum(alpha_lisa_and_truhit_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['AlphaLisa And TruHit Model'].update({'probability': str(round(sum(alpha_lisa_and_truhit_proba_1) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['AlphaLisa Model'].update({'probability': str(round(sum(alpha_lisa_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['AlphaLisa Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['AlphaLisa Model'].update({'probability': str(round(sum(alpha_lisa_proba_1) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'].update({'probability': str(round(sum(ace2_enzymatic_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'].update({'probability': str(round(sum(ace2_enzymatic_proba_1) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['3CL Enzymatic Model']['prediction'] == 'INACTIVE':
                    consensus_prediction_results[processed_smiles]['3CL Enzymatic Model'].update({'probability': str(round(sum(three_cl_enzymatic_proba_0) / 3, 2))})

                if consensus_prediction_results[processed_smiles]['3CL Enzymatic Model']['prediction'] == 'ACTIVE':
                    consensus_prediction_results[processed_smiles]['3CL Enzymatic Model'].update({'probability': str(round(sum(three_cl_enzymatic_proba_1) / 3, 2))})
                ################################################################################################
                # Selecting the best model by looking at val/test results ; That is, below models DO NOT have consensus predictions!!!
                consensus_prediction_results[processed_smiles]['AlphaLisa Model'] = dict_all[smi_list[j]]['AlphaLisaDes']
                consensus_prediction_results[processed_smiles]['ACE2 Enzymatic Model'] = dict_all[smi_list[j]]['ACE2EnzymaticDes']
                ################################################################################################

                dict_all[smi_list[j]]['consensus_prediction_results'] = consensus_prediction_results[processed_smiles]

                if USE_OCHEM_API:
                    #########<OCHEM ALOGPS API CALCULATIONS>#########
                    try:
                        ochem_api_ob = OchemAPIResults()
                        logp = ochem_api_ob.get_ochem_model_results(processed_smiles, 535)  # logp
                        logs = ochem_api_ob.get_ochem_model_results(processed_smiles, 536)  # logs

                    except:
                        logp = '-'
                        logs = '-'
                    ######################################

                    dict_all[smi_list[j]]['logp'] = logp
                    dict_all[smi_list[j]]['logs'] = logs

            except Exception as e:
                print(e)
                dict_all[smi_list[j]]['consensus_prediction_results'] = None

            # Save processed smiles -->
            dict_all[smi_list[j]]['processed_query_smiles'] = processed_smiles

        pool.close()

        return dict_all


#########################----MAIN FUNCTION BELOW-------###################################
def main():

    # Calculate start time
    start_time = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', dest='input_path', required=False, type=str, help='A csv file having a column as "SMILES"')
    parser.add_argument('--output_path', action='store', dest='output_path', required=False, type=str, help='Path to store json file')
    parser.add_argument('--calculate_alogps', action='store', dest='calculate_alogps', required=False, type=int, default=0, help='For calculating ALOGPS using OCHEM API. If set to "0", then ALOGPS will NOT be calculated')
    args = parser.parse_args()

    if not (args.input_path) or (args.input_path == ''):
        parser.error('No input is given, add --input_path')

    if args.input_path:
        smi_list = pd.read_csv(args.input_path)['SMILES'].tolist()
        # smi_list = [sm for sm in args.smiles.split(',')]

        if len(smi_list) != 0:
            p = Predict()
            # pprint.pprint(p.model_initialization(smi_list))
            dict_all_ = p.model_initialization(smi_list, USE_OCHEM_API=args.calculate_alogps)
            out_name = os.path.splitext(os.path.basename(args.input_path))[0]

            with open(args.output_path + '/'+out_name+'_results.json', 'w') as f:
                json.dump(dict_all_, f, indent=4)

        else:
            parser.error('SMILES list is empty')

    # Calculating end time
    end_minus_start_time = ((time() - start_time))
    print("RUNTIME: {:.3f} seconds".format(end_minus_start_time))  # Calculating end time

if __name__ == "__main__":
    main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################

'''
    1. Example command: python3 redial_batch_screen.py --input_path test.csv --output_path ./ --calculate_alogps 0
    2. Takes one argument (path to csv file, containining "SMILES" as column) and returns a dict
    3. current runtime for this script, on 4 cores ,for 1 SMILES, intel i5 cpu = ~1.5sec
'''
