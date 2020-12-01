######################################################################################
# Author: Srijan Verma, BITS Pilani, India                                           #
# Code developed in Sirimulla Research Group (http://sirimullaresearchgroup.com/)    #
# University of Texas at El Paso, Tx, USA                                            #
# Last modified: 25/08/2020                                                          #
######################################################################################

from time import time
from time import sleep
from datetime import datetime
from requests import get
from random import randint
from IPython.core.display import clear_output
from warnings import warn
import json
import re
import csv
from datetime import datetime
import argparse
import pandas as pd
from urllib import parse
parse.quote('Cc1cc(N[S+](=O)([O-])c2ccc(N)cc2)no1')

# datetime object containing current date and time
now = datetime.now()
error_codes = [401, 400, 404]

# Ochem URL
'''
http://rest.ochem.eu/
http://rest.ochem.eu/predict?MODELID=536&SMILES=Cc1ccccc1
'''


def save_file(smi_dict, model_id, save_dir, i, res_code):
    save_path = save_dir + '/smi_' + str(i + 1) + '-response_code_' + str(res_code) \
                + '-model_id_' + str(model_id) + '.json'

    with open(save_path, 'w') as f:
        json.dump(smi_dict, f, indent=4)


def fetch_ochem(smiles, model_id, save_dir):
    requests = 0
    start_time = time()
    total_runtime = datetime.now()

    for i in range(len(smiles)):

        s_time = time()
        url_smi = parse.quote(smiles[i])
        smi_dict = {}
        smi_dict[smiles[i]] = {'results': -1, 'response_code': -1, 'time_taken': -1,
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
            print("====================================================")
            print("Total Request:{}; Frequency: {} request/s; Total Run Time: {}".format(requests,
                                                                                         requests / elapsed_time,
                                                                                         datetime.now() - total_runtime))
            #             clear_output(wait=True)

            print("Response Code: ", response.status_code)

            # Throw a warning for non-200 status codes
            if response.status_code in error_codes:
                smi_dict[smiles[i]].update({'results': json.loads(response.text),
                                            'response_code': int(response.status_code),
                                            'time_taken': round((time() - s_time), 3),
                                            'model_id': model_id, 'short_error': 'ERROR',
                                            'long_error': str(response.text)})

                save_file(smi_dict, model_id, save_dir, i, response.status_code)
                continue

            if response.status_code == 206 or response.status_code == 200:

                while (response.status_code == 206):

                    response = get("http://rest.ochem.eu/predict?MODELID={0}&SMILES={1}".format(model_id, url_smi))

                    # Pauses the loop between 1 - 2 seconds
                    sleep(randint(1, 2))

                    # If results are not ready, then continue
                    if response.text == 'not yet ready':
                        continue

                    # If error in results, then break
                    if response.status_code in error_codes or \
                            'Empty molecule provided!' in json.loads(response.text)['predictions'][0]['error']:
                        break

                if response.status_code == 200:
                    err_code = None
                else:
                    err_code = 'ERROR'

                smi_dict[smiles[i]].update({'results': json.loads(response.text),
                                            'response_code': int(response.status_code),
                                            'time_taken': round((time() - s_time), 3),
                                            'model_id': model_id, 'short_error': err_code})

                save_file(smi_dict, model_id, save_dir, i, response.status_code)
                continue


        except Exception as e:
            smi_dict[smiles[i]].update({'short_error': str(e.__class__.__name__),
                                        'long_error': str(e),
                                        'time_taken': round((time() - s_time), 3)})
            save_file(smi_dict, model_id, save_dir, i, 502)



#########################----MAIN FUNCTION BELOW-------###################################
def main():

    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--csv_file', type=str, default=None,
                        help='File containing SMILES column')
    parser.add_argument('--model_id', type=int, default=None,
                        help='MODELID for OCHEM')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Path to save results')
    args = parser.parse_args()

    if not (args.csv_file) or (args.csv_file == ''):
        parser.error('No input is given, add --csv_file')

    if args.csv_file:
        smiles = pd.read_csv(args.csv_file)['SMILES'].tolist()
        fetch_ochem(smiles, args.model_id, args.save_dir)

    else:
        parser.error('Argument Error')

if __name__ == "__main__":
    main()

#########################----DOCUMENTATION OF THIS .PY FILE GIVEN BELOW-------###################################
'''
python fetch_ochem_predictions.py --csv_file ../latest_v2.csv --model_id 536 --save_dir testing
'''