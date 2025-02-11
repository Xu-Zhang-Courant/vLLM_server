import requests
import threading
from transformers import AutoTokenizer
import pandas as pd
import json
import sys
import io
import random
import time

# the prompt files in csv format
df = pd.read_csv("/root/short_LLM_test_Q.csv", index_col = 0)

all_responses = []

# randomly select some questions from the list
question_list = [1, 6, 9, 10, 12, 17, 20, 25, 34, 37, 42, 43, 44, 48, 52, 56, 63, 69]

print("\n \n \n")


def send_request(text, idx):
    # send request to this url (server is here)
    url = "http://127.0.0.1:6006/generate"
    json_data = {"prompt": text, "idx": idx,"stream": False}

    response = requests.post(url, json = json_data)
    
    print(response.status_code)
    if response.status_code == 200:
        
        data_received = response.json()  # Use .json() to parse the JSON response
        print(data_received)
        all_responses.append(data_received)  # Collect the JSON response
        process_id = data_received['task_id']
    else:
        print("Failed to retrieve data")
    print()


def load_questions_into_threads(threads, question_list):
    question_indices = []
    for i in range(num_threads):
        question_indices.append(question_list.pop())
        which_thread_to_send = i
        which_question_to_send = question_indices[which_thread_to_send]
        data_to_send = df.iloc[which_question_to_send,0]
        idx = which_question_to_send
        print("*******************************************")
        print("The prompt was:")
        print(data_to_send)
        print("*******************************************")

        thread = threading.Thread(target=send_request, args=(data_to_send, idx))
        threads.append(thread)
        thread.start()

threads = []
num_threads = 3
while (len(question_list) != 0):        
    load_questions_into_threads(threads, question_list)
    for thread in threads:
        thread.join()


output_file_name_prefix = 'responses_' + "multithread_vLLM"

process_id = str(int(time.time()))[-7:-1]

with open('/root/select_Q/testdata' + output_file_name_prefix + "-" + process_id + '.json', 'w') as json_file:
    json.dump(all_responses, json_file)

# Optionally, save responses to a text file
with open('/root/select_Q/testdata' + output_file_name_prefix + "-" + process_id + '.txt', 'w') as txt_file:
    for response in all_responses:
        txt_file.write(f"{response}\n")

print("输出已保存到", '/root/test_results/testdata' + output_file_name_prefix + "-" + process_id + '.json')