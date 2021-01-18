import requests
import string
BASE = "http://127.0.0.1:5000/"

input_urls = input("URLs (seperated by comma): ")

url_list = []
for i in input_urls.split(","):
    url_list.append(i)

try:
    res = requests.post(BASE + 'predict/', json={"urls":url_list})
except:
    print("Failed to send request")