import requests
import string
BASE = "http://127.0.0.1:5000/"

input_urls = input("URLs (seperated by comma): ")
for i in input_urls.split(","):
    try:
        response = requests.get(BASE + "predict/", {"url":i})
        print(response.json())
    except:
        pass
