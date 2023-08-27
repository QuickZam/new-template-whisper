from potassium import Potassium, Request, Response
from stable_whisper import modify_model 
from pytube import YouTube
from logger import logger 
from datetime import timedelta
import torch, whisper, os, base64, urllib.request, json
import numpy as np 
import base64
import requests, sys


app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = whisper.load_model("large-v1")
    modify_model(model)
   
    context = {
        "model": model
    }

    return context

    

# @app.handler runs for every call
# @app.handler("/")
# def handler(context: dict, request: Request) -> Response:
#     link = request.json.get("link")
#     model = context.get("model")
#     logger.info(f"Inputs: link=> {link}")

#     if 'tinyurl' in link: 
#         path = urllib.request.urlretrieve(link, f"{link.split('/')[-1]}.mp4")[0]

#     else:
#         yt = YouTube(link)
#         path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
#         logger.info("It's youtube video!! ")

#     # translate_options = dict(task="translate", suppress_silence=True, ts_num=16, lower_quantile=0.05, lower_threshold=0.1)4
#     translate_options = dict(task="translate",max_initial_timestamp=None)
#     result = model.transcribe(path, **translate_options)
#     result = result.to_dict()

#     logger.info("Prediction done!!"); logger.info(f"Type of the result: {type(result)}") 
#     logger.info(f"Original Output: \n{result}")

#     os.remove(path)

#     return Response(
#         json = {"outputs": result}, 
#         status=200
#     )


@app.background("/background")
def handler(context: dict, request: Request) -> Response:

    #Grabbing the inputs!! 
    link = request.json.get("link")
    email = request.json.get("email")
    youtube_title = request.json.get("youtube_title")
    url = request.json.get("url")
    headers = request.json.get("headers")

    #Grabbing the model
    model = context.get("model")

    try: 

        if 'tinyurl' in link: 
            path = urllib.request.urlretrieve(link, f"{link.split('/')[-1]}.mp4")[0]

        else:
            yt = YouTube(link)
            path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
            logger.info("It's youtube video!! ")

        # translate_options = dict(task="translate", suppress_silence=True, ts_num=16, lower_quantile=0.05, lower_threshold=0.1)4
        translate_options = dict(task="translate",max_initial_timestamp=None)
        result = model.transcribe(path, **translate_options)
        result = result.to_dict()

        outs = result
        all_prob = [  np.exp(i["avg_logprob"]) * 100 for i in outs['segments']]
        all_prob = ','.join([str(i) for i in all_prob])

        out = create_subtitle(outs)

        logger.info("The output is created and it's preparing to send to bubble io!")

        mp3 = base64.b64encode(bytes(str(out), 'utf-8'))
        payload={'file': mp3, 'Email': email, 'youtube_title': youtube_title, 'status':'Success', 'confidence' : all_prob }  

        logger.info("Payload is Ready! ")

        response = requests.request("PATCH", url, headers=headers, data=payload)

        logger.info(f"Request status: {response}")
        logger.info(f"payload: {payload}")
        logger.info("Succesfully sent the file to bubble! Check in bubble")
        logger.info('Work Finished ')


        # send_webhook(url="http://localhost:8001", json={"outputs": outputs})

        return

    except Exception as e: 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logger.info(f"\n{'*'*100}\nError Type: {exc_type},\nOriginal Error: {str(e)}\nLine Number: {exc_tb.tb_lineno}\n{'*'*100}\n")

        payload = {'file': 'RXJyb3IgaW4gZmlsZSEg', 'Email': email, 'youtube_title':youtube_title, 'status': 'Failed'}

        logger.info(f"File is not processed there are some error: {e}")
        response = requests.request("PATCH", url, headers=headers, data=payload)

        return str(e)

def create_subtitle(data:dict) -> str:
    """ Takes the input as banana output and convert to youtube format"""
    #data = data['modelOutputs'][0]

    all = ""
    for idx in range(len(data['segments'])):
        start = str(timedelta(seconds=data['segments'][idx]['start']))
        end = str(timedelta(seconds=data['segments'][idx]['end']))
        text = data['segments'][idx]['text']
        final =str(idx+1)+'\n'+start+' --> '+end+'\n'+text+'\n\n'
        all += final


    return all[:-2]



if __name__ == "__main__":
    app.serve()


