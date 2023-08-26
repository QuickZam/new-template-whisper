from potassium import Potassium, Request, Response
from stable_whisper import modify_model 
from pytube import YouTube
import torch, whisper, os, base64, urllib.request



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
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    link = request.json.get("link")
    model = context.get("model")

    if 'tinyurl' in link: 
        path = urllib.request.urlretrieve(link, f"{link.split('/')[-1]}.mp4")[0]

    else:
        yt = YouTube(link)
        path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

    # translate_options = dict(task="translate", suppress_silence=True, ts_num=16, lower_quantile=0.05, lower_threshold=0.1)4
    translate_options = dict(task="translate",max_initial_timestamp=None)
    result = model.transcribe(path, **translate_options)
    os.remove(path)


    return Response(
        json = {"outputs": result[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()