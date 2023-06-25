import json
import base64
import requests
from huggingface_hub import InferenceClient
import streamlit as st
import plotly.express as px


API_TOKEN = st.secrets["HF"]

picture = st.file_uploader("Take a picture")

if picture:
    st.image(picture)
    client = InferenceClient(model="Salesforce/blip-image-captioning-base", token=API_TOKEN)
    st.write(client.image_to_text(picture.getvalue()))


    headers = {f"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"
    def query(filename, classes):
        img = base64.b64encode(picture.getvalue())
        data = {"image": img.decode(), "parameters": {"candidate_labels": classes}}
        response = requests.request("POST", API_URL, headers=headers, data=json.dumps(data))
        return json.loads(response.content.decode("utf-8"))


    res = query(picture, "pothole, fallen tree")
    fig = px.bar(res, y='label', x='score', orientation='h')
    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig)
