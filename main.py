import streamlit 
from keras.applications.xception import Xception
import numpy as np
from PIL import Image
from keras.models import load_model
from pickle import load
from keras.preprocessing.sequence import pad_sequences
import PIL.Image
import google.generativeai as genai

genai.configure(api_key=streamlit.secrets["GOOGLE_GEMINI_AI"])

streamlit.set_page_config(page_title="Image Captioning App", page_icon="🔍", layout="centered", initial_sidebar_state="expanded")
streamlit.title("Image Captioning App")

streamlit.write("This is an Image Captioning App which uses a Deep Learning Model to generate captions for the uploaded image. The model is trained on Flickr8k dataset and uses Xception model for feature extraction and LSTM for generating captions. The model is trained for 10 epochs. The model is also integrated with GenerativeAI API to generate a response for the uploaded image.")

def pridict_caption(image_path):
    def extract_features(filename, model):
            try:
                image = Image.open(filename)
            except:
                print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
            image = image.resize((299,299))
            image = np.array(image)
            if image.shape[2] == 4:
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            return feature

    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None


    def generate_desc(model, tokenizer, photo, max_length):
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text


    img_path = image_path
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")

    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)

    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    print(description)
    description.replace('start', ' ')
    description.replace('end', ' ')
    print(description)
    description = description[6:-3]
    return description

uploaded_file = streamlit.file_uploader("Choose an image...", type="jpg")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    streamlit.image(image, caption='Uploaded Image.', use_column_width=False)
    model = genai.GenerativeModel("models/gemini-1.0-pro-vision-latest")
    image = PIL.Image.open(uploaded_file)
    with streamlit.spinner('Generating Description...'):
        response = model.generate_content(image)
        description = pridict_caption(uploaded_file)

        streamlit.markdown(f'''
                           # Results
                           
                           ## Response from Our Model:
                           {description}

                            ## Response from GenerativeAI:
                            {response.text}
                           ''')
