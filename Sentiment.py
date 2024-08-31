import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import time


@st.cache_data
def predict(_model, _tokenizer, text):
    encoded_text = _tokenizer(text, return_tensors='pt')
    output = _model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }
    for key, value in scores_dict.items():
        if value == max(scores):
            return f"This text is {round(value*100, 2)}% {key}"


def response_generator(response):
    for word in response.split():
        yield word + ' '
        time.sleep(0.05)


def main():
    st.header("Sentiment Analysis")

    if "model" not in st.session_state:
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        st.session_state.model = AutoModelForSequenceClassification.\
            from_pretrained(MODEL)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Input text to extract sentiment from"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            prediction = predict(st.session_state.model,
                                 st.session_state.tokenizer,
                                 prompt)
            response = st.write_stream(response_generator(prediction))
        st.session_state.messages.append({"role": "assistant",
                                          "content": response})


if __name__ == '__main__':
    main()
