import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from utils import CustomLLM
from utils import qna_jira

import logging

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# st.set_page_config(page_title="QnA Digital.AI Products")


similarity_threshold = 0.3
broad_k = 20
k = 5

## streamlit app
with st.sidebar:
    st.image("./logo.png")
    st.title('QnA Jira')
    st.markdown('''
    ## About
    This app is an LLM-powered built using:
    - [H2O AI's Falcon 7B based model fine tuned on Open Assistant dataset](https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2)
    - [Langchain](https://python.langchain.com/en/latest/index.html)
    - [Streamlit](https://streamlit.io/)
    ''')
    add_vertical_space(5)

with st.form(key='qna_jira'):
    query = st.text_area('Ask from Jira ', '''
        What are metrics in Flow Acceleration Dashboard
        ''')
    with st.spinner('Wait for it...'):
        if st.form_submit_button('Get response on Product'):
            r = qna_jira(query,
                         similarity_threshold,
                         broad_k,
                         k)
            st.write(r)
        else:
            pass
