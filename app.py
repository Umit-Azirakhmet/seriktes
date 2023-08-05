import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from PIL import Image

APP_ICON_URL = "llog.png"

# Setup web page
st.set_page_config(
    page_title="Seriktes",
    page_icon=APP_ICON_URL,
    layout="wide",
)
st.markdown(
    """
    <style type="text/css">
    
    [data-testid=stSidebar] {
        background-color: rgb(230, 246, 252);
        color: #000000;
    }
    
    .title-centered {
        text-align: center;
    }
    .stApp {
        margin-top: 0;
        background: #3383a6;
    }
    div.stTabs button {
        background: #3383a6;
    }

    .stApp header {
        background: #3383a6;
        margin-top: 0;
        display: none;
    }
    
    .big-button {
        font-size: 24px;
        font-weight: bold;
        color: white;
        background-color: #3383a6;
        border: none;
        border-radius: 10px;
        padding: 20px;
        cursor: pointer;
        display: block;
        margin: 0 auto;
    }
    
    </style>
""",
    unsafe_allow_html=True,
)


# Sidebar contents
with st.sidebar:
    img = Image.open("llog.png")
    new_size = (150, 150)
    st.image(img, use_column_width=True)
    img = img.resize(new_size)
    # st.image(img)
    add_vertical_space(2)
    st.markdown(
        "<h1 class='title-centered'>SEriktES Chat App</h1>", unsafe_allow_html=True
    )

    st.markdown(
        """
        ## "SEriktES" is the app that helps with your academics. By uploading a PDF file of a paper, you can ask any question about the content.
        """
    )
    add_vertical_space(11)
    st.write("Made with ❤️ by Umit Azirakhmet")
    add_vertical_space(5)

load_dotenv()


# Function to load the chat page
def load_chat_page():
    st.markdown(
        "<h1 class='title-centered' style='color: white;'>Seriktes Chat App</h1>",
        unsafe_allow_html=True,
    )

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f"{store_name}")
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            # Modified prompt:
            prompt = "User will ask you questions about content of the file. Answer as you are user's study buddy. Encourage to study, give detailed information, give page of that information. Answer in the language of user."

            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(
                    input_documents=docs,
                    question=f"{prompt} {query}",
                    max_output_length=300,
                )
                print(cb)
            st.write(response)

    # Rest of your chat page code...
    # ...


# Function to load the main page
def load_main_page():
    st.markdown(
        "<h1 class='title-centered' style='color: white; margin-top: 0; padding-top: 0;'>Seriktes Chat App</h1>",
        unsafe_allow_html=True,
    )
    # st.image("bg01.png", use_column_width=True)  # Make the image resizable
    img = Image.open("gr1.png")
    new_size = (10, 10)
    st.image(img, use_column_width=True)
    img = img.resize(new_size)
    # Styled button to navigate to the chat page
    st.markdown(
        """
        <button class='big-button' style='margin-top: 0; padding-top: 0;' onclick='openChatPage()'>Go to Chat Page</button>
        """,
        unsafe_allow_html=True,
    )
    # JavaScript code to handle link clicks and change pages
    st.markdown(
        """
        <script>
        const handleLinkClick = () => {
            Streamlit.setComponentValue("go_to_chat", true);
        };
        </script>
        """,
        unsafe_allow_html=True,
    )

    # JavaScript code to handle link clicks and change pages
    st.markdown(
        """
        <script>
        const handleLinkClick = () => {
            Streamlit.setComponentValue("go_to_chat", true);
        };
        </script>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Create a sidebar navigation menu
    st.sidebar.title("Navigation")
    pages = ["Main Page", "Chat Page"]
    selected_page = st.sidebar.radio("Go to", pages)

    if selected_page == "Main Page":
        load_main_page()
    elif selected_page == "Chat Page":
        load_chat_page()


if __name__ == "__main__":
    main()
