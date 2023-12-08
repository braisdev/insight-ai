import streamlit as st
from frontend.templates.htmlTemplates import css, bot_template, user_template
from backend.chat_engine import get_conversation_chain
from backend.faiss_engine import get_vectorstore
from backend.whisper_engine import transcribe_audio

def handle_user_input(user_question):
    """
    Handle the user input
    """
    if st.session_state.conversation:
        response = st.session_state.conversation({'question':user_question})
        st.session_state.chat_history = response['chat_history']

        for i,message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def main():

    st.set_page_config(page_title="Chat with your Videos and Audios!", page_icon=":video_camera:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with your Videos and Audios! :video_camera:")

    user_question = st.text_input("Ask a question about your files:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Multimedia")
        multimedia_files = st.file_uploader(
            "Upload your files here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing your files..."):

                # create the vectorstore
                vectorstore = get_vectorstore(transcribe_audio(multimedia_files))

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':

    main()