from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def get_conversation_chain(vectorstore):
    """
    Create the conversation chain
    """
    template = """
        You are a supportive and courteous chatbot engaged in a conversation with a user.
        To respond accurately to the user's question, adhere to the following steps:

        • Thoroughly review the provided context below.
        • Compose a response to the user's question based solely on the information available in the context.
        • If the answer to the question is not discernible from the context, kindly inform the user that you are unable to 
        provide assistance on that particular query, without attempting to fabricate an answer.

        Context: {context}
        Chat History: {chat_history}
        User Question: {question}
        """

    custom_answer_prompt = PromptTemplate(
        input_variables=["question", "context", "chat_history"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chat = ChatOpenAI(model="gpt-4-1106-preview",temperature=0)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = chat,
        retriever = vectorstore.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs=dict(prompt=custom_answer_prompt)
    )

    return conversation_chain
