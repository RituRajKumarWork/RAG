from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

def build_qa_chain(vector_store: FAISS):
    llm = Ollama(model="mistral", temperature=0.2)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Only answer the question using the information provided in the context below.\n"
            "If the context does not have enough information to answer, say politely:\n"
            "\"I do not have enough information from the uploaded documents to answer this question.\"\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "score_threshold": 0.5
        }
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=True
    )

    return chain