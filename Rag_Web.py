import streamlit as st 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



st.set_page_config(page_title="car Insurance AI",layout="centered")
st.title("car Insurance AI")

user_input = st.text_input(label="Type your question", placeholder="Ask anything in your mind")

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError("There is no OPENAI_API_KEY avaliable in environment variables")
else:
    print("Successfully verify that OPENAI_API_KEY is in enviromental variables")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
text_spliter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 200, separators=["\n\n","\n"," ", ""])
loader = PyPDFLoader("Statefarm.pdf")
print("Loading documents...")
documents = loader.load()
print("Splitting documents...")
chunks = text_spliter.split_documents(documents)
print(f"Splitted the documents into {len(chunks)} chunks")
print("Creating vector store DB...")
vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
print("Creating retriever...")
retriever = vector_store.as_retriever()


llm_model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a knowledgeable car insurance company assistant specialized in answering questions about insurance policies, coverage, claims, and related services.
You must strictly follow these rules:
Only use information from the retrieved context (insurance policy documents, terms, and conditions).
Do not provide information that is not contained in the retrieved documents.
Provide clear, simple explanations about insurance policies, coverage options, claims processes, and related topics.
IF THE ANSWER IS NOT CONTAINED IN THE RETRIEVED TEXT, RESPOND WITH:
"The retrieved text does not contain enough information to answer this question. Please contact customer service for assistance."
When relevant:
Mention specific policy sections, coverage types, or terms from the provided context.
Summarize using plain language while maintaining accuracy about insurance terms and conditions.
Maintain a professional, helpful, and customer-service oriented tone.
Your job is to answer the user's question based solely on the insurance documentation provided in the context below."""),
("user","Question:{question}  \n Context: {context}")

])

chain = prompt | llm_model

if user_input:
    docs = retriever.invoke(user_input)
    docs_as_string = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({"question": user_input, "context": docs_as_string})
    st.write(response.content)

