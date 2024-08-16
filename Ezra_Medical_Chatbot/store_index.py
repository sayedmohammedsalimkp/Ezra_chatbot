from src.helper import load_pdf,text_split,download_hf_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os 
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')



extracted_data=load_pdf("data/")
text_chunks =text_split(extracted_data)
embeddings=download_hf_embeddings()


index_name="ezra-chatbot"

docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
