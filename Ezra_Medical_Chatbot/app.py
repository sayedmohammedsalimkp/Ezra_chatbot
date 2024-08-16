from flask import Flask,render_template,request,jsonify
from src.helper import download_hf_embeddings
import os
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *



app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

embeddings=download_hf_embeddings()

index_name="ezra-chatbot"

docsearch=PineconeVectorStore.from_existing_index(index_name,embeddings)

PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt":PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                model_type="llama",
                config={'max_new_tokens':512,
                        'temperature':0.6})



qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ =='__main__':
    app.run(host="0.0.0.0",port=8080, debug=True)