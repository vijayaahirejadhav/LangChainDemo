import os
from dotenv import load_dotenv
load_dotenv()
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


# We have to store vector database for Question Answer system
# It need path and hence we provide following path here.
vectordb_file_path='QAVectorDB'

llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.1)
# news = llm('what is the 1+1 ?')
# print(news)


# Create Hugging Face Embedding 
embeddings = HuggingFaceEmbeddings()

# We use this function only once to create vector database for given csv file
def create_vector_db():
    loader = CSVLoader(file_path='question_ans.csv',source_column="prompt")
    document = loader.load()    
    vectordb = FAISS.from_documents(documents=document, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)

def get_QA_Chain():
    # Load the vector database for QA chain
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)
    
    #create a retriver for querying the vector database
    retriever  = vectordb.as_retriever()
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}
    QUESTION: {question}"""

    prompt  = PromptTemplate(
        template = prompt_template, 
        input_variables =['context','question'])
    
    # Create retrieve chain
    chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                input_key='query',
                return_source_documents=True,
                chain_type_kwargs={'prompt':prompt})
    return chain

if __name__ == "__main__"    :
    create_vector_db()
    chain  = get_QA_Chain()

    response = chain("Do you provide any EMI ? and Is their any job gurantee?")
    print(response['result'])