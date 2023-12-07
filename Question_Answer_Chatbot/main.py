import streamlit as st
from langchain_QA_helper import create_vector_db
from langchain_QA_helper import get_QA_Chain

st.title("Question Answer ")
# Create button to create vector database
btn = st.button("Create Knowledgebase")

if btn:
    create_vector_db()

question = st.text_input("Ask your Question: ")


if question:
    chain = get_QA_Chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response['result'])