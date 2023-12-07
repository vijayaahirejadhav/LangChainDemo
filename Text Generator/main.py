# In this UI part created 
import streamlit as st 
from langchain_helper import generate_restaurant_name_and_items

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a Cuisine", ('Indian','Italian','Arabic','Chinese','French',))

if cuisine:
    response = generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    item_names = response['menu_items'].strip().split(',')
    st.write("**Menu Items:**")
    for item in item_names:
        st.write("-",item)
