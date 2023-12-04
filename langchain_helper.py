# This is the main LangChain helper class all logic written over here
import os
#from secret_key import  openai_key

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Add your own created OpenAI API key
os.environ['OPENAI_API_KEY'] = 'sk-...'

llm = OpenAI(temperature=0) 


def generate_restaurant_name_and_items(cuisine):
    # Chain 1 : Restaurant Item Chain
    prompt_template_name = PromptTemplate(
    input_variables = ['cuisine'],
    template = 'I want to open restaurant for {cuisine} food. Suggest a fency name for this'
    )
    name_chain = LLMChain(llm = llm, prompt = prompt_template_name,output_key='restaurant_name')

    # Chain 2 : Menu Item Chain
    prompt_template_item = PromptTemplate(
    input_variables =['restaurant_name'],
    template = "Suggest menu item for {restaurant_name}. Return it as a comma spearated list"
    )
    food_item_chain = LLMChain(llm = llm, prompt = prompt_template_item,output_key='menu_items')
    
    # Call Sqeuential Chain
    chain = SequentialChain(
    chains = [name_chain,food_item_chain],
    input_variables = ['cuisine'],
    output_variables = ['restaurant_name',"menu_items"]
    )

    response = chain({'cuisine':cuisine})
    
    return response

# # Following code used to test the function
# if __name__ == "__main__":
#     print(generate_restaurant_name_and_items("Indian"))