import os
import sys
import warnings
from dotenv import load_dotenv
import streamlit as st
import time
from operator import itemgetter
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings 
from llama_index.core import load_index_from_storage

if not sys.warnoptions:
    warnings.simplefilter("ignore")


#Loading the HuggingFace Token from .env
HF_TOKEN = ""
try:
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
except:
    HF_TOKEN = st.secrets["HF_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

#specifying the directory that contains the vector embeddings
PERSIST_DIR = "./db"

#Loading the LLM model from HugginFace API
llm = HuggingFaceEndpoint(
    endpoint_url="huggingfaceh4/zephyr-7b-alpha",
    max_new_tokens = 2048,
    repetition_penalty =  1.1,
    temperature = 0.5,
    top_p = 0.9,
    return_full_text = False,
)
#To Load the Embedding model from HugginFace API
#embed_model = LangchainEmbedding(HuggingFaceInferenceAPIEmbeddings(model_name="thenlper/gte-large", api_key=HF_TOKEN))
#To Load the Embedding model from HugginFace Locally
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")



#Load index
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings, chunk_size=512)
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(service_context=service_context, storage_context=storage_context)
query_engine = index.as_query_engine()

#Function to chain into Langchain
def get_ans(text):
    print(text)
    return query_engine.retrieve(text)

#Prompt Template for Standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

#Prompt Template for Actual question with context
template = """
User: You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

Keep in mind, you will lose the job, if you answer out of CONTEXT questions

CONTEXT: {context}
Query: {question}
Return the output in a structured manner in under 200 words
Assistant:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

templateff = """
User: You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

Keep in mind, you will lose the job, if you answer out of CONTEXT questions

CONTEXT: {context}
Parse The above context containing the eligibility criteria into a list 
Assistant:
"""
ANSWER_PROMPTFF = ChatPromptTemplate.from_template(templateff)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


template_questions = """
Generate possible questions from the given query
The questions must be well defined along with the relevant keywords
Query: {question}
All the questions must be separated using a double comma:',,'
"""
prompt_questions = ChatPromptTemplate.from_template(template_questions)
output_parser = StrOutputParser()


#Converts list of dict to string
def get_history(d):
    #print("hello ", d)
    txt = ""
    for i in d:
        v = [str(j) for j in i.values()]
        txt = txt + str(v[0]) + ":" + str(v[1]) + "\n\n"
    return txt

#Chain for Standalone question
_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history= itemgetter("chat_history")|RunnableLambda(get_history)
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)
#Chain to Generate the Context
_context = {
    "context": itemgetter("standalone_question") | RunnableLambda(get_ans),
    "question": lambda x: x["standalone_question"],
}

_contextff = {
    "context": itemgetter("standalone_question") | RunnableLambda(get_ans),
}

#Final Chain
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm
conversational_ff_chain = _inputs | _contextff | ANSWER_PROMPTFF | llm




#Chain for Generating Questions
chain_questions = (
    {
        "question": RunnablePassthrough(),
    }
    | prompt_questions
    | llm
    | output_parser
)
#Function for Parsing the Questions
def parse_questions(full_text):
    full_ques= chain_questions.invoke(full_text)
    txl = full_ques.split('.')
    ques = []
    for q in txl[1:len(txl)-1]:
        ques.append(q[1:len(q)-2])
    return ques


st.title("HelpBOT")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "buttons" not in st.session_state:
    st.session_state["buttons"] = []

if "eligi" not in st.session_state:
    st.session_state["eligi"] = []

if "questions" not in st.session_state:
    st.session_state["questions"] = ["Tamil Nadu Government Schemes", "Central Government Schemes", "Andhra Pradesh Government Schemes", "Kerala Government Social Justice Department Schemes"]


def run(prompt):
    st.session_state["messages"].append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_text = conversational_qa_chain.invoke({ "chat_history":st.session_state["messages"][-(min(len(st.session_state["messages"]), 2)):]
                                                    , "question":prompt,})

        #chain.invoke()
        text_container = st.empty()
        for _ in range(0, len(full_text), 10):
            text_container.markdown(full_text[:_])
            time.sleep(0.05)
        text_container.markdown(full_text[:len(full_text)])
        #st.markdown(full_text)
        st.session_state["questions"] = parse_questions(full_text)
        #st.session_state["questions"] = []
        
        message = full_text
        st.session_state["messages"].append({"role":"assistant", "content":message})



if  prompt:= st.chat_input("Enter your queries about any Government Schemes "):
    run(prompt)
    
q_len = min(4, len(st.session_state["questions"]))
st.session_state["buttons"] = [st.button(label=i) for i in st.session_state["questions"][:q_len]]   
for index, clicked in enumerate(st.session_state["buttons"]):
    if clicked:
        run(st.session_state["questions"][index])
        st.rerun()

with st.sidebar:
    st.title("Know the Eligibility Criteria")
    inp = st.text_input("Enter the Scheme Name")
    btn = st.button(label="Check Criteria")
    if btn:
        full_text = conversational_ff_chain.invoke({ "chat_history":st.session_state["messages"][-(min(len(st.session_state["messages"]), 2)):]
                                                        , "question":"What is the Eligibility Criteria for "+inp,})
        mk = st.empty()
        #print(full_text)
        mk.markdown(full_text)
    
    #st.write("slider", slider_val, "checkbox", checkbox_val)
