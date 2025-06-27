import torch
import os

# Suppress LangSmith warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub


Loader = PyPDFLoader
FILE_PATH = "./yolov10.pdf"
loader = Loader(FILE_PATH)
document = loader.load()

embedding = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

semantic_splitter = SemanticChunker(
    embeddings=embedding,
    buffer_size=1,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
    min_chunk_size=500,
    add_start_index=True
)
docs = semantic_splitter.split_documents(document)
print("Number of semantic chunks: ", len(docs))

vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vector_db.as_retriever()

result = retriever.invoke("What is YOLO?")
print("Number of relevant documents: ", len(result))

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto",
)

llm = HuggingFacePipeline(pipeline=model_pipeline)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_question = "YOLOv10 ?"
output = rag_chain.invoke(user_question)
answer = output.split("Answer:")[1].strip()
print(answer)

