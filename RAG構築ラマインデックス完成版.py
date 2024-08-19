import os
import tempfile # PDFアップロードの際に必要

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel
from typing import Union
# from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st



folder_name = "./.data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# ストリーム表示
# class StreamCallbackHandler(BaseCallbackHandler):
#     def __init__(self):
#         self.tokens_area = st.empty()
#         self.tokens_stream = ""

#     def on_llm_new_token(self, token, **kwargs):
#         self.tokens_stream += token
#         self.tokens_area.markdown(self.tokens_stream)

# UI周り
st.title("QA")
uploaded_file = st.file_uploader("Upload a file after paste OpenAI API key", type="pdf")
    
with st.sidebar:
    select_chunk_size = st.slider("Chunk", min_value=0.0, max_value=1000.0, value=300.0, step=10.0,)

# 環境変数からAPIキーを取得
api_key = os.getenv('OPENAI_API_KEY')

# APIキーが存在するか確認
if not api_key:
    st.error("APIキーが設定されていません。環境変数 'OPENAI_API_KEY' にAPIキーを設定してください。")
    st.stop()

# OpenAIクライアントの初期化
try:
    openai_client = OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0.0)
except Exception as e:
    st.error(f"OpenAIクライアントの初期化に失敗しました: {e}")
    st.stop()

if uploaded_file:
    # 一時ファイルにPDFを書き込みバスを取得
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyMuPDFReader() 
    documents = loader.load(file_path=tmp_file_path) 

    text_splitter = TokenTextSplitter(
        chunk_size = select_chunk_size,
        chunk_overlap  = 100,
    )

    data = text_splitter(documents)
    pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])
    nodes = pipeline.run(documents=data)

    # データをCollectionに加える
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection("new_collection")


    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
    )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# モデルの応答
    messages = [
        ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="What is your name"),
    ]   
    response = OpenAI().chat(messages)
    # 環境変数からAPIキーを取得
    api_key = os.getenv('OPENAI_API_KEY')

    # 会話履歴を初期化
    if "memory" not in st.session_state:
        st.session_state.memory = SimpleChatStore(
            memory_key="chat_history",
            return_messages=True
        )

    # メモリの初期化  
    chat_store = st.session_state.memory

    # ChatMemoryBufferの設定
    chat_memory = ChatMemoryBuffer(
        token_limit=1000,
        memory=chat_store,  # 'memory' を使って、chat_storeを渡す
    )

    # チャットエンジンの設定
    chat_engine = index.as_chat_engine(
        memory=chat_memory,
        similarity_top_k=5
    )

    # UI用の会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # UI用の会話履歴を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # UI周り
    prompt = st.chat_input("Ask something about the file.")

    if prompt:
        # UI用の会話履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIからの応答
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_engine.chat(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response.response[0]})
                    st.markdown(response.response[0])
                except Exception as e:
                    st.error(f"応答の生成に失敗しました: {e}")

        # UI用の会話履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response.response[0]})

    # メモリの内容をターミナルで確認
    print(chat_memory)
