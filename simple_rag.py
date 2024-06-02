import os

import torch

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, ServiceContext, load_index_from_storage
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.storage.storage_context import StorageContext

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

llm = LlamaCPP(

    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

documents = SimpleDirectoryReader(
    input_files = ["path_to_documents"]
).load_data()

documents = Document(text = "\n\n".join([doc.text for doc in documents]))

# Sentence Window Retrieval approach: retrieve based on smaller sentences to get a better match for the relevant context
# and then synthesize based on the expanded context window around the sentence.

# split the texts into sentence chunks and store the embedding of the chunks to the vector index

def build_index(documents, llm, embed_model, sentence_window_size=3, save_dir="vector_store/index"):

    node_parser = SentenceWindowNodeParser(
        window_size = sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    sentence_context = ServiceContext.from_defaults(
        llm = llm,
        embed_model = embed_model,
        node_parser = node_parser
    )

    if not os.path.exists(save_dir):
        # create and load index
        index = VectorStoreIndex.from_documents(
            [documents], service_context = sentence_context
        )
        index.storage_context.persist(persist_dir=save_dir)
    else:
        # load the existing index
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context = sentence_context
        )
    
    return index

# create vector index

vector_index = build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")
    
    engine = sentence_index.as_query_engine(similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank])

    return engine

# create the query engine

query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=6, rerank_top_n=2)

query = ""
response = query_engine.query(query)

print()
print("-- QUERY: ")
print(query)
print("-- RESPONSE: ")
print(response)