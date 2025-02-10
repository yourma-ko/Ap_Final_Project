import streamlit as st
import logging
import time
import os
from typing import List, Optional
import tempfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# Configure logging
logging.basicConfig(level=logging.INFO)
# Add after existing session state initialization
if 'model' not in st.session_state:
    st.session_state.model = "llama3.2"  # default model
# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_path = os.path.join(os.getcwd(), "chroma_db")
if not os.path.exists(db_path):
    os.makedirs(db_path)

def initialize_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

vectorstore: Optional[Chroma] = None
MAX_DOCUMENTS = 1000  # Maximum number of documents allowed
MAX_CHUNK_SIZE = 500  # Maximum size of text chunks
BATCH_SIZE = 50  # Number of documents to process at once

def process_in_batches(documents, chunk_size=BATCH_SIZE):
    """Process documents in batches to prevent memory issues"""
    for i in range(0, len(documents), chunk_size):
        yield documents[i:i + chunk_size]
def process_file(uploaded_file):
    """Process uploaded files using LangChain document loaders with batch processing"""
    global vectorstore
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
    
    # Check current document count
    current_count = vectorstore._collection.count() if vectorstore else 0
    if current_count >= MAX_DOCUMENTS:
        raise ValueError(f"Vector store limit reached ({MAX_DOCUMENTS} documents maximum)")
    
    file_path = None    
    try:
        # Create a temporary file with the correct extension
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            file_path = tmp.name

        logging.info(f"Processing file: {uploaded_file.name} (type: {uploaded_file.type})")
        
        # Choose appropriate loader based on file type
        if uploaded_file.type == "application/pdf" or suffix.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
            logging.info("Using PDF loader")
        elif uploaded_file.type == "text/plain" or suffix.lower() == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
            logging.info("Using text loader")
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")

        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            raise ValueError("No documents were loaded from the file")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        if len(splits) + current_count > MAX_DOCUMENTS:
            raise ValueError(f"Adding these documents would exceed the maximum limit of {MAX_DOCUMENTS}")
        
        # Process documents in batches
        total_processed = 0
        for batch in process_in_batches(splits):
            try:
                vectorstore.add_documents(batch)
                total_processed += len(batch)
                logging.info(f"Processed batch of {len(batch)} documents")
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue
        
        return total_processed
    
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logging.info("Temporary file cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {str(e)}")

def get_conversation_chain():
    """Initialize the conversational retrieval chain"""
    global vectorstore
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
        
    llm = Ollama(model=st.session_state.model, temperature=0.7)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        return_source_documents=True,
        verbose=True,
        output_key="answer"
    )
    
    return chain
    
def process_webpage(url: str) -> int:
    """Process webpage content using BeautifulSoup with size limits"""
    global vectorstore
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
    
    # Check current document count
    current_count = vectorstore._collection.count()
    if current_count >= MAX_DOCUMENTS:
        raise ValueError(f"Vector store limit reached ({MAX_DOCUMENTS} documents maximum)")
    
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format")
            
        # Fetch webpage content
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', '[document]', 'header', 'footer', 'nav']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        if not text:
            raise ValueError("No text content found on webpage")
        
        # Create document with metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=100
        )
        splits = text_splitter.create_documents(
            texts=[text],
            metadatas=[{"source": url, "type": "webpage"}]
        )
        
        if len(splits) + current_count > MAX_DOCUMENTS:
            raise ValueError(f"Adding this content would exceed the maximum limit of {MAX_DOCUMENTS}")
        
        # Process in batches
        total_processed = 0
        for batch in process_in_batches(splits):
            try:
                vectorstore.add_documents(batch)
                total_processed += len(batch)
                logging.info(f"Processed batch of {len(batch)} documents")
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue
        
        logging.info(f"Successfully processed webpage: {url}")
        return total_processed
        
    except requests.RequestException as e:
        logging.error(f"Error fetching webpage: {str(e)}")
        raise ValueError(f"Failed to fetch webpage: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing webpage: {str(e)}")
        raise

def generate_wordcloud(text: str) -> plt.Figure:
    """Generate a word cloud visualization from text"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_knowledge_graph(text: str, num_entities: int = 10) -> plt.Figure:
    """Create a knowledge graph visualization"""
    # Extract entities (simple approach using most common words)
    vectorizer = CountVectorizer(stop_words='english', max_features=num_entities)
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for word in words:
        G.add_node(word)
    
    # Add edges between co-occurring words
    for i, word1 in enumerate(words):
        for word2 in words[i+1:]:
            if text.find(word1) != -1 and text.find(word2) != -1:
                G.add_edge(word1, word2)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=10, font_weight='bold')
    return fig

def perform_topic_modeling(text: str, num_topics: int = 3) -> tuple:
    """Perform topic modeling on the text"""
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform([text])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-5-1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return topics
def generate_summary(text: str, max_length: int = 150) -> str:
    """Generate a summary using the LLM"""
    llm = Ollama(model=st.session_state.model)
    prompt = (
        f"Please summarize the following text in {max_length} words or less. "
        f"Focus on the main points and key information:\n\n{text}"
    )
    return llm(prompt)
def main():
    global vectorstore
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
        
    st.title("Enhanced Chat with LangChain Integration")
    logging.info("App started")

    # Sidebar for document management and input methods
    with st.sidebar:
        st.title("Model Configuration")
        # Add model selector at the top of sidebar
        model_option = st.selectbox(
            "Select Model",
            options=["llama3.2", "llama2"],
            index=0,
            help="Choose the model to use for generating responses"
        )
        
        # Update model in session state if changed
        if model_option != st.session_state.model:
            st.session_state.model = model_option
            st.session_state.memory.clear()  # Clear memory when model changes
            st.success(f"Switched to {model_option} model")
        
        st.divider()
        
        st.title("Document Management")
        
        # URL Input section
        st.subheader("Process Web Content")
        url = st.text_input("Enter a webpage URL:", placeholder="https://example.com")
        if url and st.button("Process Webpage"):
            try:
                with st.spinner("Processing webpage..."):
                    num_chunks = process_webpage(url)
                    st.success(f"Successfully processed webpage into {num_chunks} chunks")
            except Exception as e:
                st.error(f"Error processing webpage: {str(e)}")

        # File Upload section
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
        if uploaded_file:
            try:
                with st.spinner("Processing file..."):
                    num_chunks = process_file(uploaded_file)
                    if num_chunks:
                        st.success(f"Successfully processed {uploaded_file.name} into {num_chunks} chunks")
                        st.session_state.uploaded_file = uploaded_file.name
                    else:
                        st.error("Failed to process file")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logging.error(f"File processing error: {str(e)}")
        
        # Clear Vector Store button
        st.divider()
        if st.button("Clear Vector Store"):
            try:
                if vectorstore:
                    vectorstore.delete_collection()
                vectorstore = initialize_vectorstore()
                st.session_state.memory.clear()
                st.success("Vector store cleared!")
            except Exception as e:
                st.error(f"Error clearing vector store: {str(e)}")
                logging.error(f"Vector store clearing error: {str(e)}")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Insights", "Summary"])
    
    with tab1:
        # Create containers with specific heights
        chat_container = st.container()
        input_container = st.container()
        
        # Display chat history in the container above
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Handle input at the bottom
        with input_container:
            if prompt := st.chat_input("Your question"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate response
                with st.spinner("Thinking..."):
                    try:
                        start_time = time.time()  # Initialize start_time here
                        chain = get_conversation_chain()
                        response = chain({"question": prompt})
                        response_message = response["answer"]
                        
                        # Add source documents to response
                        if response.get("source_documents"):
                            response_message += "\n\nSources:"
                            for i, doc in enumerate(response["source_documents"][:2], 1):
                                response_message += f"\n{i}. {doc.page_content[:200]}..."

                        duration = time.time() - start_time
                        response_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_with_duration
                        })
                        
                        # Rerun to update chat history immediately
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        logging.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Document Insights")
        
        if vectorstore and vectorstore._collection.count() > 0:
            try:
                # Get all documents from vectorstore
                results = vectorstore.get()
                # Extract text content correctly from the results
                all_text = " ".join(results['documents'])
                
                # Add visualization controls
                with st.expander("Visualization Settings", expanded=True):
                    num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=3)
                    num_entities = st.slider("Number of Entities for Knowledge Graph", 
                                        min_value=5, max_value=20, value=10)
                
                # Word Cloud
                st.subheader("Word Cloud")
                wordcloud_fig = generate_wordcloud(all_text)
                st.pyplot(wordcloud_fig)
                
                # Topic Modeling
                st.subheader("Topic Analysis")
                topics = perform_topic_modeling(all_text, num_topics=num_topics)
                for topic in topics:
                    st.write(topic)
                
                # Knowledge Graph
                st.subheader("Knowledge Graph")
                graph_fig = create_knowledge_graph(all_text, num_entities=num_entities)
                st.pyplot(graph_fig)
                
                # Save figures to BytesIO for download
                import io
                
                # Save Word Cloud
                wordcloud_bytes = io.BytesIO()
                wordcloud_fig.savefig(wordcloud_bytes, format='png', bbox_inches='tight')
                wordcloud_bytes.seek(0)
                
                # Download buttons
                st.download_button(
                    label="Download Word Cloud",
                    data=wordcloud_bytes,
                    file_name="wordcloud.png",
                    mime="image/png"
                )
            
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
                logging.error(f"Visualization error: {str(e)}")
    with tab3:
        st.subheader("Document Summary")
        
        if vectorstore and vectorstore._collection.count() > 0:
            try:
                # Get all documents from vectorstore
                results = vectorstore.get()
                all_text = " ".join(results['documents'])
                
                # Summary controls
                with st.expander("Summary Settings", expanded=True):
                    max_length = st.slider(
                        "Maximum Summary Length (words)", 
                        min_value=50, 
                        max_value=500, 
                        value=150,
                        help="Adjust the length of the generated summary"
                    )
                
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        try:
                            summary = generate_summary(all_text, max_length)
                            st.markdown("### Summary")
                            st.write(summary)
                            
                            # Add download button for summary
                            st.download_button(
                                label="Download Summary",
                                data=summary,
                                file_name="summary.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                            logging.error(f"Summary generation error: {str(e)}")
                
                # Show original text
                with st.expander("View Original Text", expanded=False):
                    st.write(all_text)
                    
            except Exception as e:
                st.error(f"Error accessing documents: {str(e)}")
                logging.error(f"Document access error: {str(e)}")
        else:
            st.info("Upload documents or process web content to generate a summary")
if __name__ == "__main__":
    main()

