import telebot
import logging
import time
import os
from typing import Dict, Optional
import tempfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import io
from telebot.types import Message
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
# Import your existing functions and configurations
from main import (
    initialize_vectorstore, process_webpage, generate_summary,
    generate_wordcloud, create_knowledge_graph, perform_topic_modeling,
    get_conversation_chain, process_file, ConversationBufferMemory
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot with your token
BOT_TOKEN = "7"
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize state
vectorstore = None
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
user_states: Dict[int, str] = {}  # Track user states for multi-step commands

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message: Message):
    """Welcome message and command list"""
    welcome_text = """
Welcome to the LangChain Chat Bot! 🤖

Available commands:
/uploadfile - Upload a PDF or TXT file
/processlink - Process a webpage
/makesummary - Generate a summary of processed documents
/viewvisualizations - Generate and view visualizations
/cleardata - Clear all stored documents
/help - Show this help message

You can also simply send me a message to chat!
    """
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['uploadfile'])
def request_file(message: Message):
    """Handle file upload request"""
    bot.reply_to(message, "Please send me a PDF or TXT file")
    user_states[message.from_user.id] = 'awaiting_file'

@bot.message_handler(content_types=['document'])
def handle_file(message: Message):
    """Process uploaded document"""
    if message.from_user.id not in user_states or user_states[message.from_user.id] != 'awaiting_file':
        return

    try:
        # Check file type and size
        if not message.document.mime_type in ['application/pdf', 'text/plain']:
            bot.reply_to(message, "Please send only PDF or TXT files")
            return

        if message.document.file_size > 10 * 1024 * 1024:  # 10MB limit
            bot.reply_to(message, "File size too large. Please send files under 10MB")
            return

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Create a unique filename
        file_extension = '.pdf' if message.document.mime_type == 'application/pdf' else '.txt'
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"doc_{int(time.time())}{file_extension}")
        
        # Save the file
        with open(temp_path, 'wb') as f:
            f.write(downloaded_file)
        
        try:
            # Log file details
            logging.info(f"Processing file: {message.document.file_name} ({message.document.mime_type})")
            logging.info(f"Saved to temporary path: {temp_path}")
            
            # Process the file
            bot.reply_to(message, "Processing file... This may take a moment.")
            
            # Initialize vectorstore if needed
            global vectorstore
            if vectorstore is None:
                vectorstore = initialize_vectorstore()
            
            num_chunks = process_file(temp_path)
            
            if num_chunks and num_chunks > 0:
                bot.reply_to(message, f"Successfully processed {message.document.file_name} into {num_chunks} chunks")
            else:
                bot.reply_to(message, "Failed to process file - no valid content found")
                
        except Exception as e:
            logging.error(f"Error during file processing: {str(e)}")
            bot.reply_to(message, f"Error processing file: {str(e)}")
            
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logging.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {str(e)}")
                
    except Exception as e:
        logging.error(f"Error in handle_file: {str(e)}")
        bot.reply_to(message, f"Error processing file: {str(e)}")
    finally:
        user_states.pop(message.from_user.id, None)

@bot.message_handler(commands=['processlink'])
def request_link(message: Message):
    """Handle webpage processing request"""
    bot.reply_to(message, "Please send me a webpage URL")
    user_states[message.from_user.id] = 'awaiting_link'

@bot.message_handler(func=lambda message: user_states.get(message.from_user.id) == 'awaiting_link')
def process_link(message: Message):
    """Process the provided webpage URL"""
    try:
        num_chunks = process_webpage(message.text)
        bot.reply_to(message, f"Successfully processed webpage into {num_chunks} chunks")
    except Exception as e:
        bot.reply_to(message, f"Error processing webpage: {str(e)}")
    finally:
        user_states.pop(message.from_user.id, None)

@bot.message_handler(commands=['makesummary'])
def make_summary(message: Message):
    """Generate and send document summary"""
    try:
        if not vectorstore or vectorstore._collection.count() == 0:
            bot.reply_to(message, "No documents to summarize. Please upload some content first.")
            return

        results = vectorstore.get()
        all_text = " ".join(results['documents'])
        summary = generate_summary(all_text)
        bot.reply_to(message, f"Summary:\n\n{summary}")
    except Exception as e:
        bot.reply_to(message, f"Error generating summary: {str(e)}")

@bot.message_handler(commands=['viewvisualizations'])
def send_visualizations(message: Message):
    """Generate and send visualizations"""
    try:
        if not vectorstore or vectorstore._collection.count() == 0:
            bot.reply_to(message, "No documents to visualize. Please upload some content first.")
            return

        results = vectorstore.get()
        all_text = " ".join(results['documents'])

        # Generate visualizations
        wordcloud_fig = generate_wordcloud(all_text)
        graph_fig = create_knowledge_graph(all_text)
        topics = perform_topic_modeling(all_text)

        # Save and send wordcloud
        wordcloud_bytes = io.BytesIO()
        wordcloud_fig.savefig(wordcloud_bytes, format='png')
        wordcloud_bytes.seek(0)
        bot.send_photo(message.chat.id, wordcloud_bytes, caption="Word Cloud Visualization")

        # Save and send knowledge graph
        graph_bytes = io.BytesIO()
        graph_fig.savefig(graph_bytes, format='png')
        graph_bytes.seek(0)
        bot.send_photo(message.chat.id, graph_bytes, caption="Knowledge Graph")

        # Send topics
        topics_text = "Topic Analysis:\n" + "\n".join(topics)
        bot.reply_to(message, topics_text)

    except Exception as e:
        bot.reply_to(message, f"Error generating visualizations: {str(e)}")

@bot.message_handler(commands=['cleardata'])
def clear_data(message: Message):
    """Clear all stored documents"""
    try:
        global vectorstore
        if vectorstore:
            vectorstore.delete_collection()
        vectorstore = initialize_vectorstore()
        memory.clear()
        bot.reply_to(message, "All data has been cleared!")
    except Exception as e:
        bot.reply_to(message, f"Error clearing data: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_message(message: Message):
    """Handle regular messages as chat prompts"""
    try:
        chain = get_conversation_chain()
        response = chain({"question": message.text})
        response_message = response["answer"]
        
        if response.get("source_documents"):
            response_message += "\n\nSources:"
            for i, doc in enumerate(response["source_documents"][:2], 1):
                response_message += f"\n{i}. {doc.page_content[:200]}..."
        
        bot.reply_to(message, response_message)
    except Exception as e:
        bot.reply_to(message, f"Error generating response: {str(e)}")
def get_conversation_chain():
    """Initialize the conversational retrieval chain"""
    global vectorstore
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
        
    llm = Ollama(model="llama2", temperature=0.7)  # Set model to llama2
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True,
        output_key="answer"
    )
    
    return chain
def main():
    """Start the bot"""
    global vectorstore
    vectorstore = initialize_vectorstore()
    logging.info("Bot started")
    bot.infinity_polling()

if __name__ == "__main__":
    main()