import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import hashlib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import logging
from twilio.rest import Client # type: ignore
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GROQ_API_KEY= os.getenv("GROQ_API_KEY")
Account_SID = os.getenv("Account_SID")
Auth_Token = os.getenv("Auth_Token")
Twilio_Number= os.getenv("Twilio_Number")
Recipient_Number = os.getenv("Recipient_Number")





############################## whatsapp message send using TWilio ######################333
def send_to_whatsapp(conversation_log):
    try:
        # Twilio account credentials
        account_sid = Account_SID
        auth_token = Auth_Token
        client = Client(account_sid, auth_token)

        # Format the log as a readable string
        log_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_log])

        # Send to WhatsApp
        message = client.messages.create(
            from_=Twilio_Number,  # Twilio's sandbox WhatsApp number
            to= Recipient_Number,  # Your WhatsApp number
            body=f"Conversation Log:\n{log_text}"
        )
        print("WhatsApp message sent:", message.sid)
    except Exception as e:
        print("Failed to send WhatsApp message:", str(e))


# Configure Streamlit page
# st.set_page_config(page_title="University OF Sialkot", page_icon="📄", layout="wide")

page_bd_img = """
<style> 
        [data-testid="stAppViewContainer"]{
            background-color: #e5e5f7;
opacity: 0.8;
background: linear-gradient(135deg, #444cf755 25%, transparent 25%) -10px 0/ 20px 20px, linear-gradient(225deg, #444cf7 25%, transparent 25%) -10px 0/ 20px 20px, linear-gradient(315deg, #444cf755 25%, transparent 25%) 0px 0/ 20px 20px, linear-gradient(45deg, #444cf7 25%, #e5e5f7 25%) 0px 0/ 20px 20px;
        }
        </style>

"""

st.markdown(page_bd_img, unsafe_allow_html=True)


# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    st.error("Failed to initialize the AI model. Please check your API key.")

# List of available models
# MODELS = [
#     "llama3-70b-8192",
#     "llama3-8b-8192",
#     "gemma-7b-it",
#     "gemma2-9b-it",
#     "mixtral-8x7b-32768"
# ]

Model = "llama3-70b-8192"
# Model = "mixtral-8x7b-32768"

@st.cache_data
def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_page = {executor.submit(extract_text, page): page for page in pdf_reader.pages}
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    text += future.result() + "\n"
                except Exception as e:
                    logger.warning(f"Skipped page {pdf_reader.pages.index(page)} due to: {str(e)}")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error("Failed to process the PDF. Please try again with a different file.")
        return ""

def extract_text(page):
    try:
        return page.extract_text()
    except Exception as e:
        logger.warning(f"Error extracting text from page: {e}")
        return ""

# @st.cache_data
# def split_into_chunks(text, chunk_size=1000, overlap=150):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks

@st.cache_data
def split_into_chunks(text, chunk_size=1000, overlap=100):
    # Validate inputs
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start = 0
    
    # Generate chunks with overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())  # Remove any trailing whitespace
        start += chunk_size - overlap  # Move the start pointer with overlap

    return chunks



@st.cache_data
def get_or_create_chunks(file_path):
    try:
        with open(file_path, 'rb') as file:  # Open in binary read mode
            file_content = file.read()  # Read the file content as bytes
            file_hash = hashlib.md5(file_content).hexdigest()

        cache_file = f"cache/{file_hash}_chunks.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        with open(file_path, 'rb') as file:
            text = process_pdf(file)  
        chunks = split_into_chunks(text)

        os.makedirs('cache', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks, f)

        return chunks
    except Exception as e:
        logger.error(f"Error in get_or_create_chunks: {e}")
        print("Failed to process the PDF chunks. Please try again.")
        return []

# @st.cache_data
# def get_or_create_chunks(file_paths):
#     try:
#         combined_text = ""  # Initialize an empty string to hold combined text
        
#         for file_path in file_paths:
#             try:
#                 with open(file_path, 'rb') as file:
#                     file_content = file.read()  # Read the file content as bytes
#                     file_hash = hashlib.md5(file_content).hexdigest()

#                 cache_file = f"cache/{file_hash}_chunks.pkl"
#                 if os.path.exists(cache_file):
#                     with open(cache_file, 'rb') as f:
#                         cached_chunks = pickle.load(f)
#                         combined_text += " ".join(cached_chunks)  # Combine cached chunks
#                         continue

#                 # If not cached, process the file and extract text
#                 with open(file_path, 'rb') as file:
#                     try:
#                         text = process_pdf(file)  # Replace with robust text extraction
#                     except Exception as e:
#                         logger.warning(f"Error processing {file_path}: {e}")
#                         continue

#                     combined_text += text  # Combine text from all files
#             except Exception as file_error:
#                 logger.warning(f"Skipping file {file_path} due to error: {file_error}")
#                 continue

        

#         # Split the combined text into chunks
#         chunks = split_into_chunks(combined_text)

#         # Cache the chunks (use a single hash for the combined text)
#         combined_hash = hashlib.md5(combined_text.encode('utf-8')).hexdigest()
#         cache_file = f"cache/{combined_hash}_chunks.pkl"
#         os.makedirs('cache', exist_ok=True)
#         with open(cache_file, 'wb') as f:
#             pickle.dump(chunks, f)

#         return chunks
#     except Exception as e:
#         logger.error(f"Error in get_or_create_chunks: {e}")
#         return []


@st.cache_resource
def get_vectorizer(chunks):
    try:
        return TfidfVectorizer().fit(chunks)
    except Exception as e:
        logger.error(f"Error creating vectorizer: {e}")
        st.error("Failed to create text vectorizer. Please try again.")
        return None

def find_most_relevant_chunks(query, chunks, vectorizer,top_k):
    try:
        chunk_vectors = vectorizer.transform(chunks)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}")
        st.error("Failed to find relevant information. Please try a different query.")
        return []

def get_ai_response(messages, context, model):
    try:
        system_message = {"role": "system", "content": "You are a helpful university chatbot assistant for answering university of sialkot related questions about the given PDF content. Use the provided context to answer questions, but also consider the conversation history."}

        # Combine system message, conversation history, and the new query with context
        all_messages = [system_message] + messages[:-1] + [{"role": "user", "content": f"Context: {context}\n\nBased on this context and our previous conversation, please answer the following question: {messages[-1]['content']}"}]

        chat_completion = client.chat.completions.create(
            messages=all_messages,
            model=model,
            max_tokens=1024,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I'm sorry, Chat session compelete please reset chat by clicking on reset button"


def render_message(message, role):
    # Define CSS for alignment
    if role == "assistant":
        # Left-aligned messages (bot)
        alignment = "left"
        bg_color = "#f0f0f0"  # Light gray for bot
        text_color = "#000000"  # Black text
        icon = "🤖"  # Chatbot icon (or use an image URL)
        st.markdown(
        f"""
        <div style='display: flex; justify-content: {alignment}; margin-bottom: 10px;'>
            <div style='display: flex; align-items: center; max-width: 70%; 
                        background-color: {bg_color}; color: {text_color}; 
                        padding: 10px; border-radius: 10px;'>
                <span style='margin-right: 10px;'>{icon}</span>  <!-- Icon -->
                <span>{message}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True)
    else:
        # Right-aligned messages (user)
        icon = "👤"  # User icon (or use an image URL)
        alignment = "right"
        bg_color = "#0078D4"  # Blue for user
        text_color = "#ffffff"  # White text
        st.markdown(
        f"""
        <div style='display: flex; justify-content: {alignment}; margin-bottom: 10px;'>
            <div style='display: flex; align-items: center; max-width: 70%; 
                        background-color: {bg_color}; color: {text_color}; 
                        padding: 10px; border-radius: 10px;'>
                <span>{message}</span>
                <span style='margin-left: 10px;'>{icon}</span>  <!-- Icon -->
            </div>
        </div>
        """,
        unsafe_allow_html=True)

    # HTML for message rendering
  

def main():

    st.title("Chat With Uskt Chatbot")
      # Add a button to clear the conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun() 

    
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'model' not in st.session_state:
        st.session_state.model = Model
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'conversation_log' not in st.session_state:
        st.session_state.conversation_log = []  # To store all messages in a session


    # st.sidebar.header("Upload PDF")
    # pdf_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")



    # pdf_file = ["./data/Directions_data.pdf", "./data/Fee Structure.pdf", "./data/General_data.pdf", "./data/Post-Graduate_Programs.pdf", "./data/Teachers data.pdf", "./data/Under_Graduate_Programs.pdf", "./data/University of Sialkot chatbot.pdf"]
    pdf_file = "./data/Fee Structure.pdf"

    if pdf_file:
        with st.spinner("Processing PDF..."):
            st.session_state.chunks = get_or_create_chunks(pdf_file)
            st.session_state.vectorizer = get_vectorizer(st.session_state.chunks)
        if st.session_state.chunks and st.session_state.vectorizer:
            print("PDF processed successfully!")
            # sucess=st.success("PDF processed successfully!")
            # time.sleep(3) # Wait for 3 seconds
            # sucess.empty()
        else:
            st.error("Failed to process PDF. Please try again.")
            

    # selected_model = st.selectbox("Select Model", MODELS, index=MODELS.index(st.session_state.model))
    # if selected_model != st.session_state.model:
    #     st.session_state.model = selected_model

    # Display chat history (This code is orignal for chatbot)
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])
    
    for message in st.session_state.messages:
        render_message(message["content"], message["role"])

  
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Uskt"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_log.append({"role": "\n\n user", "content": prompt})
        render_message(prompt, "user")
        
        # with st.chat_message("user"):
        #     st.markdown(prompt)

        # with st.chat_message("assistant"):
        #     message_placeholder = st.empty()
        #     relevant_chunks = find_most_relevant_chunks(prompt, st.session_state.chunks, st.session_state.vectorizer) if st.session_state.chunks else []
        #     context = "\n\n".join(relevant_chunks)

        #     full_response = get_ai_response(st.session_state.messages, context, st.session_state.model)
        #     message_placeholder.markdown(full_response)

        # st.session_state.messages.append({"role": "assistant", "content": full_response})
        

        relevant_chunks = find_most_relevant_chunks(prompt, st.session_state.chunks, st.session_state.vectorizer, top_k=3) if st.session_state.chunks else []
        context = "\n\n".join(relevant_chunks)

        # prompt_limit = f"{st.session_state.messages} + {context}"
        # render_message(f"Len of chunk0 = {len(relevant_chunks[0])} \n Len of chunk1 = {len(relevant_chunks[1])} \n Len of chunk2 =  total chunks length:{len(relevant_chunks[2])} \n  relevent chunk:{len(relevant_chunks)}---------prompt size: {len(prompt_limit)}--------session state message len: -{len(st.session_state.messages)}--------Context length: -{len(context)}", "assistant")
        with st.spinner("Generating response..."):
            full_response = get_ai_response(st.session_state.messages, context, st.session_state.model)

        render_message(full_response, "assistant")
        render_message(context,  "assistant")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.conversation_log.append({"role": "assistant", "content": full_response})
        
        ################# Sending whatsapp log ##########################33
    
    if len(st.session_state.conversation_log) >= 4:  # Example threshold
        # Send the log to WhatsApp
        send_to_whatsapp(st.session_state.conversation_log)
        # Optionally clear the log after sending
        print("whatsapp message send")
        st.session_state.conversation_log = []



    

if __name__ == "__main__":
    main()
