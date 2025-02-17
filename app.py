import os
import streamlit as st
import asyncio
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from multiprocessing import Pool
# from functools import partial

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# model constants
CONTEXT_GATHERER = "openai/gpt-4o-mini"
CHAT_MODEL = "openai/gpt-4o-mini"
 
#Streamlit UI

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .user_message {
            background-color: #f4f4f4;
            padding: 8px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .assistant_message {
            background-color: #f4f4f4;
            padding: 8px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'context_text' not in st.session_state:
        st.session_state.context_text = ""
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    if 'persistent_context' not in st.session_state:
        st.session_state.persistent_context = ""



    # Sidebar for context gathering
    with st.sidebar:
        st.title("Context Settings")
        st.info("💡 Use [gitingest.com](https://gitingest.com/) for secure token management.")
        st.warning("Note: Context processing happens only once when you first ask a question. Subsequent questions will use the same processed context until you update it.")
        new_context = st.text_area("Enter your context here:",
                                   value= st.session_state.context_text, height=300)
        
        if st.button("Set context"):
            st.session_state.context_text = new_context
            st.session_state.processed_chunks = None # Reset processed chunks when context changes
            st.success("Context updated successfully!")

    st.title("Ask ai")

    # display the conversation so far
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # Add a form to accept new user input, only *after* everything else has been displayed
    # This means form will appear at the bottom of the page, below the conversation history
    user_input = None
    with st.form(key="message_form"):
        user_input = st.text_area("enter your message here:", height=100)
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("Send")
        with col2:
            reprocess_button = st.form_submit_button("Send with context Reprocess")

    # If the user provide a new message, handle it
    if (submit_button or reprocess_button) and user_input:
        # Add the user's message to the conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Place holder for streaming response
        response_placeholder = st.empty()

        if not st.session_state.context_text.strip():
            st.warning("Please set a context before asking a question in the sidebar.")

        # Force reprocess of chunks if reprocess button is clicked
        if reprocess_button:
            st.session_state.processed_chunks = None

        # Process chunks
        if st.session_state.processed_chunks is None:
            chunk_size = 16000
            overlap = 1000 # Number of characters to overlap between chunks

            # modify chunking logic with overlap
            chunks = [
                st.session_state.context_text[i:i+chunk_size]
                for  i in range(0, len(st.session_state.context_text), chunk_size - overlap)
            ]

            # Create chunks-question pairs
            chunk_and_question_pairs = [(chunk, user_input) for chunk in chunks]

            with st.spinner("Processing context..."):
                # Use multi processing to process chunks in parallel
                with Pool() as pool:
                    responses = pool.map(process_chunk_sync, chunk_and_question_pairs)

            # filter out errors and "no relevant context"
                relevant_contexts = [
                    resp for resp in responses
                    if not resp.startswith("Error") and resp != "no relevant answer"
                ]
                st.session_state.processed_chunks = "\n".join(relevant_contexts).strip()

        # Now generate the final answer
        combined_context = st.session_state.processed_chunks

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def process_final_answer():
                full_response = ""
                async for token in get_final_answer(combined_context, user_input):
                    full_response += token
                    response_placeholder.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)
            loop.run_until_complete(process_final_answer())
        finally:
            loop.close()

        # Rerun the app to refresh UI
        st.rerun()



def process_chunk_sync(chunk_and_question):
    """
    Synchronous version of chunk processing for multiprocessing
    """
    chunk, question = chunk_and_question
    client = OpenAI(
        base_url= "https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        )
    
    # Build convesation history for context

    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing a chunk of text to copy the"
                "relevant context to the question provided by the user."
                "If no relevant context is found, just output"
                "'no relevant answer' and no other explanation."
            )
        },
        {
            "role": "user",
            "content": (
                f"Based on this text:\n\n{chunk}\n\n"
                f"Find and copy the relevant context to answer this question: {question}"
            )
        }
    ]

    try:
        completion = client.chat.completions.create(
            extra_headers=
            {
                "HTTP-Referer": "http://localhost",
                "X-Title": "Local Script",
            },
            model=CONTEXT_GATHERER,
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error processing chunk: str{e}"
    
async def get_final_answer(combined_context, question):
    """
    Use the combined context to get a final, direct answer from the model.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # Build conversation history for context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the provided <context> to answer "
                "the user's question. Maintain a natural conversational flow. If you don't have enough context, just say you need more context."
            )
        }
    ]
    
    # Add conversation history
    for msg in st.session_state.conversation_history[-4:]:  # Include last 3-4 exchanges for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current context and question
    messages.append({
        "role": "user",
        "content": (
            f"Based on this context:\n\n<context>{combined_context}</context>\n\n"
            f"Please answer this question: {question}"
        )
    })

    try:
        completion = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Local Script",
            },
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        
        collected_messages = []
        async for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                collected_messages.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
                
        # After completion, update conversation history
        full_response = "".join(collected_messages)
        st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        error_msg = f"Error getting final answer: {str(e)}"
        st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
        yield error_msg

if __name__ == "__main__":
    main()