"""
Streamlit Web Demo for MiniMind Chat Interface

Interactive web interface for chatting with MiniMind models. Supports both local
model inference and remote API calls. Features include:

- Multi-turn conversation with configurable history length
- Model selection sidebar (local models or API endpoint)
- Real-time token streaming for responsive UX
- Conversation management (delete messages, regenerate responses)
- Reasoning visualization for R1 models (collapsible <think> blocks)
- Mobile-friendly UI with custom CSS styling

Usage:
    streamlit run web_demo.py
    
Requirements:
    - streamlit
    - transformers
    - torch
    - openai (for API mode)
"""

import random
import re
from threading import Thread
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Configure Streamlit page
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# Custom CSS for improved UI/UX
st.markdown("""
    <style>
        /* Custom button styling for message actions */
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        
        /* Adjust main container margins */
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* Compact delete button styling */
        .stButton > button {
            all: unset !important;
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Configuration
system_prompt: List[Dict[str, str]] = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_assistant_content(content: str) -> str:
    """
    Process assistant content to format reasoning blocks as collapsible HTML.
    
    For R1 reasoning models, content contains <think>...</think> tags with
    chain-of-thought reasoning. This function converts them into styled,
    collapsible HTML details/summary elements for better UI presentation.
    
    Args:
        content: Raw assistant response text
        
    Returns:
        HTML-formatted string with collapsible reasoning blocks
    """
    
    # Check if model is a reasoning model (R1 variants)
    is_reasoning_model = False
    if model_source == "API" and 'R1' in api_model_name:
        is_reasoning_model = True
    elif model_source == "Local" and 'R1' in MODEL_PATHS[selected_model][1]:
        is_reasoning_model = True
    
    if not is_reasoning_model:
        return content
    
    # Case 1: Complete think block (with both opening and closing tags)
    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); '
            r'padding: 10px; border-radius: 10px;">'
            r'<summary style="font-weight:bold;">Reasoning (click to expand)</summary>\2</details>',
            content,
            flags=re.DOTALL
        )
    
    # Case 2: Incomplete think block (only opening tag, reasoning in progress)
    elif '<think>' in content and '</think>' not in content:
        content = re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); '
            r'padding: 10px; border-radius: 10px;">'
            r'<summary style="font-weight:bold;">Reasoning in progress...</summary>\1</details>',
            content,
            flags=re.DOTALL
        )
    
    # Case 3: Only closing tag (malformed, attempt to wrap preceding text)
    elif '<think>' not in content and '</think>' in content:
        content = re.sub(
            r'(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); '
            r'padding: 10px; border-radius: 10px;">'
            r'<summary style="font-weight:bold;">Reasoning (click to expand)</summary>\1</details>',
            content,
            flags=re.DOTALL
        )
    
    return content


@st.cache_resource
def load_model_tokenizer(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with Streamlit caching.
    
    Uses st.cache_resource to load model only once and reuse across re-runs.
    This significantly improves UI responsiveness and reduces memory usage.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer) loaded on configured device
    """
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.eval().to(device)
    return model, tokenizer


def clear_chat_messages() -> None:
    """
    Clear all chat messages from session state.
    
    Resets both messages and chat_messages to start a fresh conversation.
    Called when user clicks the "New Chat" button in the UI.
    """
    
    if "messages" in st.session_state:
        del st.session_state.messages
    if "chat_messages" in st.session_state:
        del st.session_state.chat_messages


def init_chat_messages() -> List[Dict[str, str]]:
    """
    Initialize or restore chat message display in the UI.
    
    Streamlit re-runs the script on every interaction. This function ensures
    the chat history persists across re-runs by restoring from session_state.
    
    For each assistant message, displays a delete button that allows removing
    the message and its corresponding user message.
    
    Returns:
        List of current messages from session state
    """
    
    if "messages" in st.session_state:
        # Restore existing messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    
                    # Delete button for removing this assistant response
                    if st.button("×", key=f"delete_{i}"):
                        # Remove both assistant and user messages
                        st.session_state.messages = st.session_state.messages[:i - 1]
                        st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                        st.rerun()
            else:
                # User message (right-aligned)
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;">'
                    f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px; '
                    f'background-color: #ddd; border-radius: 10px; color: black;">'
                    f'{message["content"]}</div></div>',
                    unsafe_allow_html=True
                )
    else:
        # Initialize empty message lists
        st.session_state.messages = []
        st.session_state.chat_messages = []
    
    return st.session_state.messages


def regenerate_answer() -> None:
    """
    Regenerate the last assistant response.
    
    Triggered when user clicks regenerate button. Stores the last user message
    and regenerates the response. The main() function detects this state and
    re-runs generation.
    """
    
    if hasattr(st.session_state, 'last_user_message'):
        # Clear messages and set regenerate flag
        st.session_state.regenerate = True
        st.rerun()


def delete_conversation(index: int) -> None:
    """
    Delete a conversation turn at specified index.
    
    Args:
        index: Index of message to delete (removes both Q and A)
    """
    
    st.session_state.messages = st.session_state.messages[:index - 1]
    st.session_state.chat_messages = st.session_state.chat_messages[:index - 1]
    st.rerun()


# Sidebar configuration
st.sidebar.title("Model Settings")

# History length slider (even numbers for Q+A pairs)
st.session_state.history_chat_num = st.sidebar.slider(
    "Historical Dialogue Rounds", 
    0, 6, 0, step=2,
    help="Number of conversation rounds to include as context (0 = no history)"
)

# Generation parameters
st.session_state.max_new_tokens = st.sidebar.slider(
    "Max New Tokens", 
    256, 8192, 8192, step=1,
    help="Maximum number of tokens to generate"
)

st.session_state.temperature = st.sidebar.slider(
    "Temperature", 
    0.6, 1.2, 0.85, step=0.01,
    help="Sampling temperature (lower = more deterministic, higher = more creative)"
)

# Model source selection
model_source = st.sidebar.radio("Model Source", ["Local Model", "API"], index=0)

if model_source == "API":
    # API configuration
    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8998/v1")
    api_model_id = st.sidebar.text_input("Model ID", value="minimind")
    api_model_name = st.sidebar.text_input("Model Name", value="MiniMind2")
    api_key = st.sidebar.text_input("API Key", value="none", type="password")
    slogan = f"Hi, I'm {api_model_name}"
else:
    # Local model selection
    MODEL_PATHS = {
        "MiniMind2-R1 (0.1B)": ["../MiniMind2-R1", "MiniMind2-R1"],
        "MiniMind2-Small-R1 (0.02B)": ["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
        "MiniMind2 (0.1B)": ["../MiniMind2", "MiniMind2"],
        "MiniMind2-MoE (0.15B)": ["../MiniMind2-MoE", "MiniMind2-MoE"],
        "MiniMind2-Small (0.02B)": ["../MiniMind2-Small", "MiniMind2-Small"]
    }
    
    selected_model = st.sidebar.selectbox(
        'Select Model', 
        list(MODEL_PATHS.keys()), 
        index=2  # Default to MiniMind2
    )
    model_path = MODEL_PATHS[selected_model][0]
    slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

# MiniMind logo and title
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    f'<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; '
    f'display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px;"> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    f'</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">'
    f'Content generated by AI, please verify carefully<br>'
    f'内容完全由AI生成，请务必仔细甄别'
    f'</span>'
    f'</div>',
    unsafe_allow_html=True
)


def setup_seed(seed: int) -> None:
    """
    Set random seeds for reproducible generation.
    
    Seeds Python, NumPy, and PyTorch (CPU, GPU, CUDA) for deterministic
    behavior. Also configures cuDNN for reproducibility.
    
    Args:
        seed: Random seed value
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """
    Main chat interface application.
    
    The application flow:
    1. Load selected model (local) or configure API client
    2. Initialize/restore chat messages from session state
    3. Display chat history with delete buttons
    4. Wait for user input via chat_input
    5. Generate response using either local model or API
    6. Stream tokens in real-time for responsive UI
    7. Store completed response in session state
    
    Streamlit Note:
    This function re-runs on every user interaction. Session state is used
    to persist chat history and model instances across re-runs.
    """
    
    # Load model and tokenizer (only once due to @st.cache_resource)
    if model_source == "Local Model":
        model, tokenizer = load_model_tokenizer(model_path)
    else:
        model, tokenizer = None, None

    # Initialize chat messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # Display chat history
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                
                # Delete button to remove this assistant response
                if st.button("×", key=f"delete_{i}"):
                    # Remove both assistant and user messages
                    st.session_state.messages = st.session_state.messages[:-2]
                    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                    st.rerun()
        else:
            # User message (right-aligned)
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;">'
                f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px; '
                f'background-color: gray; border-radius: 10px; color: white;">'
                f'{message["content"]}</div></div>',
                unsafe_allow_html=True
            )

    # Chat input for user message
    prompt = st.chat_input(key="input", placeholder="Send a message to MiniMind")

    # Check for regenerate state (triggered by regenerate button)
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')

    if prompt:
        # Display user message
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;">'
            f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px; '
            f'background-color: gray; border-radius: 10px; color: white;">'
            f'{prompt}</div></div>',
            unsafe_allow_html=True
        )
        
        # Add to message history (truncated to max_new_tokens)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        # Generate assistant response
        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            if model_source == "API":
                # API mode: call remote endpoint
                try:
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=api_key,
                        base_url=api_url
                    )
                    
                    # Prepare conversation history (include system + recent messages)
                    history_num = st.session_state.history_chat_num + 1
                    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]
                    
                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )

                    # Stream tokens from API
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

                except Exception as e:
                    answer = f"Error calling API: {str(e)}"
                    placeholder.markdown(answer, unsafe_allow_html=True)
            
            else:
                # Local model mode
                # Set random seed for reproducibility
                random_seed = random.randint(0, 2 ** 32 - 1)
                setup_seed(random_seed)

                # Prepare prompt with chat template
                history_num = st.session_state.history_chat_num + 1
                st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-history_num:]
                
                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    new_prompt,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                # Setup streaming generation
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                    "num_return_sequences": 1,
                    "do_sample": True,
                    "attention_mask": inputs.attention_mask,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "temperature": st.session_state.temperature,
                    "top_p": 0.85,
                    "streamer": streamer,
                }

                # Run generation in separate thread
                Thread(target=model.generate, kwargs=generation_kwargs).start()

                # Stream tokens in real-time
                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            # Store assistant response
            messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            
            # Add delete button for the just-generated response
            with st.empty():
                if st.button("×", key=f"delete_{len(messages) - 1}"):
                    # Remove last Q&A pair
                    st.session_state.messages = st.session_state.messages[:-2]
                    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                    st.rerun()


if __name__ == "__main__":
    main()