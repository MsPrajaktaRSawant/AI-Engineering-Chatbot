
import streamlit as st

from rag.rag_pipeline import (
    create_rag_pipeline,
    ask_question_stream
)


# CONFIGURE STREAMLIT PAGE

st.set_page_config(
    page_title="AI Engineering Chatbot",
    page_icon="🤖",
    layout="centered"
)


# LOAD RAG PIPELINE

@st.cache_resource
def load_rag_pipeline():
    return create_rag_pipeline()


try:
    with st.spinner("Loading AI Engineering knowledge base..."):
        chunks, index = load_rag_pipeline()

except Exception as e:
    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()


# SYSTEM PROMPT

system_prompt = """
You are an AI Engineering tutor.

Your responsibilities:
1. Explain technical concepts in simple language.
2. Provide practical examples.
3. Include Python code when relevant.
4. Structure answers using headings and bullet points.
5. Focus on AI, Machine Learning, LLMs,
   Python, and software engineering.
6. Politely redirect unrelated questions.
"""


# INITIALIZE CONVERSATION MEMORY

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


# APPLICATION TITLE


st.markdown("""
<div style="
    padding: 22px 18px;
    border-radius: 16px;
    background: linear-gradient(120deg, #111827, #1E3150, #172554);
    border: 1px solid rgba(100,160,255,0.25);
    text-align: center;
    margin-bottom: 20px;
">
    <h1 style="
        margin: 0;
        font-size: 43px;
        font-weight: 850;
        letter-spacing: -1.5px;
        color: #FFFFFF;
    ">
        AI <span style="
            background: linear-gradient(90deg, #60A5FA, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">SPARK ⚡</span>
    </h1>  
    YOUR AI ENGINEERING COMPANION
</div>
""", unsafe_allow_html=True)



# DISPLAY PREVIOUS CONVERSATION

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# FIXED CHAT INPUT WITH CLEAR BUTTON

st.markdown(
    """
    <style>
    .st-key-chat_composer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: min(850px, calc(100% - 40px));
        z-index: 1000;
        background: var(--background-color);
        padding: 10px 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
    }

    .st-key-chat_composer [data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 10px;
    }

    .st-key-chat_composer [data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    .st-key-chat_composer input {
        height: 44px;
        border-radius: 10px;
    }

    .st-key-chat_composer button {
        height: 44px;
        border-radius: 10px;
        font-weight: 600;
        white-space: nowrap;
    }

    .main .block-container {
        padding-bottom: 120px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# CHAT INPUT

user_prompt = None
send_clicked = False
clear_clicked = False

with st.container(key="chat_composer"):

    with st.form("chat_form", clear_on_submit=True):

        input_col, send_col, clear_col = st.columns(
            [7, 0.5, 0.5]
        )

        with input_col:
            user_prompt = st.text_input(
                "Message",
                placeholder="Ask me anything about AI Engineering...",
                label_visibility="collapsed"
            )

        with send_col:
            send_clicked = st.form_submit_button(
                "➤",
                use_container_width=True
            )

        with clear_col:
            clear_clicked = st.form_submit_button(
                "🗑️",
                type="secondary",
                use_container_width=True
            )


# CLEAR CONVERSATION

if clear_clicked:

    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    st.rerun()


# PROCESS SUBMITTED QUESTION

if send_clicked and user_prompt and user_prompt.strip():

    user_prompt = user_prompt.strip()

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:

        # Generate assistant response
        with st.chat_message("assistant"):

            answer = st.write_stream(
                ask_question_stream(
                    user_prompt,
                    chunks,
                    index,
                    conversation_history=[
                        message
                        for message in st.session_state.messages
                        if message["role"] in ["user", "assistant"]
                    ]
                )
            )

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:

        # Remove failed user message
        st.session_state.messages.pop()

        st.error(f"Something went wrong: {e}")
