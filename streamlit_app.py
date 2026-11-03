
import streamlit as st

from rag.rag_pipeline import (
    create_rag_pipeline,
    ask_question_stream
)


# ==========================================
# 1. CONFIGURE STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="AI Engineering Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# 2. LOAD RAG PIPELINE
# ==========================================

@st.cache_resource
def load_rag_pipeline():

    chunks, index = create_rag_pipeline()

    return chunks, index


try:

    with st.spinner("Loading AI Engineering knowledge base..."):

        chunks, index = load_rag_pipeline()

except Exception as e:

    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()


# ==========================================
# 3. SYSTEM PROMPT
# ==========================================

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


# ==========================================
# 4. INITIALIZE CONVERSATION MEMORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


# ==========================================
# 5. APPLICATION TITLE
# ==========================================

st.title("🤖 AI Engineering Chatbot")

st.caption(
    "Your personal AI Engineering tutor powered by RAG"
)

# ==========================================
# 6. DISPLAY PREVIOUS CONVERSATION
# ==========================================

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        st.markdown(message["content"])




# ==========================================
# 7. FIXED CHAT INPUT WITH CLEAR BUTTON
# ==========================================

st.markdown(
    """
    <style>
    .st-key-chat_composer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: min(736px, calc(100% - 40px));
        z-index: 1000;
        background: var(--background-color);
        padding: 8px 0;
    }

    .st-key-chat_composer [data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    .st-key-chat_composer [data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    .st-key-chat_composer button {
        height: 44px;
    }

    .st-key-chat_composer input {
        height: 44px;
    }

    /* Leave room for the fixed composer */
    .main .block-container {
        padding-bottom: 110px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

user_prompt = None

with st.container(key="chat_composer"):

    input_col, clear_col = st.columns([6, 1])

    with input_col:

        with st.form("chat_form", clear_on_submit=True):

            user_prompt = st.text_input(
                "Message",
                placeholder="Ask me anything about AI Engineering...",
                label_visibility="collapsed"
            )

            submitted = st.form_submit_button(
                "Send",
                use_container_width=True
            )

    with clear_col:

        if st.button("Clear", use_container_width=True):

            st.session_state.messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

            st.rerun()

if not submitted:
    user_prompt = None

# ==========================================
# 8. GENERATE RAG RESPONSE
# ==========================================

if user_prompt:

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
                    index
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
