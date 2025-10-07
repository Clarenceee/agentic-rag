import streamlit as st
from process.main import NbaRag

st.set_page_config(page_title="NBA Rules Assistant", page_icon="🏀", layout="centered")

# --- Sidebar Info ---
with st.sidebar:
    st.image("public/favicon.jpg", width=80)
    st.markdown("## 📖 About")
    st.markdown(
        """
    Welcome to the **NBA Rules Assistant**!
    Ask me any question about NBA rules and I'll help clarify.

    **Created by:** Clarence
    **Powered by:** RAG pipeline
    """
    )
    st.divider()
    st.markdown("⚡ Tip: Try asking *What is a travel in basketball?*")

# --- Main App Layout ---
st.title("NBA Rules Assistant 🏀")
st.write("Hello! I'm your NBA rules assistant.")

# --- Initialize RAG Instance ---
if "rag_instance" not in st.session_state:
    st.session_state["rag_instance"] = NbaRag(user_id="Clarence")

rag_instance = st.session_state["rag_instance"]

# --- Chat Input & Response ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for chat in st.session_state.messages:
    #  This insert a chat message container so that can display messages
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with RAG
    with st.chat_message("assistant", avatar="public/favicon.jpg"):
        with st.spinner("Thinking..."):
            response = rag_instance.main(query=prompt)
            if hasattr(response, "__iter__") and not isinstance(response, str):
                # Handle stream-like objects
                final_response = st.write_stream(response)
            else:
                # Handle string responses
                st.markdown(response)
                final_response = response

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_response})

# Command to start the app
# streamlit run main.py --server.runOnSave true
