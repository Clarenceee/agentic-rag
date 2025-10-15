import streamlit as st

# Temporary placeholder for users : will update to Postgres
VALID_USERS = {
    "admin": "admin",
    "clarence": "clarence",
}


def login_form():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

        if submitted:
            if VALID_USERS.get(username) == password:
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()  # Forces to reexcutes the main function
            else:
                st.error("Invalid username or password")


def logout():
    if st.session_state.get("username"):
        del st.session_state.username
        st.success("Logged out")
        st.rerun()


def setup_page():
    st.set_page_config(page_title="NBA Rules Assistant", page_icon="🏀", layout="centered")
    st.title("🏀 NBA Rules Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI rules assistant. How can I help you today?",
            }
        ]


def render_sidebar():
    with st.sidebar:
        # st.image("public/favicon.jpg", width=40)
        if "username" in st.session_state:
            st.markdown(f"**Logged in as:** {st.session_state.username}")
            if st.button("Logout", key="sidebar_logout"):
                logout()
            st.divider()

        st.markdown("## About")
        st.markdown("Welcome to the **NBA Rules Assistant**!")
        st.markdown("This AI assistant can answer your questions about NBA rules and regulations.")
        st.divider()

        st.markdown("No chat history")


def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        avatar = "public/favicon.jpg" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
