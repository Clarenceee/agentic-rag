import uuid
import streamlit as st
from langgraph.types import Command
from langfuse.langchain import CallbackHandler
from orchestrator.main_graph_node import MainGraph
from states.graph_states import OverallState, ContextSchema
from ui.utilities import render_sidebar, setup_page, display_chat_history, login_form
from utils.logger import get_logger

logger = get_logger(__name__)


@st.dialog("Confirm Action")
def confirm_action():
    """Function to handle user decision for HITL"""
    st.write("Do you want to proceed with the web search?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Accept", use_container_width=True):
            st.session_state.decision = "accept"
            st.session_state.session_messages[-1]["content"] = "User accepted the web search"
            # st.session_state.auto_trigger_search = True
            st.rerun()

    with col2:
        if st.button("❌ Reject", use_container_width=True):
            st.session_state.decision = "reject"
            st.session_state.session_messages[-1]["content"] = "User rejected the web search"
            st.rerun()


def main():
    # Show login if not authenticated
    if "username" not in st.session_state:
        login_form()
        return

    setup_page()
    render_sidebar()
    display_chat_history()

    # Initializations
    if "main_graph" not in st.session_state:
        st.session_state["main_graph"] = MainGraph()
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        logger.info(f"New session started with ID: {st.session_state['session_id']}")

    main_graph = st.session_state["main_graph"].graph

    langfuse_handler = CallbackHandler()
    config = {
        "thread_id": st.session_state["session_id"],
        "recursion_limit": 10,
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": st.session_state.username,
            "langfuse_tags": ["test-hitl"],
        },
    }
    context = ContextSchema(user_id=st.session_state.username)

    # Accepted HITL
    if "decision" in st.session_state and st.session_state.decision == "accept":
        del st.session_state.decision
        with st.chat_message("assistant", avatar="public/favicon.jpg"):
            with st.spinner("Searching the web..."):
                output = main_graph.invoke(Command(resume=True), config=config, context=context)
                final_response = output["messages"][-1].content
                st.markdown(final_response)
                st.session_state.session_messages[-1]["content"] = final_response
        st.rerun()

    # Rejected HITL
    elif "decision" in st.session_state and st.session_state.decision == "reject":
        del st.session_state.decision
        st.rerun()

    # Handle New Messages
    if prompt := st.chat_input("Type your question here..."):
        logger.info(f"Received input message : {prompt}")
        st.session_state.session_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="public/favicon.jpg"):
            with st.spinner("Thinking..."):
                try:
                    input_state = OverallState(
                        query=prompt,
                        input_guardrails=False,
                        use_rag=False,
                        formatted_query=None,
                        sub_results=[],
                    )

                    logger.info(f"Input State : {input_state}")
                    output = main_graph.invoke(
                        input_state,
                        config=config,
                        context=context,
                    )
                    log_output = {k: v for k, v in output.items() if k != "sub_results"}
                    logger.info(f"LangGraph Output: {log_output} \n\n")
                    logger.info("=" * 50)
                    logger.info(f"Query: {input_state.query}")
                    logger.info(f"Chat Response: {output['final_result']}")
                    if output.get("__interrupt__", []):
                        final_response = "Human Action needed for web search."
                        logger.info(f"Human Action needed : {output['__interrupt__'][0].value}")
                        confirm_action()
                    else:
                        final_response = output["messages"][-1].content
                        st.markdown(final_response)

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    final_response = error_msg
                st.session_state.session_messages.append(
                    {"role": "assistant", "content": final_response}
                )


if __name__ == "__main__":
    main()

# Command to start the app
# streamlit run main.py --server.runOnSave true
