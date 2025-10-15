import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from orchestrator.main_graph_node import MainGraph
from states.graph_states import OverallState, ContextSchema
from ui.utilities import render_sidebar, setup_page, display_chat_history, login_form
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # Show login if not authenticated
    if "username" not in st.session_state:
        login_form()
        return

    setup_page()
    render_sidebar()
    display_chat_history()

    # Setup Main Graph for session
    if "main_graph" not in st.session_state:
        st.session_state["main_graph"] = MainGraph()
    main_graph = st.session_state["main_graph"].graph

    # Handle New Messages
    if prompt := st.chat_input("Type your question here..."):
        logger.info(f"Received input message : {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="public/favicon.jpg"):
            with st.spinner("Thinking..."):
                try:
                    langchain_messages = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            langchain_messages.append(AIMessage(content=msg["content"]))

                    input_state = OverallState(
                        query=prompt,
                        messages=langchain_messages,
                        input_guardrails=False,
                        use_rag=False,
                        formatted_query=None,
                        sub_results=[],
                    )

                    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                    context = ContextSchema(user_id=st.session_state.username)
                    logger.info(f"Input State : {input_state}")
                    logger.info(f"Context : {context}")
                    output = main_graph.invoke(
                        input_state,
                        config=config,
                        context=context,
                    )
                    # Log output without sub_results for cleaner logs
                    log_output = {k: v for k, v in output.items() if k != "sub_results"}
                    logger.info(f"LangGraph Output: {log_output}")
                    logger.info(f"Chat Response: {output['final_result']}")
                    final_response = output["final_result"]
                    st.markdown(final_response)

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    final_response = error_msg

        st.session_state.messages.append({"role": "assistant", "content": final_response})


if __name__ == "__main__":
    main()

# Command to start the app
# streamlit run main.py --server.runOnSave true
