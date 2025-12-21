from langchain_openai import ChatOpenAI


def web_search(query):
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        max_tokens=200,
    )
    tool = {"type": "web_search_preview"}
    llm_with_tools = llm.bind_tools([tool])
    enhanced_query = query + "\n Return the response to be as concise as possible"
    print(f"Searching with query : {enhanced_query}")
    response = llm_with_tools.invoke(enhanced_query)
    return response
