import streamlit as st
from snowflake.core import Root # requires snowflake>=0.8.0
from snowflake.cortex import complete
from snowflake.snowpark import Session



MODELS = [
    "mistral-large2",
    "llama3.1-70b",
    "llama3.1-8b",
    "openai-gpt-4.1"
]

def init_messages():
    """
    Initialize the session state for chat messages. If the session state indicates that the
    conversation should be cleared or if the "messages" key is not in the session state,
    initialize it as an empty list.
    """
    session = get_active_session()
    if st.session_state.get("clear_conversation", False) or "messages" not in st.session_state:
        st.session_state.messages = []



def init_service_metadata():
    """
    Initialize the session state for cortex search service metadata.
    Runs only once and ensures a default service is always selected.
    """
    session = get_active_session()
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES IN SCHEMA POLICYBOT_DB.EMPLOYEE;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append(
                    {"name": svc_name, "search_column": svc_search_col}
                )

        st.session_state.service_metadata = service_metadata

    # ✅ Ensure default service is set at startup
    if "selected_cortex_search_service" not in st.session_state:
        default_service = "SVC_EMP_BOT"   # 👈 change default here
        if any(s["name"] == default_service for s in st.session_state.service_metadata):
            st.session_state.selected_cortex_search_service = default_service
        elif st.session_state.service_metadata:
            st.session_state.selected_cortex_search_service = st.session_state.service_metadata[0]["name"]




def init_config_options():
    """
    Render the settings panel for the chatbot.
    Allows selecting a cortex search service, clearing conversation,
    toggling debug mode, and setting advanced options.
    """
    session = get_active_session()
    with st.expander("⚙️ Settings", expanded=False):
        service_names = [s["name"] for s in st.session_state.service_metadata]

        st.selectbox(
            "Select cortex search service:",
            service_names,
            index=service_names.index(st.session_state.selected_cortex_search_service),
            key="selected_cortex_search_service",
        )

        st.button("Clear conversation", key="clear_conversation")
        st.toggle("Debug", key="debug", value=False)
        st.toggle("Use chat history", key="use_chat_history", value=True)

        with st.expander("Advanced options"):
            st.selectbox("Select model:", MODELS, key="model_name")
            st.number_input(
                "Select number of context chunks",
                value=5,
                key="num_retrieved_chunks",
                min_value=1,
                max_value=10,
            )
            st.number_input(
                "Select number of messages to use in chat history",
                value=5,
                key="num_chat_messages",
                min_value=1,
                max_value=10,
            )

        st.expander("Session State").write(st.session_state)


def query_cortex_search_service(query, columns = [], filter={}):
    """
    Query the selected cortex search service with the given query and retrieve context documents.
    Display the retrieved context documents in the sidebar if debug mode is enabled. Return the
    context documents as a string.

    Args:
        query (str): The query to search the cortex search service with.

    Returns:
        str: The concatenated string of context documents.
    """
    session = get_active_session()
    db, schema = session.get_current_database(), session.get_current_schema()

    cortex_search_service = (
        root.databases[db]
        .schemas[schema]
        .cortex_search_services[st.session_state.selected_cortex_search_service]
    )

    context_documents = cortex_search_service.search(
        query, columns=columns, filter=filter, limit=st.session_state.num_retrieved_chunks
    )
    results = context_documents.results

    service_metadata = st.session_state.service_metadata
    search_col = [s["search_column"] for s in service_metadata
                    if s["name"] == st.session_state.selected_cortex_search_service][0].lower()

    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"

    if st.session_state.debug:
        st.sidebar.text_area("Context documents", context_str, height=500)

    return context_str, results


def get_chat_history():
    """
    Retrieve the chat history from the session state limited to the number of messages specified
    by the user in the sidebar options.

    Returns:
        list: The list of chat messages from the session state.
    """
    session = get_active_session()
    start_index = max(
        0, len(st.session_state.messages) - st.session_state.num_chat_messages
    )
    return st.session_state.messages[start_index : len(st.session_state.messages) - 1]


def completee(model, prompt):
    """
    Generate a completion for the given prompt using the specified model.

    Args:
        model (str): The name of the model to use for completion.
        prompt (str): The prompt to generate a completion for.

    Returns:
        str: The generated completion.
    """
    session = get_active_session()
    return complete(model, prompt).replace("$", r"\$")


def make_chat_history_summary(chat_history, question):
    """
    Generate a summary of the chat history combined with the current question to extend the query
    context. Use the language model to generate this summary.

    Args:
        chat_history (str): The chat history to include in the summary.
        question (str): The current user question to extend with the chat history.

    Returns:
        str: The generated summary of the chat history and question.
    """
    session = get_active_session()
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """

    summary = completee(st.session_state.model_name, prompt)

    if st.session_state.debug:
        st.sidebar.text_area(
            "Chat history summary", summary.replace("$", r"\$"), height=150
        )

    return summary


def create_prompt(user_question):
    """
    Create a prompt for the language model by combining the user question with context retrieved
    from the cortex search service and chat history (if enabled). Format the prompt according to
    the expected input format of the model.

    Args:
        user_question (str): The user's question to generate a prompt for.

    Returns:
        str: The generated prompt for the language model.
    """
    session = get_active_session()
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history != []:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context, results = query_cortex_search_service(
                question_summary,
                columns=["chunk", "file_url", "relative_path"],
                filter={"@and": [{"@eq": {"language": "English"}}]},
            )
        else:
            prompt_context, results = query_cortex_search_service(
                user_question,
                columns=["chunk", "file_url", "relative_path"],
                filter={"@and": [{"@eq": {"language": "English"}}]},
            )
    else:
        prompt_context, results = query_cortex_search_service(
            user_question,
            columns=["chunk", "file_url", "relative_path"],
            filter={"@and": [{"@eq": {"language": "English"}}]},
        )
        chat_history = ""

    prompt = f"""
            [INST]
            You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
            you will also be given context provided between <context> and </context> tags. Use that context
            with the user's chat history provided in the between <chat_history> and </chat_history> tags
            to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
            and directly relevant to the user's question.

            If the user asks a generic question which cannot be answered with the given context or chat_history,
            just say "I don't know the answer to that question.

            Don't saying things like "according to the provided context".

            <chat_history>
            {chat_history}
            </chat_history>
            <context>
            {prompt_context}
            </context>
            <question>
            {user_question}
            </question>
            [/INST]
            Answer:
            """
    return prompt, results


def main():
    st.title(f":speech_balloon: Hello, I am your IntelliHeal Tech AI Assistant. I am happy to assist you with your questions about IT policies and procedures")

    init_service_metadata()
    init_config_options()
    init_messages()

    icons = {"assistant": "💬", "user": "👤"}
     
    session = get_active_session()
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.markdown(message["content"])

    disable_chat = (
        "service_metadata" not in st.session_state
        or len(st.session_state.service_metadata) == 0
    )
    if question := st.chat_input("Ask a question...", disabled=disable_chat):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user", avatar=icons["user"]):
            st.markdown(question.replace("$", r"\$"))

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=icons["assistant"]):
            message_placeholder = st.empty()
            question = question.replace("'", "")
            prompt, results = create_prompt(question)
            with st.spinner("Thinking..."):
                generated_response = completee(
                    st.session_state.model_name, prompt
                )
                # build references list for citation
                # build references list for citation
                references = "###### References\n\n"
                seen_files = set()   # 👈 Track unique file paths

                for ref in results:
                   rel_path = ref['relative_path']
                   if rel_path not in seen_files:   # 👈 Only process unique files
                       seen_files.add(rel_path)
                       # Generate fresh presigned URL valid for 24 hours (86400 seconds)
                       presigned_url = session.sql(
                       f"SELECT GET_PRESIGNED_URL(@DOCS1, '{rel_path}', 86400)"
                       ).collect()[0][0]
                       references += f"- **{rel_path}** → [Open Document]({presigned_url})\n"


                message_placeholder.markdown(generated_response + "\n\n" + references, unsafe_allow_html=True)



        st.session_state.messages.append(
            {"role": "assistant", "content": generated_response}
        )


def get_snowflake_session():
    # get connection parameters from secrets
    params = st.secrets["connections"]["snowflake"]
    # build Snowpark session
    if "sf_session" not in st.session_state:
        st.session_state.sf_session = Session.builder.configs({
        	"account": params["account"],
        	"user": params["user"],
        	"password": params.get("password"),
        	"role": params.get("role"),
        	"warehouse": params.get("warehouse"),
        	"database": params.get("database"),
        	"schema": params.get("schema"),
    	}).create()
    return st.session_state.sf_session

if __name__ == "__main__":

    # @st.cache_resource(ttl="5m")
    # def get_session():
    	# return st.connection("snowflake").session()

    session = get_snowflake_session()
    root = Root(session)
    main()

