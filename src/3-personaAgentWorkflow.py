import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------
# Load API KEY from secrets.toml
# -----------------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# You can choose: "gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5-mini" ...
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# -----------------------------------------------------------
# 1. Blob Writing Agent
# -----------------------------------------------------------
blob_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Blob Writing Agent. You write long-form, casual, unstructured content about a topic."),
    ("user", "Write a blob about: {topic}")
])

blob_agent = blob_prompt | llm

# -----------------------------------------------------------
# 2. SEO Checking Agent
# -----------------------------------------------------------
seo_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an SEO Checking Agent. Analyze content and provide optimization suggestions."),
    ("user", "Content:\n{content}\n\nGive SEO analysis and improvements.")
])

seo_agent = seo_prompt | llm

# -----------------------------------------------------------
# 3. Fact Checking Agent
# -----------------------------------------------------------
fact_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Fact Checking Agent. Identify factual claims and verify them."),
    ("user", "Fact-check the following content:\n\n{content}")
])

fact_agent = fact_prompt | llm

# -----------------------------------------------------------
# 4. Manager Agent (Coordinator)
# -----------------------------------------------------------
def manager_chain(topic):
    # Step 1: blob writing
    blob = blob_agent.invoke({"topic": topic}).content

    # Step 2: SEO check
    seo = seo_agent.invoke({"content": blob}).content

    # Step 3: fact check
    facts = fact_agent.invoke({"content": blob}).content

    # Step 4: Manager summary
    summary = llm.invoke([
        ("system", "Combine all workflow outputs into a final structured report."),
        ("user", f"""
        Topic: {topic}

        === Blob Content ===
        {blob}

        === SEO Analysis ===
        {seo}

        === Fact Check ===
        {facts}

        Create a final summarized Manager Report.
        """)
    ]).content

    return blob, seo, facts, summary


# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.title("Multi-Agent Workflow with OpenAI + LangChain")

topic = st.text_input("Enter a topic")

if st.button("Run Workflow"):
    blob, seo, facts, report = manager_chain(topic)

    st.subheader("📝 Blob Content")
    st.write(blob)

    st.subheader("🔎 SEO Analysis")
    st.write(seo)

    st.subheader("✔️ Fact Checking")
    st.write(facts)

    st.subheader("📘 Final Manager Report")
    st.write(report)
