from dotenv import load_dotenv
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.agents import AgentType, initialize_agent
from huggingface_llm import HuggingfaceLlm


index_name = "langchain-doc-index"
load_dotenv()

def ingest_jira_docs():
    jira = JiraAPIWrapper()

    # # Define your JQL query to search for issues
    # jql_query = "project = SCRUM AND assignee = currentUser()"
    # # Retrieve issues using the Jira API wrapper
    # issues = jira.search(jql_query)
    # print(issues)

    toolkit = JiraToolkit.from_jira_api_wrapper(jira)

    hfLlm = HuggingfaceLlm()
    hfLlm.init_llm()

    agent = initialize_agent(toolkit.get_tools(), hfLlm.get_llm(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("can you fetch the issues assigned to me ?")
    

if __name__ == "__main__":
    print("Hello world")
    load_dotenv()
    ingest_jira_docs()
