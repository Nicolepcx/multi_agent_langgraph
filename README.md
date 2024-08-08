# Market Research Agent with LangGraph


This is the repo for the event [Prompt to Product from O'Reilly Media](oreillymedia.pxf.io/DKPdgq). This repo draws inspiration from the paper [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) by Wu et al., and from the [examples from LangGraph](https://github.com/langchain-ai/langgraph/tree/main/examples/multi_agent). 

This repo to construct a Market Research Team and by including the following tools:

- [Taviliy](https://tavily.com/) for web search, get your API key here.
- [Exa](https://exa.ai/search), after account login, get your API key here. To find the exact content you’re looking for on the web using embeddings-based search.
- [SerpApi](https://serpapi.com/) here, after account login, get your API key to do look for existing patents.
- Tools to access and write to a .txt file.

The interaction will look like this:

![Research_Agent_LangGraph-Overview.png](resources%2FResearch_Agent_LangGraph-Overview.png)


You will also learn how to:

- Define utilities to help create the graph.
- Create a team supervisor and a team of agents.


## Overall Workflow

- The Project Manager Agent (supervisor) assigns tasks to the appropriate agents.
- The Patent Research Agent and two Internet Research Agent to gather external data.
- The Document Writer Agent compiles the research into a document.
    
## Possible Extensions: 
- The Internal Product Research Agent collects internal product information.
- The Review and Editing Agent reviews and refines the document for final submission.


In addition, you can use this setup as a boilerplate to develop a Recruiting Agent, by just following the documentation [here](https://docs.exa.ai/reference/exa-recruiting-agent). 

You can either run the agents via the [notebook](market_research_team_LangGraph.ipynb) or you can use [LangGraph studio](https://github.com/langchain-ai/langgraph-studio), where you can also debug your agents. You need to have Docker installed on your computer to be able to run LangGraph studio. If you run the application via LangGraph Studio, two folders will be automatically created, one called `app` and one called `working_directory`. Here the produced report will be stored. To access the file via Docker, you can use Docker Desktop and navigate to your Docker Container for your agent, then to `app` and then to the `working_directory` folder as shown here:

![Access_document_Dcoker.png](resources%2FAccess_document_Dcoker.png)

If you are working with Google Colab, you can access the folder by clicking on the folder icon and then on the `working_directory` folder and download the report to your computer. 
