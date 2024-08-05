# Basic imports
import os
import functools
import operator
from typing import Annotated, List, Dict, Optional
from typing_extensions import TypedDict

# LangChain & LangGraph imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph, START


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

###### ATT ######
# The above API is for US-accounts, if you have an EU account you have use this one:
# os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com/"


project_name = "Market Research Team"  # Update with your project name
os.environ["LANGCHAIN_PROJECT"] = project_name  # Optional: "default" is used if not set

# Define a persistent working directory
import platform
from pathlib import Path



# Check if running in Docker
RUNNING_IN_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

# Detect the operating system
operating_system = platform.system()

# Define a persistent working directory based on the environment and operating system
if RUNNING_IN_DOCKER:
    WORKING_DIRECTORY = Path("/app/working_directory")  # Adjust this path if needed
else:
    if operating_system == "Darwin" or operating_system == "Windows":
        WORKING_DIRECTORY = Path(__file__).parent / "working_directory"
    elif operating_system == "Linux":
        WORKING_DIRECTORY = Path("/content/working_directory")
    else:
        raise ValueError(f"Unsupported operating system: {operating_system}")

# Ensure the working directory exists
if not WORKING_DIRECTORY.exists():
    WORKING_DIRECTORY.mkdir(parents=True)
    print(f"Created working directory: {WORKING_DIRECTORY}")
else:
    print(f"Working directory already exists: {WORKING_DIRECTORY}")


# Create Tools
@tool("patent_search")
def patent_search(query: str) -> str:
    """Search with Google SERP API by a query to find news about patents related to the query."""
    from langchain_community.utilities import SerpAPIWrapper
    params = {
        "engine": "google_patents",
        "gl": "us",
        "hl": "en",
    }
    patent_search = SerpAPIWrapper(params=params)
    return patent_search.run(query)


@tool("exa_search")
def exa_search(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    from exa_py import Exa
    exa = Exa()

    response = exa.search_and_contents(
        question,
        type="neural",
        use_autoprompt=True,
        num_results=3,
        highlights=True
    )

    parsedResult = ''.join([f'<Title id={idx}>{eachResult.title}</Title><URL id={idx}>{eachResult.url}</URL><Highlight id={idx}>{"".join(eachResult.highlights)}</Highlight>' for (idx, eachResult) in enumerate(response.results)])

    return parsedResult

# Load Tavily Search Wrapper from LangChain
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced"
)

# Document Tools
@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"

# Create Agent and Team Supervisor
def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    system_prompt += " Do not ask for clarification."
    system_prompt += " Your other team members (and other teams) will collaborate with you with their own specialties."
    system_prompt += " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

# Define Graph State
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str

def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir(parents=True)
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
            if f.is_file()
        ]
    except Exception as e:
        print(f"Error reading files: {e}")
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Define the Agents
web_search_agent = create_agent(
    llm,
    [tavily_tool],
    "You are a research assistant who can search for up-to-date info using the Tavily Search Engine.",
)
search_node = functools.partial(agent_node, agent=web_search_agent, name="Search")

exa_search_agent = create_agent(
    llm,
    [exa_search],
    "You are a research assistant who can search for up-to-date info on Exa Search. Your response should clearly articulate the key points.",
)
exa_search_node = functools.partial(agent_node, agent=exa_search_agent, name="ExaSearch")

patent_search_agent = create_agent(
    llm,
    [patent_search],
    "You are a market research assistant, very knowledgeable in patent research to find up-to-date info about patents using the Google patents API.",
)
patent_search_node = functools.partial(agent_node, agent=patent_search_agent, name="PatentSearch")

doc_writer_agent = create_agent(
    llm,
    [write_document, edit_document, read_document],
    "You are an expert in writing market research white papers for your product development team.\n"
    "Below are files currently in your directory:\n{current_files}",
)
context_aware_doc_writer_agent = prelude | doc_writer_agent
doc_writing_node = functools.partial(
    agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
)

note_taking_agent = create_agent(
    llm,
    [create_outline, read_document],
    "You are an expert Senior Market Research Analyst tasked with writing a paper outline and"
    " taking notes to craft a perfect paper which provides insights to inform product development.{current_files}",
)
context_aware_note_taking_agent = prelude | note_taking_agent
note_taking_node = functools.partial(
    agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
)

research_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["DocWriter", "Search", "ExaSearch", "PatentSearch"],
)

# Create the Graph
authoring_graph = StateGraph(ResearchTeamState)
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("Search", search_node)
authoring_graph.add_node("ExaSearch", exa_search_node)
authoring_graph.add_node("PatentSearch", patent_search_node)
authoring_graph.add_node("supervisor", research_writing_supervisor)

authoring_graph.add_edge("DocWriter", "supervisor")
authoring_graph.add_edge("PatentSearch", "supervisor")
authoring_graph.add_edge("Search", "supervisor")
authoring_graph.add_edge("ExaSearch", "supervisor")

authoring_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "DocWriter": "DocWriter",
        "Search": "Search",
        "PatentSearch": "PatentSearch",
        "ExaSearch": "ExaSearch",
        "FINISH": END,
    },
)

authoring_graph.add_edge(START, "supervisor")
graph = authoring_graph.compile()


