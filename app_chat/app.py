from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits import create_sql_agent
import functools
from langchain_core.pydantic_v1 import BaseModel
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START


from typing import Literal
from langchain_core.prompts import (
 ChatPromptTemplate,
 FewShotPromptTemplate,
 MessagesPlaceholder,
 PromptTemplate,
 SystemMessagePromptTemplate,
)

llm = ChatOllama( model_name = 'llama3.2', temperature = 0)

def set_prompts(examples, prompt):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    K = 5,
    input_keys = ["input"],)
    
    few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = PromptTemplate.from_template(
    "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect",],
    prefix = prompt,
    suffix="",)

    full_prompt = ChatPromptTemplate.from_messages(
    [
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
    ])
    return full_prompt


system_prefix = """You are an MS SQL agent designed to interact with a SQL database. Given an
input question, create a syntactically correct {dialect} query to run.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. """

sales_agent_prompt = system_prefix + """You are an agent designed to answer only sales & 
order related queries. Never answer anything out of sales domain.
Consider domestic orders as open market orders.
Be vigilant in identifying customer names from user query.
Here are some examples of user inputs and their corresponding SQL queries: """

sales_agent_examples = [ 
{
"input": " Total revenue for order # 1234 ",
"query": " Select sum(total_price) as total_revenue from order_table where order_number = 123;"
}
]

sales_full_prompt = set_prompts(sales_agent_examples, sales_agent_prompt)

agent_sales = create_sql_agent(
 llm = llm,
 db = sales_db_engine,
 #extra_tools = [tool_1, tool_2], ( If there are any tools)
 prompt = sales_full_prompt,
 verbose = True,
 agent_type = "Llama-tool",
)


members = ["Sales"]
options = ["FINISH"] + members
def agent_node(state, agent, name):
    input_msg = state['messages'][0][-1]
    result = agent.invoke(input_msg)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


class routeResponse(BaseModel):
 """Responsible for letting supervisor decide which node to call next """
 next: Literal["FINISH", "Sales", "Freight"]

supervisor_prompt = (
 "You are a supervisor tasked with managing a conversation between the"
 " following workers : {members}. Given the following user request,"
 " respond with the worker to act next. Each worker will perform a"
 " task and respond with their results and status. When finished,"
 " respond with FINISH. After a worker will perform task, respond with FINISH."
)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", supervisor_prompt),
    MessagesPlaceholder(variable_name="messages"),
    (
    "system", 
    "If you have answered the query"
    "Select FINISH",
    ),
    ]
    ).partial(options=str(options), members=", ".join(members))

def supervisor_agent(state):
    supervisor_chain = (
    prompt
    | llm.with_structured_output(routeResponse)
    )


#Creating workflow using graph state
workflow = StateGraph(AgentState)
#Creating nodes of graph
sales_node = functools.partial(agent_node, agent = agent_sales, name="Sales")
freight_node = functools.partial(agent_node, agent = agent_freight, name="Freight")
workflow.add_node("Sales", sales_node)
workflow.add_node("Freight", freight_node)
workflow.add_node("supervisor", supervisor_agent)

#Adding edges in graph for communication 
for member in members:
 # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")
graph = workflow.compile()