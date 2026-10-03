import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Sub-agents import karein
from agents.rag_agent import run_rag_agent
from agents.github_agent import run_github_agent
from agents.google_agent import run_google_agent

load_dotenv()

# Graph ki State define karein
class MultiAgentState(TypedDict):
    question: str
    destination: str
    generation: str

# LLM Router initialize karein jo decide karega ke task kahan bhejna hai
router_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# ---------------------------------------------------------
# 1. Router Node: Faisla karega ke kahan jana hai
# ---------------------------------------------------------
def router_node(state: MultiAgentState):
    print("\n--- [ROUTER]: Analyzing user query ---")
    query = state["question"]
    
    prompt = f"""You are a smart router for a multi-agent system. 
    Analyze the user query and choose ONE destination strictly from these three:
    - "rag" (if the query is about documents, uploaded PDF, notes, or general knowledge search)
    - "github" (if the query is about GitHub repositories, issues, PRs, or code hosting)
    - "google" (if the query is about calendar meetings, scheduling, reading/writing emails, or drafting emails)
    
    User Query: {query}
    Destination (only write rag, github, or google):"""
    
    response = router_llm.invoke(prompt)
    destination = response.content.strip().lower()
    
    # Fallback agar router kuch aur likh de
    if "github" in destination:
        dest = "github"
    elif "google" in destination or "email" in destination or "calendar" in destination:
        dest = "google"
    else:
        dest = "rag"
        
    print(f"--- [ROUTER ROUTED TO]: {dest.upper()} ---")
    return {"destination": dest}

# ---------------------------------------------------------
# 2. Sub-Agent Execution Nodes
# ---------------------------------------------------------
def call_rag(state: MultiAgentState):
    ans = run_rag_agent(state["question"])
    return {"generation": ans}

def call_github(state: MultiAgentState):
    ans = run_github_agent(state["question"])
    return {"generation": ans}

def call_google(state: MultiAgentState):
    ans = run_google_agent(state["question"])
    return {"generation": ans}

# ---------------------------------------------------------
# 3. Conditional Routing Logic
# ---------------------------------------------------------
def route_decision(state: MultiAgentState) -> Literal["call_rag", "call_github", "call_google"]:
    dest = state["destination"]
    if dest == "github":
        return "call_github"
    elif dest == "google":
        return "call_google"
    else:
        return "call_rag"

# ---------------------------------------------------------
# 4. Build LangGraph Workflow
# ---------------------------------------------------------
workflow = StateGraph(MultiAgentState)

# Nodes add karein
workflow.add_node("router", router_node)
workflow.add_node("call_rag", call_rag)
workflow.add_node("call_github", call_github)
workflow.add_node("call_google", call_google)

# Entry point set karein
workflow.set_entry_point("router")

# Conditional edges (Router se sahi agent ki taraf rasta)
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "call_rag": "call_rag",
        "call_github": "call_github",
        "call_google": "call_google"
    }
)

# Sabhi agents kaam khatam karke END ho jayenge
workflow.add_edge("call_rag", END)
workflow.add_edge("call_github", END)
workflow.add_edge("call_google", END)

# Graph Compile karein
app = workflow.compile()

# ---------------------------------------------------------
# 5. Run & Test via Interactive Terminal Input
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==================================================")
    print("🚀 LANGGRAPH MULTI-AGENT SYSTEM (RAG + GITHUB + GOOGLE)")
    print("==================================================")
    
    while True:
        user_query = input("\nApna sawal ya task yahan likhein (ya 'exit' likhein band karne ke liye): ")
        
        if user_query.lower() == 'exit':
            print("System band ho raha hai. Khuda hafiz!")
            break
            
        if not user_query.strip():
            continue
            
        inputs = {"question": user_query}
        
        for output in app.stream(inputs):
            for key, value in output.items():
                if key != "router":
                    print(f"\nFinal Agent Output:\n{value.get('generation')}")