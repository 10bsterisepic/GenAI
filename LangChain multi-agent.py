pip install langchain transformers
pip install -q langchain-community

#load HuggingFace local model
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

hf_pipeline=pipeline('text-generation', model='distilgpt2', max_new_tokens=150)
llm=HuggingFacePipeline(pipeline=hf_pipeline)
#Define custom tool functions
def destination_planner_tool(query: str)->str:
    return "Suggested destination: Paris, known for its ruch culture and exquisite cuisine."

def budget_analyst_tool(query: str)->str:
    return "Estimated budget: ₹65,000 for 5 days including flight, stay, food and local travel."

def activity_planner_tool(query: str)->str:
    return (
        "Day 1: Eiffel Tower and Seine River Cruise\n"
        "Day 2: Louvre Museum and Montmartre\n"
        "Day 3: Day trip to Versailles\n"
        "Day 4: Notre-Dame and Latin Quarter\n"
        "Day 5: Shopping and Departure"
    )
#Stimulate multi-agent collaboration
def multi_agent_travel_planner():
    print("🧠 Destination Planner Agent:")
    destination = destination_planner_tool("Looking for a cultural food trip in Europe")
    print(destination)

    print("\n💰Budget Analyst Agent:")
    budget = budget_analyst_tool("5-day trip to Paris")
    print(budget)

    print("\n📊Activity Planner Tool:")
    activity = activity_planner_tool("5-day trip to Paris")
    print(activity)

multi_agent_travel_planner()
