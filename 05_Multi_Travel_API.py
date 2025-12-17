import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyAEG2kTUXa4jxKDwlE1eCFTxSWTMkMsdRQ'
# Import required modules
from crewai import Agent, Task, Crew, LLM
import os
# Load API key
gemini_key = os.getenv('GEMINI_API_KEY')
if gemini_key is None:
    raise ValueError("Please set GEMINI_API_KEY environment variable")
# Initialize shared LLM (all agents can share or optionally have separate ones)
llm = LLM(
model="gemini/gemini-2.0-flash", # you can choose appropriate model version
api_key=gemini_key,
temperature=0.3
)
# Define Agent 1: Destination Planner Agent
destination_agent = Agent(
role="Destination Planner Agent",
goal="Help the user with planning an internationl trip along with brief details about the destination.",
backstory="""You are a destination planning assistant in a travel support team.
You know global destinations, travel requirements, cultural highlights, and can help
users plan international trips with brief, helpful insights.""",
llm=llm,
#allow_delegation=True
)
budget_agent = Agent(
role="Trip Budget Analyst",
goal="""Estimate the total budget for an international trip based on number of days and provide
a brief breakdown of costs for general activities like accommodation, food, transport, and sightseeing.""",
backstory="""You are the budget analyst agent in a travel planning team. Based on trip duration,
you calculate estimated costs for key travel categories such as lodging, meals, local transport,
and attractions. You provide a clear budget summary to help users plan effectively.""",
llm=llm,
#allow_delegation=True
)

activity_agent = Agent(
role="Trip Activity Planner",
goal="""Create a day-by-day itinerary for an international trip based on the number of days,
including suggested activities, local highlights, and pacing recommendations.""",
backstory="""You are the activity planner agent in a travel support team. Based on the trip duration,
you design a daily layout of activities including sightseeing, cultural experiences, relaxation time,
and local recommendations. You help users visualize their trip and make the most of each day.""",
llm=llm,
#allow_delegation=True
)
task_destination = Task(
name="Handle Destination Query",
description="""Customer asks: {customer_message}. Agent: Destination Planner Tool should respond with brief details about the destination or ask for clarification to assist with trip planning.""",
agent=destination_agent,
expected_output="""Concise destination overview or follow-up question to guide trip planning."""
)
task_budget = Task(
name="Estimate Trip Budget",
description="""Customer asks: {customer_message}. Agent: Trip Budget Analyst should estimate the budget based on trip duration and provide a breakdown of typical travel costs.""",
agent=budget_agent,
expected_output="""Estimated total budget with category-wise breakdown or request for missing trip details."""
)
task_activity = Task(
name="Plan Trip Itinerary",
description="""Customer asks: {customer_message}. Agent: Trip Activity Planner should create a day-by-day layout of activities based on the number of trip days and destination type.""",
agent=activity_agent,
expected_output="""Daily itinerary with suggested activities or request for destination and trip duration."""
)
# Create the Crew
crew = Crew(
agents=[destination_agent, budget_agent, activity_agent],
tasks=[task_destination, task_budget, task_activity],
process="sequential", # run tasks in sequence
llm=llm,
verbose=True
)import time
# Simulate a user interaction sequence
inputs = {"customer_message": "I’m thinking about visiting Japan. Can you tell me a bit about what to expect there?"
}
print("=== Step 1: Trip Query ===")
res1 = crew.kickoff(inputs=inputs)
print(res1.raw)
time.sleep(5) # wait 5s between requests
