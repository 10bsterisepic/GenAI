!pip install crewai openai
from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_KEY"] = "AIzaSyCd2pQIJoIZn_lrJcAdFcQMcO9G3v9VLuo"
os.environ["OPENAI_MODEL_NAME"] = "gemini-flash-latest"  # lightweight model for demo

general_summarizer = Agent(
    role="General Summarizer",
    goal="Summarize text concisely and clearly.",
    backstory=(
        "You are an expert summarizer who reads large documents "
        "and condenses them into short, meaningful summaries."
    )
)

keyword_extractor = Agent(
    role="Keyword Extractor",
    goal="Extract key terms and important phrases from text.",
    backstory=(
        "You specialize in identifying the most relevant keywords "
        "that represent the main ideas of each section."
    )
)

conceptual_abstractor = Agent(
    role="Conceptual Abstractor",
    goal="Create high-level conceptual summaries emphasizing main ideas.",
    backstory=(
        "You focus on abstracting the conceptual meaning and insights "
        "from each text section rather than detailed facts."
    )
)

coordinator_agent = Agent(
    role="Coordinator Agent",
    goal="Combine outputs from all summarizer agents into one unified summary.",
    backstory=(
        "You are a coordinator that merges multiple summaries into "
        "a coherent, logically flowing final version."
    )
)
# Step 4: Define the Input Document
# -------------------------------------------------------------
document = """
Artificial Intelligence (AI) has transformed industries through automation,
decision-making, and data analysis. From healthcare to finance, AI systems
enhance efficiency and accuracy.

Machine Learning (ML), a branch of AI, focuses on creating algorithms that
learn from data without explicit programming. It includes supervised and
unsupervised learning approaches.

Natural Language Processing (NLP) enables computers to understand and respond
to human language. Applications include chatbots, translation, and sentiment
analysis.

Deep Learning, part of ML, uses multi-layered neural networks to process complex
data such as images, sound, and video with remarkable precision.
"""

sections = document.strip().split("\n\n")  # divide into sections

# -------------------------------------------------------------
# Step 5: Define Tasks for Each Agent
# -------------------------------------------------------------
tasks = [
    Task(
        description=f"Summarize Section 1: {sections[0]}",
        agent=general_summarizer,
        expected_output="A concise and clear summary of the provided text section."
    ),
    Task(
        description=f"Extract keywords from Section 2: {sections[1]}",
        agent=keyword_extractor,
        expected_output="A list of key terms and important phrases extracted from the text section."
    ),
    Task(
        description=f"Generate conceptual summary from Section 3: {sections[2]}",
        agent=conceptual_abstractor,
        expected_output="A high-level conceptual summary emphasizing the main ideas of the text section."
    ),
    Task(
        description=f"Combine all previous outputs into a coherent summary.",
        agent=coordinator_agent,
        expected_output="A unified final summary combining all agent insights."
    )
]
# Step 6: Create and Run the Crew
# -------------------------------------------------------------
crew = Crew(
    agents=[general_summarizer, keyword_extractor, conceptual_abstractor, coordinator_agent],
    tasks=tasks,
    process=Process.sequential,  # tasks run in order
)

print("🚀 Starting Multi-Agent Summarization Crew...\n")
results = crew.kickoff()
# Step 7: Display the Results
# -------------------------------------------------------------
print("\n===== FINAL UNIFIED SUMMARY ====")
print(results)
print("\n===== FINAL UNIFIED SUMMARY ====")
print(results)  # The results variable already holds the final coordinator output
