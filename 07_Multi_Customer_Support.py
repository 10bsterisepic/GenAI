!pip install crewai crewai_tools
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
# Define Agent 1: Product Query Specialist
product_agent = Agent(
role="Product Query Specialist",
goal="""Help the customer with product information, answer queries about
availability, features, pricing.""",
backstory="""You are a product support agent in a customer-service team.
You know the catalog of products, stock levels, pricing, and can help
customers decide.""",
llm=llm,
allow_delegation=True
)
# Define Agent 2: Order Placement Specialist
order_agent = Agent(
role="Order Placement Specialist",
goal="""Assist the customer to place an order, confirm details, collect
shipping info and prepare order summary""",
backstory="""You are the order desk agent. After product selection you
capture customer order details, shipping address, quantity, generate an
order id and hand off to payment.""",
llm=llm,
allow_delegation=True
)
# Define Agent 3: Payment & Processing Specialist
payment_agent = Agent(
role="Payment Processing Specialist",
goal="""Handle payment processing simulation, confirm payment, update
order status and notify customer""",
backstory="""You are the payments agent. You take payment method,
simulate transaction, update order status to 'paid' and notify customer
the order is confirmed.""",
llm=llm,
allow_delegation=True
)
# Define Tasks
task_query = Task(
name="Handle Product Query",
description="""Customer asks: {customer_message}. Agent: Product Query
Specialist should respond with product info or redirect to order
placement.""",
agent=product_agent,
expected_output="""Informative product information or ask for
clarification/customer decision."""
)

task_order = Task(
name="Place Order",
description="""Given product choice and quantity, shipping address and
customer confirmation, the Order Placement Specialist should generate an
order summary, order ID, and pass to payment.""",
agent=order_agent,
expected_output="""Order summary including order_id, total_amount,
shipping_address"""
)
task_payment = Task(
name="Process Payment",
description="""Given the order_id and payment details, the Payment
Processing Specialist should simulate payment, update order status, and
send confirmation to customer.""",
agent=payment_agent,
expected_output="""Payment success message, order status update,
confirmation to customer"""
)
# Create the Crew
crew = Crew(
agents=[product_agent, order_agent, payment_agent],
tasks=[task_query, task_order, task_payment],
process="sequential", # run tasks in sequence
llm=llm,
verbose=True
)
import time
# Simulate a user interaction sequence
inputs = {
"customer_message": "Hi, I am looking for a wireless Bluetooth speaker under 5000 rupees. What do you have?"
}
print("=== Step 1: Product Query ===")
res1 = crew.kickoff(inputs=inputs)
print(res1.raw)
time.sleep(5) # wait 5s between requests
