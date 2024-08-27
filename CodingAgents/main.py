

from custom_code_interpreter_tool import CustomCodeInterpreterTool

from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import warnings

# Warning control
warnings.filterwarnings('ignore')

# Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    api_key='xxx'
)

# Create the custom code interpreter tool with Docker image and container names
custom_code_interpreter_tool = CustomCodeInterpreterTool(
    image_name="custom-code-interpreter-ey1:latest",  # Docker image name
    container_name="custom-code-interpreter-ey1"  # Docker container name
)

# Create an agent with code execution enabled
coding_agent = Agent(
    llm=llm,
    role="Python Data Analyst",
    goal="Analyze data and provide insights using Python",
    backstory="You are an experienced data analyst with strong Python skills.",
    allow_code_execution=True,
    tools=[custom_code_interpreter_tool],  # Add the custom tool to the agent
    verbose=True
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Import the {dataset_name} dataset from the {library_name} package and {operation}.",
    expected_output="The {operation} was performed on the {dataset_name} dataset, and the result is {result_description}.",
    agent=coding_agent
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task],
    verbose=True
)

# Execute the crew
result = analysis_crew.kickoff(inputs={
    "dataset_name": "titanic",  # ... dataset
    "library_name": "pandas",  # Using the ... library
    "operation": "count the number of people older than 60",  # The operation to perform
    "result_description": "The code used was given in an orderly manner and the total number of people older than 60"  # Description of the expected result
})

print(result)