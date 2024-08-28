

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

# 1. Custom Code Interpreter Tool for Data Analysis Agent
data_analysis_tool = CustomCodeInterpreterTool(
    image_name="code_interpreter_image",  # Docker image for data analysis
    container_name="code_interpreter_container"  # Docker container for data analysis
)

# 2. Custom Code Interpreter Tool for Report Writing Agent
report_generation_tool = CustomCodeInterpreterTool(
    image_name="report_container_image",  # Docker image for report writing
    container_name="report_container"  # Docker container for report writing
)

# Create the data analysis agent
coding_agent = Agent(
    llm=llm,
    role="Python Data Analyst",
    goal="Analyze data and provide insights using Python",
    backstory="You are an experienced data analyst with strong Python skills.",
    allow_code_execution=True,
    tools=[data_analysis_tool],  # Add the data analysis tool
    verbose=True
)

# Create the report generation agent
report_agent = Agent(
    llm=llm,
    role="Report Generator",
    goal="Generate a summary report based on data analysis results",
    backstory="You are responsible for generating reports.",
    allow_code_execution=True,
    tools=[report_generation_tool],  # Add the report generation tool
    verbose=True
)

# Create a task for data analysis
data_analysis_task = Task(
    description="Import the {dataset_name} dataset from the {library_name} package and {operation}.",
    expected_output="The {operation} was performed on the {dataset_name} dataset, and the result is {result_description}.",
    agent=coding_agent
)

# Create a task for report generation
report_task = Task(
    description="Take the output and code from the data analysis task and generate Python code to write a report to a file."
                "The report should include the Python code that was run and the result in a user-friendly language.",
    expected_output="Python code to print the results report of the data analysis task to a file. "
                    "Make sure to print the report in a user-friendly language. "
                    "Create the output file in a folder named: reports "
                    "Name the output file as: report0.txt",
    agent=report_agent
)

# Create a crew and add the tasks
analysis_crew = Crew(
    agents=[coding_agent, report_agent],
    tasks=[data_analysis_task, report_task],
    verbose=True
)

# Execute the crew with inputs for data analysis
result = analysis_crew.kickoff(inputs={
    "dataset_name": "titanic",
    "library_name": "pandas",
    "operation": "count the number of people older than 60",
    "result_description": "The code used was given in an orderly manner and the total number of people older than 60"
})

print(result)