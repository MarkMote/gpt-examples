import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from keys import get_openAI_key

os.environ["OPENAI_API_KEY"] = get_openAI_key()

template = """Answer the question with a single number coresponding to the aisle number that best matches the question, according to the context below.

    Context: 
    here's an example list of what might be contained in each aisle of a hardware store with only 5 aisles:
    aisle 1. Lumber, Tools and hardware - hammers, wrenches, screwdrivers, nails etc.
    aisel 2. Plumbing and electrical - pipes, fittings, faucets, valves, etc.
    aisle 3. Paint and building materials - paint brushes, rollers, trays, etc.
    aisle 4: Garden and outdoor - lawn mowers, trimmers, hoses, shovels etc.
    aisle 5: Safety and automotive - locks, alarms, smoke detectors, etc.
    
    Here is an example: 
    Question: Where can I find a hammer?
    Answer: 1
    
    Question: {question}

    Answer:
    """


prompt = PromptTemplate(template=template, input_variables=["question"])

print(
    """\nContext: 
    aisle 1. Lumber, Tools and hardware - hammers, wrenches, screwdrivers, nails etc.
    aisel 2. Plumbing and electrical - pipes, fittings, faucets, valves, etc.
    aisle 3. Paint and building materials - paint brushes, rollers, trays, etc.
    aisle 4: Garden and outdoor - lawn mowers, trimmers, hoses, shovels etc.
    aisle 5: Safety and automotive - locks, alarms, smoke detectors, etc."""
)

# user question
print("\nTell me what your looking for and I'll tell you which aisle to go to!\n")

davinci = OpenAI(model_name="text-davinci-003")

llm_chain = LLMChain(prompt=prompt, llm=davinci)

while True:
    print("How can I help you?")
    user_input = input("> ")
    if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
        print("Goodbye!")
        break
    print("Go to aisle: ", llm_chain.run(user_input), "\n")
