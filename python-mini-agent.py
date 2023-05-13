import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from keys import get_openAI_key

os.environ["OPENAI_API_KEY"] = get_openAI_key()

template = """Generate a Python script that will solve the task answered in the question. The code should be self contained and sufficient for completing the task.

    Context: 
    None     

    Examples:
    None 
    
    Question: {question}
    Answer:
    """

loop_template_string = """Use the previous code and error message below to help correct the python script to get correct code. The code should be self contained python script that is sufficient for completing the task.

    Context:
    What the code should do: {objective}

    Previous code that's been attempted: {code}

    Exception (error message): {error}

    Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
loop_prompt = PromptTemplate(
    template=loop_template_string, input_variables=["objective", "code", "error"]
)

davinci = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=2500)

llm_chain_0 = LLMChain(prompt=prompt, llm=davinci)
llm_chain_1 = LLMChain(prompt=loop_prompt, llm=davinci)


def print_to_file(text):
    with open("result.py", "w") as f:
        f.write(text)


# Main
max_evals = 4
evals = 0
error_message = ""
while True:
    if evals > max_evals:
        print("Max evals reached. Exiting...")
        break
    if evals == 0:
        print("Ask me what to code in python!")
        user_input = input("> ")
        llm_result = llm_chain_0.run(user_input)
        result = llm_result
        print("\nANSWER: ", result, "\n")

    else:
        print("\n\nTrying again...\n\n")
        llm_result = llm_chain_1.run(
            {"objective": user_input, "code": result, "error": error_message}
        )
        result = llm_result
        print("\n\n------------------------------------------------------------\n\n")
        print("\nNew ANSWER: ", result, "\n")
    try:
        print("going to try to run the code...")
        exec(result)
        print("ran the code sucessfully!")
        print("COMPLETE: the results have been printed to the file 'result.py'")
        print_to_file(result)
        break
    except Exception as e:
        error_message = e
        print("EXCEPTION: ", e)

    evals += 1
