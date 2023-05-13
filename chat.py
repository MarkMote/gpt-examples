import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import time
from keys import get_openAI_key


os.environ["OPENAI_API_KEY"] = get_openAI_key()

f_text = True
f_speak = False

template = """Answer the question in rhyme. Be concise and to the point. 

    Context: 
    None

    Examples:
    None
    
    Question: {question}
    Answer:
    """

prompt = PromptTemplate(template=template, input_variables=["question"])

davinci = OpenAI(model_name="text-davinci-003", temperature=0)

llm_chain = LLMChain(prompt=prompt, llm=davinci)

### TEXT to SPEECH ###
language = "en"


def speak(text, slow=False):
    if f_speak:
        myobj = gTTS(text=text, lang=language, slow=slow)
        myobj.save("welcome.mp3")
        playsound("welcome.mp3")


### SPEECH 2 TEXT ###
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        heard_correctly = False
        while not heard_correctly:
            audio = r.listen(source, timeout=3)
            try:
                # recognize speech using Google Speech Recognition
                text = r.recognize_google(audio)
                print(f"You said: {text}")
                heard_correctly = True
                if text.lower() in [
                    "goodbye",
                    "stop",
                    "quit",
                    "exit",
                    "that's all",
                    "nothing else",
                    "thank you",
                ]:
                    print("Goodbye!")
                    speak(text="Thank you, Goodbye!")
                    return "STOP"
            except sr.UnknownValueError:
                print("Sorry, I didn't get that.")
                speak(text="Sorry, I didn't get that.")
                heard_correctly = True  # HACK
                return None

    return text


### Main ###
while True:
    print("How can I help you?")
    speak(text="How can I help you?")
    if f_text:
        user_input = input("> ")
    else:
        user_input = listen()
    if user_input is None:
        continue
    if user_input == "STOP":
        break
    llm_result = llm_chain.run(user_input)
    print("llm_result: ", llm_result)
    result = llm_result

    print(result, "\n")
    speak(text=result, slow=False)
    time.sleep(1)
