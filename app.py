# import openai
import os
import sys
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
# Set OpenAI API key
llm = OpenAI(openai_api_key="OPENAI_API_KEY")
# Set OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# query = sys.argv[1]
# print(query)


loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
#print(index.query(query,llm=ChatOpenAI()))

# prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
# current_prompt = prompt.format(product="colorful computers")

# llm = OpenAI(temperature=0.9)
# # ans = llm.predict(current_prompt)
# # print(ans)
# print(current_prompt)






# audio_file = open("1.wav", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)

transcript1 = "Not right now. Can I ask you what is the address of your emergency? Hi, my name's Eleanor, I'm here in Bedlands near the Perth Worth Cemetery and Karrakatta Cemetery."
# transcript2 = "A fire's just started up just outside in Hollywood Reserve near Bill Day Walk. OK, what's the nearest street to it,"
# transcript3 = "like what's the last street we're going to have to come down to get to it? Corolla Street we've got, you can see the street here, it's actually, I don't know if someone's started the fire. What's actually, what is on fire?"
# transcript4 = "Trees, shrubs, wood falling out, lots of glaze. OK, no worries, we're going to crew out to have a look, we'll be there shortly. Thanks for your call. Yeah, no problem. Thank you."

prompt = PromptTemplate.from_template("Here is transcript from a emergency call generator by whisper:''' {transcript} ''' please answer the question: The location of the emergency?")
current_prompt = prompt.format(transcript=transcript1)

print(index.query(current_prompt,llm=ChatOpenAI()))

# print(transcript)
