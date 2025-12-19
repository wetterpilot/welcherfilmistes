# app.py (bereinigt)

import os
import traceback
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env (only in development)
load_dotenv()

# Read GROQ API key from environment (recommended)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY fehlt. Bitte in .env oder in Streamlit-Secrets setzen.")
    st.stop()

# Ensure libraries that read env vars can access it
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Output parser / Pydantic models
class MyMovieOutput(BaseModel):
    title: str = Field(description="The movie title")
    director: str = Field(
        description="The director of the movie in the form 'surname, firstname'",
        examples=["Cameron, James"],
    )
    actors: list[str] = Field(description="The 5 most relevant actors of the movie")
    release_year: int

class MyMovieOutputs(BaseModel):
    movies: list[MyMovieOutput]

output_parser = PydanticOutputParser(pydantic_object=MyMovieOutputs)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Du lieferst Filminformationen auf Basis der übergebenen Handlung.
        Gib die 5 relevantesten Filme an.
        {format_instructions}
    """),
    ("user", "Handlung: <<{handlung}>>")
]).partial(format_instructions=output_parser.get_format_instructions())

# Model configuration
MODEL_NAME = "llama-3.3-70b-versatile"
# If ChatGroq accepts an api_key parameter, you can pass it explicitly:
# model = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY)
model = ChatGroq(model=MODEL_NAME)

# Chain erstellen
chain = prompt_template | model | output_parser

# Streamlit UI
st.title("Welcher Film ist es?")

filmbeschreibung = st.chat_input("Handlung eingeben")
if filmbeschreibung:
    with st.chat_message("user"):
        st.write(f"Nutzerbeschreibung: {filmbeschreibung}")

    try:
        res = chain.invoke({"handlung": filmbeschreibung})
        movies = res.model_dump().get("movies", [])
        with st.chat_message("ai"):
            if not movies:
                st.write("Keine Filme gefunden.")
            for r in movies:
                st.markdown(f"**Titel:** {r.get('title', '')}")
                st.markdown(f"**Regisseur:** {r.get('director', '')}")
                actors = r.get('actors', [])
                if actors:
                    st.markdown(f"**Schauspieler:** {', '.join(actors)}")
                st.markdown(f"**Jahr:** {r.get('release_year', '')}")
                st.divider()
    except Exception:
        st.error("Fehler beim Abfragen des Modells. Siehe Traceback unten.")
        st.text(traceback.format_exc())
