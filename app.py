import streamlit as st
import hydralit_components as hc
import os
import json
from dotenv import load_dotenv, find_dotenv
import tempfile

from src.functions import (
    DataExtractor,
    OpenAIClient,
    Summarizer,
    MarketingResearchAssistant_v1,
    MarketingResearchAssistant_v2,
    DocumentProcessor,
)

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize your classes
openai_client = OpenAIClient()
summarizer = Summarizer(openai_client)
assistant_v1 = MarketingResearchAssistant_v1()
doc_processor = DocumentProcessor()
query_processor = MarketingResearchAssistant_v2()

# Style
css = """
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2D3E50;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        color: #4A6F8A;
    }
    .description {
        font-size: 18px;
        color: #6C8798;
    }
</style>
"""

# Page Configuration
st.set_page_config(
    page_title="Modjo Conversation Analysis Tool",
    page_icon=":robot:", layout="wide"
)
st.markdown(css, unsafe_allow_html=True)

over_theme = {
    "bgcolor": "rgba(0,0,0,0)",
    "title_color": "#FFFFFF",
    "content_color": "#FFFFFF",
    "icon_color": "#FFFFFF",
    "icon_size": 26,
    "option_border": "2px solid #FF9800",
    "container_border": "2px solid #2196F3",
}


def main():
    st.markdown(
        """<h1 style='text-align: center;
                background-color: #2D3E50;
                color: #FFFFFF'>Modjo Conversation Analysis Tool</h1>""",
        unsafe_allow_html=True,
    )

    menu_data = [
        {"icon": "ℹ️", "label": "About"}
    ]

    menu_id = hc.nav_bar(
        menu_definition=menu_data,
        override_theme=over_theme,
        home_name="Home",
        hide_streamlit_markers=True,
        sticky_nav=True,
        sticky_mode="pinned",
    )

    if menu_id == "About":
        about()
    else:
        home()


def home():
    st.markdown(
        "<div class='title'>Welcome to Modjo Conversation Analysis Tool</div>",
        unsafe_allow_html=True,
    )
    option_data = [
            {"icon": "📝", "label": "Summarization"},
            {"icon": "❓", "label": "Question Answering"},
    ]
    font_fmt = {"font-class": "h2", "font-size": "100%"}

    conversation_option = hc.option_bar(
            option_definition=option_data,
            title="Que vous voulez faire ?",
            key="PrimaryOption_",
            override_theme=over_theme,
            font_styling=font_fmt,
            horizontal_orientation=True,
        )
    upload_file()
    if conversation_option == "Summarization":
        summarization()
    elif conversation_option == "Question Answering":
        question_answering()


def about():
    st.markdown(
        """<div class='title'>À propos de Modjo Conversation
                Analysis Tool</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='description'>
    Modjo Conversation Analysis Tool est une application puissante conçue pour
    analyser et extraire des informations précieuses des conversations
    d'affaires.
    En utilisant des technologies avancées d'intelligence artificielle, notre
    outil offre des fonctionnalités uniques pour améliorer votre compréhension
    et votre prise de décision.

    Principales caractéristiques :

    1. <b>Extraction et transformation des données :</b> Notre outil peut
    extraire efficacement des données à partir de fichiers JSON et les
    transformer en un format facilement analysable.

    2. <b>Résumés intelligents :</b> Générez des résumés concis de 60 mots,
    des résumés détaillés au format libre, ou des résumés structurés pour
    obtenir rapidement les points clés de chaque conversation.

    3. <b>Analyse basée sur l'IA :</b> Utilisez notre assistant de recherche
    alimenté par l'IA pour poser des questions spécifiques sur le contenu
    de la conversation et obtenir des réponses précises.

    4. <b>Traitement avancé des documents :</b> Notre processeur de documents
    peut charger, diviser et vectoriser les conversations pour une recherche
    et une analyse plus efficaces.

    5. <b>Interface utilisateur intuitive :</b> Grâce à l'intégration de
    Streamlit et Hydralit, nous offrons une expérience utilisateur fluide
    et agréable.

    Que vous soyez un professionnel des ventes, un responsable marketing ou
    un analyste d'entreprise, Modjo Conversation Analysis Tool vous aide à
    tirer le meilleur parti de vos interactions client, à identifier les
    tendances importantes et à prendre des décisions éclairées basées sur
    des données concrètes.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='subtitle'>Technologie</div>",
                unsafe_allow_html=True)
    st.markdown(
        """
    <div class='description'>
    Notre application utilise des technologies de pointe en matière
    d'intelligence artificielle et de traitement du langage naturel,
    notamment :

    - OpenAI API pour la génération de texte et l'analyse de contenu
    - LangChain pour le traitement avancé des documents et la création
    de chaînes de requêtes
    - Streamlit et Hydralit pour une interface utilisateur web réactive
    et moderne

    Toutes ces technologies sont soigneusement intégrées pour offrir une
    expérience utilisateur fluide et des résultats précis.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='subtitle'>Confidentialité et sécurité</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='description'>
    Chez Modjo, nous prenons la confidentialité et la sécurité de vos données
    très au sérieux. Toutes les conversations analysées sont traitées de
    manière sécurisée et ne sont pas stockées après l'analyse. Nous utilisons
    des protocoles de cryptage standard de l'industrie pour protéger vos
    informations pendant la transmission et le traitement.
    </div>
    """,
        unsafe_allow_html=True,
    )


def upload_file():
    st.markdown(
        "<div class='subtitle'>Upload Conversation File</div>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Choose a conversation file", type="json")

    if uploaded_file is not None:
        json_data = json.load(uploaded_file)
        dataextractor = DataExtractor(json_data)
        conversation = dataextractor.data_frame_to_text()
        with st.expander("Conversation Content"):
            st.text_area("", conversation, height=200)
        st.session_state["conversation"] = conversation
        st.success("File uploaded successfully!")


def summarization():
    st.markdown("<div class='subtitle'>Summarization</div>",
                unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.warning("Please upload a file first.")
        return

    if st.button("Generate Summaries"):
        with st.spinner("Generating summaries..."):
            summary_60 = summarizer.summarize_60_words(
                st.session_state["conversation"]
            )
            summary_free = summarizer.summarize_free_format(
                st.session_state["conversation"]
            )
            summary_structured = summarizer.summarize_structured(
                st.session_state["conversation"]
            )

        tab1, tab2, tab3 = st.tabs(
            ["60-word Summary", "Free-format Summary", "Structured Summary"]
        )
        with tab1:
            st.write(summary_60.replace("`", ""))
        with tab2:
            st.write(summary_free)
        with tab3:
            st.write(summary_structured)


def question_answering():
    st.markdown(
        "<div class='subtitle'>Question Answering</div>",
        unsafe_allow_html=True
    )
    if "conversation" not in st.session_state:
        st.warning("Please upload a file first.")
        return

    question = st.text_input("Enter your question about the conversation")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Answer with v1"):
            with st.spinner("Generating answer..."):
                answer_v1 = assistant_v1.rag(
                    question,
                    st.session_state["conversation"]
                )
            st.write(answer_v1)

    with col2:
        if st.button("Answer with v2"):
            with st.spinner("Processing documents..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, mode="w+", suffix=".txt", encoding="utf-8"
                ) as temp_file:
                    temp_file.write(st.session_state["conversation"])
                    temp_file_path = temp_file.name
                texts = doc_processor.load_and_split_documents(temp_file_path)
                vectordb = doc_processor.create_vector_store(texts)
                chain = query_processor.create_retrieval_chain(vectordb)
                os.unlink(temp_file_path)

            with st.spinner("Generating answer..."):
                answer_v2 = query_processor.process_query(chain, question)
            st.write(answer_v2)


def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2C3E50;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        </style>
        <div class="footer">
            © 2024 Modjo Conversation Analysis Tool
            | Développé par Bourahima
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
    footer()
