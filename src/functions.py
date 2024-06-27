from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

import pandas as pd
import json

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class DataExtractor:
    """
    A class to extract and transform data from a JSON object into a structured DataFrame and text format.

    Attributes:
    json_data : json, optional
        The input JSON data (default is None).
    data : pd.DataFrame
        The DataFrame to store the transformed data.
    res : list or None
        The extracted data segments from the JSON.

    Methods:
    extract_data() -> None:
        Extracts data segments from the JSON data.

    transform_data() -> pd.DataFrame:
        Transforms the extracted data into a DataFrame, combining consecutive entries by the same speaker.

    data_frame_to_text() -> str:
        Converts the DataFrame into a formatted text string.
    """

    def __init__(self, json_data: json):
        """
        Initializes the DataExtractor with optional JSON data.

        Parameters:
            json_data : json
                The input JSON data (default is None).
        """
        self.json_data = json_data
        self.data = pd.DataFrame()
        self.res = None

    def extract_data(self) -> None:
        """
        Extracts data segments from the JSON data.

        Raises:
            KeyError:
                If the "segments" key is not found in the JSON data.
        """
        if self.json_data is None:
            print("No JSON data provided.")
            return
        try:
            self.res = self.json_data["segments"]
        except KeyError as e:
            print(f"Error extracting data: {e}")
            self.res = []

    def transform_data(self) -> pd.DataFrame:
        """
        Transforms the extracted data into a DataFrame, combining consecutive entries by the same speaker.

        Returns:
            pd.DataFrame
                A DataFrame with combined content for consecutive entries by the same speaker.

        Raises:
            None
        """
        if self.res is None:
            self.extract_data()

        if not self.res:
            print("No data to transform.")
            return pd.DataFrame()

        data = [
            {"speaker": item.get("speaker"), "content": item.get("content")}
            for item in self.res
        ]
        data = pd.DataFrame(data)

        indexes_to_drop = []

        for ligne in range(data.shape[0] - 1):
            if data.iloc[ligne, 0] == data.iloc[ligne + 1, 0]:
                data.at[ligne, "content"] = (
                    data.at[ligne, "content"] + " " + data.at[ligne + 1, "content"]
                )
                indexes_to_drop.append(ligne + 1)

        data = data.drop(indexes_to_drop)
        self.data = data.reset_index(drop=True)
        return self.data

    def data_frame_to_text(self) -> str:
        """
        Converts the DataFrame into a formatted text string.

        Returns:
            str
                A string representation of the DataFrame with each row formatted as "speaker: content".

        Raises:
            None
        """
        if self.data.empty:
            self.transform_data()

        if self.data.empty:
            return "No data available."

        return "\n".join(
            f"{row.speaker}: {row.content}" for _, row in self.data.iterrows()
        )



class OpenAIClient:
    """
    A client for interacting with the OpenAI API.

    Attributes:
        system_message (dict): The system message to be used in the chat prompt.
        client (OpenAI): The OpenAI client instance.

    Methods:
        __init__(api_key): Initializes the OpenAI client with the given API key.
        get_completion(prompt, model): Sends a prompt to the OpenAI API and returns the completion.
        get_system_message(system_message): Sets the system message for the client.
    """

    def __init__(self, api_key=OPENAI_API_KEY):
        """
        Initializes the OpenAI client.

        Parameters:
            api_key (str): The API key for the OpenAI API.
        """
        self.system_message = None
        if api_key is None:
            raise ValueError(
                "OpenAI API key is missing. Please set it in the .env file."
            )
        else:
            self.client = OpenAI(api_key=api_key)

    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        """
        Sends a prompt to the OpenAI API and returns the completion.

        Parameters:
            prompt (str): The prompt to be sent to the OpenAI API.
            model (str): The model to be used for generating the completion. Default is "gpt-3.5-turbo".

        Returns:
            str: The content of the completion response.
        """
        try:
            messages = [self.system_message, {"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred when calling the OpenAI API: {e}")
            return None

    def get_system_message(self, system_message):
        """
        Sets the system message for the client.

        Parameters:
            system_message (dict): The system message to be used in the chat prompt.
        """
        self.system_message = system_message


class Summarizer:
    """
    A summarizer for generating conversation summaries using the OpenAI API.

    Attributes:
        openai_client (OpenAIClient): An instance of the OpenAIClient.

    Methods:
        __init__(openai_client): Initializes the summarizer with the given OpenAI client.
        summarize_60_words(texte): Generates a 60-word summary of the given conversation.
        summarize_free_format(texte): Generates a comprehensive summary of the given conversation.
        summarize_structured(texte): Generates a structured summary of the given conversation.
    """

    def __init__(self, openai_client):
        """
        Initializes the summarizer with the given OpenAI client.

        Parameters:
            openai_client (OpenAIClient): An instance of the OpenAIClient.
        """
        self.openai_client = openai_client
        self.openai_client.get_system_message(
            {
                "role": "system",
                "content": (
                    "You are a helpful expert on conversation summarization wizards."
                ),
            }
        )

    def summarize_60_words(self, texte):
        """
        Generates a 60-word summary of the given conversation.

        Parameters:
            texte (str): The conversation text to be summarized.

        Returns:
            str: The 60-word summary of the conversation.
        """
        prompt = f"""
        You are an expert in conversation summarization. Your task is to create a concise and informative summary of the following conversation, delimited by triple backticks.

        Guidelines:
        1. Identify the key points, persons and main themes of the conversation.
        2. Capture the essence of the exchange, including important opinions or decisions made.
        3. Use clear and precise language.
        4. Avoid superfluous details and focus on the essential.
        5. Strictly adhere to the 60-word limit.
        6. Ensure the summary is coherent and self-contained.

        Conversation: ```{texte}```

        Summary (max 60 words):
        """
        return self.openai_client.get_completion(prompt)

    def summarize_free_format(self, texte):
        """
        Generates a comprehensive summary of the given conversation.

        Parameters:
            texte (str): The conversation text to be summarized.

        Returns:
            str: The comprehensive summary of the conversation.
        """
        prompt = f"""
        You are a highly skilled sales conversation analyst. Your task is to create a comprehensive and actionable summary of the following conversation, delimited by triple backticks.

        Objective: 
        Generate a detailed summary that will be invaluable for sales managers, coaches, and team members involved in the deal.

        Guidelines:
        1. Identify and highlight the key points, persons, outcomes, and any critical decisions made during the call.
        2. Outline the main topics discussed and the flow of the conversation.
        3. Capture important customer information, including needs, pain points, and objections.
        4. Note any commitments made by either party or next steps agreed upon.
        5. Highlight potential areas for improvement or coaching opportunities for the sales representative.
        6. Include relevant context about the deal's status, size, or importance if mentioned.
        7. Use clear, professional language and organize the summary in a logical structure.
        8. Provide insights that could be useful for deal strategy or future interactions with the customer.

        Your summary should be thorough enough to give a clear understanding of the call to someone who wasn't present, yet concise enough to be quickly digestible by busy professionals.

        Conversation: ```{texte}```

        Detailed Summary:
        """
        return self.openai_client.get_completion(prompt)

    def summarize_structured(self, texte):
        """
        Generates a structured summary of the given conversation.

        Parameters:
            texte (str): The conversation text to be summarized.

        Returns:
            str: The structured summary of the conversation.
        """
        prompt = f"""
        As an expert in business communication analysis, create a comprehensive and structured summary of the following conversation. Your summary should be valuable for team members who did not participate in the call, providing them with clear insights and actionable information.

        Structure your summary as follows:

        1. Purpose of the Call:
        - Clearly state the main objective(s) of the conversation.
        - Include any context or background information that sets the stage for the call.

        2. Key Points Discussed:
        - Provide a bulleted list of the main topics covered.
        - For each point, include brief but essential details.
        - Highlight any significant insights, challenges, or opportunities mentioned.

        3. Results and Next Steps:
        - Summarize the outcomes of the conversation.
        - List any decisions made or conclusions reached.
        - Outline the agreed-upon next steps or future plans.

        4. Action Points:
        - Create a clear, bulleted list of specific tasks or actions to be taken.
        - For each action item, specify who is responsible (if mentioned) and any deadlines.
        - Include any follow-up meetings or communications planned.

        5. Additional Insights (optional):
        - Note any underlying issues, potential risks, or opportunities not explicitly discussed but implied.
        - Provide any strategic recommendations based on the conversation content.

        Guidelines:
        - Use clear, concise language.
        - Focus on factual information and avoid personal interpretations unless specifically relevant.
        - Ensure the summary is self-contained and understandable without additional context.
        - Aim for a balance between comprehensiveness and brevity.

        Conversation: ```{texte}```

        Detailed Summary:
        """
        return self.openai_client.get_completion(prompt)


class MarketingResearchAssistant_v1:
    """
    A virtual assistant for answering questions based on a given conversation using the OpenAI API.

    Attributes:
        openai_client (OpenAI): The OpenAI client instance.
        model (str): The model used for generating responses.
        system_message (dict): The system message containing the guidelines for generating responses.

    Methods:
        __init__(model): Initializes the assistant with the specified model.
        rag(query, conversation): Generates a response to the query based on the given conversation.
        set_model(model): Sets the model for the assistant.
        get_model(): Returns the current model of the assistant.
    """

    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the marketing research assistant with the specified model.

        Parameters:
            model (str): The model used for generating responses. Default is "gpt-3.5-turbo".
        """
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.system_message = {
            "role": "system",
            "content": (
                "You are an expert AI assistant tasked with answering questions using only the information provided in the given conversation. Follow these guidelines: "
                "1. Carefully analyze the question and the provided conversation."
                "2. Answer solely based on the information in the conversation. Do not make assumptions or add outside information."
                "3. If the answer is not in the conversation, clearly state 'I cannot answer this question based on the provided conversation.' and said why"
                "4. Cite relevant parts of the conversation to support your answer."
                "5. If the conversation contains contradictory information, mention it and explain the different perspectives."
                "6. Structure your response logically:"
                "a) Start with a direct sentence answering the question."
                "b) Provide additional details or explanations if necessary."
                "c) Conclude by summarizing the key points."
                "7. Limit your response to three sentences maximum, unless more details are absolutely necessary for a complete and accurate answer."
                "8. If you are uncertain about any element of your answer, clearly indicate your level of certainty."
            ),
        }

    def rag(self, query, conversation):
        """
        Generates a response to the query based on the given conversation.

        Parameters:
            query (str): The question to be answered.
            conversation (str): The conversation text to be analyzed.

        Returns:
            str: The generated response from the OpenAI API.
        """
        messages = [
            self.system_message,
            {
                "role": "user",
                "content": f"Question: {query}\nConversation: {conversation}",
            },
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while processing the query: {e}")
            return None

    def set_model(self, model):
        """
        Sets the model for the assistant.

        Parameters:
            model (str): The model to be set.
        """
        self.model = model

    def get_model(self):
        """
        Returns the current model of the assistant.

        Returns:
            str: The current model used by the assistant.
        """
        return self.model


class DocumentProcessor:
    """
    A class to process documents by loading, splitting, and creating a vector store.

    Attributes:
        api_key (str): The API key for OpenAI.
        persist_directory (str): The directory to persist the vector store.
        embedding (OpenAIEmbeddings): The embeddings instance from OpenAI.

    Methods:
        load_and_split_documents(file_path): Loads and splits documents from the given file path.
        create_vector_store(texts): Creates and persists a vector store from the given texts.
    """

    def __init__(self):
        """
        Initializes the DocumentProcessor with the embedding.
        """
        self.persist_directory = "data"
        self.embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    def load_and_split_documents(self, file_path):
        """
        Loads and splits documents from the given file path.

        Parameters:
            file_path (str): The path to the document file.

        Returns:
            list: A list of split document chunks.
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n"]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, texts):
        """
        Creates and persists a vector store from the given texts.

        Parameters:
            texts (list): A list of text documents.

        Returns:
            Chroma: The created and persisted vector store.
        """
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
        )
        vectordb.persist()
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )


class MarketingResearchAssistant_v2:
    """
    A virtual assistant for answering questions based on a given context using a retrieval chain.

    Attributes:
        llm (ChatOpenAI): The language model used for generating responses.

    Methods:
        create_retrieval_chain(vectordb): Creates a retrieval chain using the given vector store.
        process_query(chain, query): Processes the query using the given chain and returns the response.
    """

    def __init__(self):
        """
        Initializes the marketing research assistant with a ChatOpenAI instance.
        """
        self.llm = ChatOpenAI(temperature=0.0, api_key=OPENAI_API_KEY)

    def create_retrieval_chain(self, vectordb):
        """
        Creates a retrieval chain using the given vector store.

        Parameters:
            vectordb (Chroma): The vector store to be used for retrieval.

        Returns:
            RetrievalChain: The created retrieval chain.
        """
        retriever = vectordb.as_retriever()
        system_prompt = (
            "You are an expert AI assistant tasked with answering questions using only the information provided in the given context. Follow these guidelines: "
            "1. Carefully analyze the question and the provided context."
            "2. Answer solely based on the information in the context. Do not make assumptions or add outside information."
            "3. If the answer is not in the context, clearly state 'I cannot answer this question based on the provided context.' and said why"
            "4. Cite relevant parts of the context to support your answer."
            "5. If the context contains contradictory information, mention it and explain the different perspectives."
            "6. Structure your response logically:"
            "a) Start with a direct sentence answering the question."
            "b) Provide additional details or explanations if necessary."
            "c) Conclude by summarizing the key points."
            "7. Limit your response to three sentences maximum, unless more details are absolutely necessary for a complete and accurate answer."
            "8. If you are uncertain about any element of your answer, clearly indicate your level of certainty."
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    def process_query(self, chain, query):
        """
        Processes the query using the given chain and returns the response.

        Parameters:
            chain (RetrievalChain): The retrieval chain to be used for processing the query.
            query (str): The query to be processed.

        Returns:
            str: The generated response from the retrieval chain.
        """
        llm_response = chain.invoke({"input": query})
        return llm_response["answer"]
