import streamlit as st

# for vector store
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# for chat agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

# weather
from geopy.geocoders import Nominatim
from openmeteopy import OpenMeteo
from openmeteopy.hourly import HourlyHistorical
from openmeteopy.daily import DailyHistorical
from openmeteopy.options import HistoricalOptions
from openmeteopy.utils.constants import *


import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import BotoCoreError, ClientError
from langchain.tools import tool
import os

import pandas as pd
from sqlalchemy import create_engine

# prepare the vector store
def prepare_vector_store():
    loader = TextLoader('./all_text.txt')
    all_texts = loader.load()

    doc_splitter = CharacterTextSplitter(
        separator="<sep>",
        chunk_size=10000,
        chunk_overlap=0,
        length_function=len,
    ) # function to split the whole thing by document

    lines_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=1000,
        length_function=len,
    ) # function to split each doc by chunc

    documents = doc_splitter.split_documents(all_texts)
    texts = []
    for doc in documents:
        texts.extend(lines_splitter.split_documents([doc]))

    # texts = text_splitter.split_documents(documents) # print this out to show the text being split
    print('number of chunks:', len(texts))

    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    model_for_creating_embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))

    # initialize a vector database db by applyign the embedding model to the text chuncks
    db = FAISS.from_documents(texts, model_for_creating_embeddings)
    return db

def prepare_retriever_tool(db):
    # create a retriever from the db
    retriever = db.as_retriever(search_kwargs={'k': 4})

    retriever_tool = create_retriever_tool(
        retriever,
        name="get_Dino_documents",
        description="Get Instructions for Dinos safety from Dino documents. (for T-Rexs and Velociraptors)",
    )
    return retriever_tool

def create_dynamo_table_from_csv(csv_name='data.csv'):
    dynamodb = boto3.resource('dynamodb', 
                            region_name='us-west-1',
                            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )

    TableName='TransportData'
    table = dynamodb.Table(TableName)

    def create_table():
        table = dynamodb.create_table(
            TableName=TableName,
            KeySchema=[
                {'AttributeName': 'Date', 'KeyType': 'HASH'}, 
            ],
            AttributeDefinitions=[
                {'AttributeName': 'Date', 'AttributeType': 'S'},
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }

        )
        table.wait_until_exists()
        return table

    try:
        table = create_table()
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        table = dynamodb.Table(TableName)
        print('Table already exists')    

    import pandas as pd
    csv_data = pd.read_csv(csv_name).to_dict(orient='records')
    with table.batch_writer() as batch:
        for row in csv_data:
            batch.put_item(Item=row)
    return dynamodb


def create_type_to_id_engine(csv_name='id_name.csv'):
    id_name_df = pd.read_csv(csv_name)
    engine = create_engine("sqlite:///id_name.db")
    id_name_df.to_sql("id_name", engine, index=False, if_exists="replace")
    return engine

if __name__ == '__main__':

    os.environ["OPENAI_API_KEY"] = st.text_input("Input your OpenAI API key:", type="password")
    os.environ["AWS_ACCESS_KEY_ID"] = st.text_input("Input your AWS access key:", type="password")
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.text_input("Input your AWS secret access key:", type="password")

    if os.environ["OPENAI_API_KEY"] == "" or os.environ["AWS_ACCESS_KEY_ID"] == "" or os.environ["AWS_SECRET_ACCESS_KEY"] == "":
        # i want to skip the rest here
        st.write("Please input your API keys")
        st.stop()


    if 'Initialization' not in st.session_state:
        st.session_state.Initialization = True 
        # os.environ["OPENAI_API_KEY"] = getpass.getpass("Input your OpenAI API key: ")
        # os.environ["AWS_ACCESS_KEY_ID"] = getpass.getpass("Input your AWS access key:")
        # os.environ["AWS_SECRET_ACCESS_KEY"] = getpass.getpass("Input your AWS secret access key:")
        st.write("initializing")
        st.session_state.previous_temperature = None
        st.session_state.previous_model_choice = None
        st.session_state.previous_current_task = None
        st.session_state.previous_system_message = None

        # databases
        st.session_state.vector_store = prepare_vector_store()
        st.session_state.dynamodb = create_dynamo_table_from_csv() # dynamo for city and dino id
        st.session_state.engine = create_type_to_id_engine() # sql dino id to dino type

        # tools
        st.session_state.toolkit = {} # a dictionary of tools
        # add retriever tool to tools
        st.session_state.toolkit["retriever_tool"] = prepare_retriever_tool(st.session_state.vector_store)

        @tool
        def get_DinoType_by_DinoID_Transported(dino_id: str) -> str:
            '''Find the type of Dino {e.g. T-Rex, Velociraptors} for a dino id.'''
            query = f"SELECT Name FROM id_name WHERE ID = '{dino_id}'"
            return pd.read_sql(query, st.session_state.engine).loc[0, 'Name']
        st.session_state.toolkit["find_dinoType_by_dinoID"] = get_DinoType_by_DinoID_Transported

        @tool
        def get_city_and_dinoid_by_date(date: str)->dict:
            '''In a dynamo database, get DinoID_Transported, Route_Number, and City by a given Date'''
            table = st.session_state.dynamodb.Table('TransportData')
            try:
                response = table.query(
                    KeyConditionExpression=Key('Date').eq(date)
                )
                return response['Items'][0]
            except:
                return 'Date not found in the database.'
        st.session_state.toolkit["get_city_and_dinoid_by_date"] = get_city_and_dinoid_by_date
        
        
        @tool
        def get_city_weather_on_date(city_name: str, date: str)->dict:
            '''Get the maximum and minimum temperature of a city on a specific date'''
            # '''Get the maximum and minimum temperature of a city on a specific date. A sample call looks like: get_city_weother_on_date(\'Toronto\', \'2024-03-19\'). Note that the date is formated in YYYY-MM-DD'''
            if '/' in date:
                month = date.split('/')[0]
                month = '0' * (2 - len(month)) + month
                date = date.split('/')[2] + '-' + month + '-' + date.split('/')[1]
            try:
                geolocator = Nominatim(user_agent="henryqiu")
                location = geolocator.geocode(city_name)
                longitude, latitude = location.longitude, location.latitude
                hourly = HourlyHistorical()
                daily = DailyHistorical()
                options = HistoricalOptions(latitude,
                                            longitude, 
                                            start_date=date,
                                            end_date=date)
                mgr = OpenMeteo(options, hourly.all(), daily.all())

                # Download data
                meteo = mgr.get_pandas()
                df = meteo[1].reset_index()  
                return {f'{city_name} maximum temperature': str(df.loc[0, 'temperature_2m_max']),
                        f'{city_name} minimum temperature': str(df.loc[0, 'temperature_2m_min'])}
            except:
                return {f'{city_name} maximum temperature': "Unknown",
                        f'{city_name} minimum temperature': "Unknown"}
        st.session_state.toolkit['get_city_weather_on_date'] = get_city_weather_on_date

        
        @tool
        def send_sms_to_phone(phone_number: str, message:str )->tuple:
            """
            Use this to send a message to a specified phone number.
            """
            sns = boto3.client('sns',
                region_name='us-east-1',
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            )

            try:
                # Send the SMS message
                response = sns.publish(
                    PhoneNumber=phone_number,
                    Message=message,
                    # Uncomment and specify the following attribute if needed
                    MessageAttributes={
                        'AWS.SNS.SMS.SenderID': {
                            'DataType': 'String',
                            'StringValue': 'Bot'
                        },
                        'AWS.SNS.SMS.SMSType': {
                            'DataType': 'String',
                            'StringValue': 'Transactional'  # or 'Promotional'
                        }
                    }
                )
                return response, None
            except (BotoCoreError, ClientError) as e:
                return None, str(e)
        st.session_state.toolkit['send_sms_to_phone'] = send_sms_to_phone
        
        st.session_state.messages = [] # every line is either a user input / an AI response alternatingly    

    # UI Components
    llm_need_initialization = False

    temperature = st.radio("Select the temperature for the LLM:", [0.0, 0.4, 0.9, 1.5])
    if temperature != st.session_state.previous_temperature: # if user has made a change to temperature
        llm_need_initialization = True
        st.session_state.previous_temperature = temperature

    model_choice = st.selectbox("Select the Chat Model/LLM:", ['gpt-3.5-turbo', 'gpt-4'])
    if model_choice != st.session_state.previous_model_choice:
        llm_need_initialization = True
        st.session_state.previous_model_choice = model_choice

    current_task = st.selectbox("Select the task:", ['task1', 'task2', 'task3', 'task4']) 
    # task3 and task4 has no functional difference, just the user's input differs
    # task4 is an advanced and end-to-end version of task3

    prompt_initialization = False

    # Extra credit: Allow users to change the system message
    system_message = st.text_input("Customize the system message:", "You are a helpful assistant. You will use all your tools to help me.")
    refine_system_message = st.checkbox("Refine the system message for task 4")
    if current_task == 'task4' and refine_system_message:
        system_message += "\nThe user will ask you to check the safety of a transportation on a given date. You need to find the dino and the city of the transportation of that date. Then, you find the type of dino given the dino id. Then, you check the documents for the safe temperature of that type of dino. You also check the local temperature of that city on that date. Finally, if the temperature is not safe for that transportation, craft a message on the actions to be taken to keep safety. The user will give you a phone number, and you will send this message to that phone number."
    st.write("system message: "+system_message)

    if system_message != st.session_state.previous_system_message:
        prompt_initialization = True
        st.session_state.previous_system_message = system_message

    # user input
    user_input = st.text_input("Type your message here:")
    send_button = st.button("Send")


    # Prompt here
    if 'prompt' not in st.session_state or prompt_initialization:
        # create a chat prompt that allows for system message
        st.session_state.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message), 
                MessagesPlaceholder("chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"), # scratchpad is a place for the agent to write notes
            ]   
        )


    # LLM here
    if llm_need_initialization or ('llm' not in st.session_state):
        st.session_state.llm = ChatOpenAI(temperature=temperature,
                                        model = model_choice,
                                        api_key=os.environ.get('OPENAI_API_KEY'))

    # TOOLs here
    tool_initialization = False
    if current_task != st.session_state.previous_current_task:
        tool_initialization = True
        st.session_state.previous_current_task = current_task
        st.session_state.tools = []
        if current_task == 'task1':
            st.session_state.tools = []
        elif current_task == 'task2':
            st.session_state.tools = [st.session_state.toolkit['retriever_tool']]
        elif current_task in ['task3', 'task4']:
            # now we take every tool "st.session_state.toolkit[tool_name]" for all tool_name in our toolkit
            st.session_state.tools = [st.session_state.toolkit[tool_name] for tool_name in st.session_state.toolkit.keys()]

    if tool_initialization or llm_need_initialization or prompt_initialization:

        if len(st.session_state.tools) > 0:
            # to create an agent, we need to pass in the llm, the tools, and the prompt
            agent = create_openai_tools_agent(st.session_state.llm, st.session_state.tools, st.session_state.prompt)
            st.session_state.agent_executor = AgentExecutor(agent=agent, tools=st.session_state.tools, verbose=True)

        st.session_state.messages = []


    # Handling user input
    if send_button: # this will be true (triggered) when the user clicks the send button
        if user_input:  # Ensure non-empty message
            # Append user input and simulated response to the session state messages
            with st.spinner("Agent thinking..."):
                # in task 2-4
                if current_task != 'task1': 
                    # the agent executor is a the agent that combines the llm, the tools, and the prompt template
                    # when being invoked by the user input and the chat history
                    # it will combine the user input, the system message, and the chat history into the prompt
                    # and it will respond to the prompt using the tools it has
                    # and returns a dictionary, where its response is under the key "output" 
                    response = st.session_state.agent_executor.invoke(
                        {
                            "input": user_input,
                            "chat_history": st.session_state.messages
                        },
                    )["output"] # llm + prompt + tools

                # in task 1
                else:
                    # first fill in the prompt with the user input and chat history
                    filled_prompt = st.session_state.prompt.invoke(
                        {
                            "input": user_input, 
                            "agent_scratchpad":[], 
                            "chat_history": st.session_state.messages
                        }
                    )
                    response = st.session_state.llm.invoke(filled_prompt).content # llm + prompt
                
            st.session_state.messages.append("User: "+user_input)
            st.session_state.messages.append("Agent: "+response)

    # Display at least the last 3 messages
    if 'messages' in st.session_state:
        # -10： to show the last 10 messages
        for message in st.session_state.messages[-10:]:
            st.text(message)

