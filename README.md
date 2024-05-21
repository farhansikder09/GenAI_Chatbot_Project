<img width="846" alt="Screenshot at May 20 16-50-39" src="https://github.com/farhansikder09/GenAI_Chatbot_Project/assets/149964944/d7aabdf3-0536-4f74-b66f-bf85e722d8b8">


## ClassProject_GenAI-Chatbot-AWS

## Overview
This project involves developing a GenAI chatbot integrated with AWS services to manage dinosaur transportation data. The chatbot leverages Streamlit for the interface, vector stores for data retrieval, and AWS DynamoDB and SNS for data management and communication. The solution includes an end-to-end workflow that automates the process of querying data, assessing environmental conditions, and sending status updates.

## Features
- Streamlit-based interactive chatbot interface.
- Vector store for efficient data retrieval.
- Integration with AWS DynamoDB for data storage.
- AWS SNS for sending notifications.
- End-to-end workflow for managing dinosaur transport data and environmental assessments.

## Project Structure
```
.
├── code
│   ├── streamlit_chatbot.py  # Streamlit chatbot implementation
│   ├── vector_store.py       # Vector store preparation and querying
│   ├── dynamodb_tools.py     # DynamoDB table management and data insertion
│   ├── weather_retrieval.py  # Weather data retrieval using OpenMeteo API
│   ├── sms_notification.py   # SMS sending functionality using AWS SNS
│   ├── end_to_end_workflow.py# End-to-end workflow integration
├── data
│   ├── TRexSafeTemp.pdf      # Document for T-Rex safety temperature
│   ├── VelociraptorsSafeTemp.pdf # Document for Velociraptor safety temperature
├── docs
│   ├── Project_Report.pdf    # Detailed project report
│   ├── Presentation_Slides.pdf # Project presentation slides
└── README.md                 # Project documentation
```

## Setup and Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Jurassic-GenAI-Chatbot-AWS.git
    cd Jurassic-GenAI-Chatbot-AWS
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:
    - Create a `.env` file in the root directory and add your API keys and other sensitive information:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    AWS_ACCESS_KEY_ID=your_aws_access_key_id
    AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    OPENMETEO_API_KEY=your_openmeteo_api_key
    ```

4. **Run the Streamlit Chatbot**:
    ```bash
    streamlit run code/streamlit_chatbot.py
    ```

## Usage
- **Streamlit Chatbot**: Interact with the chatbot via the Streamlit interface to query dinosaur transport data and receive updates.
- **Vector Store**: Utilize the vector store for efficient retrieval of information from the provided documents.
- **DynamoDB Management**: Manage transportation data using AWS DynamoDB.
- **Weather Data Retrieval**: Retrieve historical and current weather data for specified locations.
- **SMS Notifications**: Send status updates and notifications via AWS SNS.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request for review.


## Acknowledgements
- Instructor: Sudi Bhattacharya
- Team Members: Md Farhan Ishmam Sikder, Henry Qiu, Mathis Girard-Soppet
```
