# WellBot.AI Backend

## Overview
This is the backend service for WellBot.AI, an AI chatbot for healthcare that assists users in analyzing symptoms and predicting potential diseases using Machine Learning.

## Prerequisites
- Python 3.x
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_folder>
    ```

2. Navigate to the `Backend` directory:
    ```sh
    cd Backend
    ```

3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Backend

To start the backend server, run the following command in the `Backend` directory:
```sh
uvicorn app.main:app --reload --port 8999
```

# WellBot.AI Frontend

## Overview
This is the frontend for WellBot.AI, an AI chatbot for healthcare that assists users in analyzing symptoms and predicting potential diseases using Machine Learning.

## Prerequisites
- A web browser (e.g., Chrome, Firefox, Safari)

## Running the Frontend

To start the frontend, follow these steps:

1. Navigate to the `Frontend` directory:
    ```sh
    cd Frontend
    ```

2. Open the `index.html` file in your web browser:
    ```sh
    open index.html
    ```

This will launch the frontend interface in your default web browser.

## Usage

Once the frontend is running, you can interact with the chatbot by typing your symptoms into the input field and clicking the "Send" button.

## License

This project is licensed under the MIT License.


