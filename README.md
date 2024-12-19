# Simple Realtime Augmented Generation (RAG) with LangChain and Chainlit

This repository demonstrates a simple implementation of Realtime Augmented Generation using Chainlit. It showcases how to integrate Chainlit with LLMs through LangChain to generate and augment content in real-time. The project includes examples and documentation to help you get started quickly.

## Features
- Integration with multiple AI models (OpenAI GPT and VertexAI Gemini)

## Getting Started

To get started with this project, follow the instructions below:

1. Clone the repository:
    ```sh
    git clone https://github.com/ghif/llm-rag.git
    ```
2. Navigate to the project directory:
    ```sh
    cd llm-rag
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Run the application:
    Without RAG:

    ```sh
    chainlit run app_plain.py
    ```

    With RAG:
    ```sh
    chainlit run app.py
    ```

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.