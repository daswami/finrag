# Financial RAG Tool

A tool that extracts key data from stock history documents. 
Here is a little more about how the tool works. 

## Features
- PDF text extraction with pdfminer
- Semantic chunking of PDF content
- LLM powered data extraction
- Django web interface

## LLM Techniques
1. Custom embeddings to reduce latency and maintain the context window
2. Semantic chunking to preprocess data based on likely sections to make embeddings more complete
3. Embedding Creation: Run embeddings as a list rather than calling each string individually to improve latency
4. Complex query decomposition: Prompt engineering technique to make multiple, smaller calls to LLM to improve accuracy
5. Cost reduction in model choice: Utilized gpt-4 traditional model rather than other document models for lower cost/latency

The application was built with scalability in mind, with the goal of increasing accuracy across a wide variety of documents with lower costs and latency. 

## Future Improvements
- Multi-threaded extraction: Further reduce latency in generating embeddings and smaller model queries through parallelization
- Vector database: A scalable solution to store embeddings
- Prompt Refinements: Generate mutliple solutions and convert that information into context to further improve the LLM accuracy

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key

## Usage
1. Upload a PDF file of the stock history through the UI
2. The system will extract relevant data
3. Results will be displayed on the page
