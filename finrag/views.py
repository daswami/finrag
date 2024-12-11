import os
from django.shortcuts import render
import pdfplumber
from openai import OpenAI
import tempfile
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar
from concurrent.futures import ThreadPoolExecutor

def upload_pdf(request):
    if request.method == 'POST':
        pdf_file = request.FILES.get('pdf_file')
        if pdf_file:
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_file.read())
                temp_pdf_path = temp_pdf.name

            query = "Retrieve the total loss/gain, name/type of account, and buy and sell price of each stock in the document. "
               
            def extract_chunks(pdf_path):
                print("Extracting text from PDF...")
                chunks = []
                current_chunk = []
                
                for page_layout in extract_pages(pdf_path):
                    for element in page_layout:
                        if isinstance(element, LTTextBox):
                            for text_line in element:
                                if isinstance(text_line, LTTextLine):
                                    text = text_line.get_text().strip()
                                    # Extract font size 
                                    font_sizes = [char.size for char in text_line if isinstance(char, LTChar)]
                                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                                    
                                    # KEY PART HERE
                                    # Splitting the text based on headers, which are usually larger than 12
                                    #Hoping that this keeps semantic meaning of sections, as the holdings are usually one section
                                    if avg_font_size > 12:  # Example threshold
                                        if current_chunk:
                                            chunks.append("\n".join(current_chunk))
                                            current_chunk = []
                                    current_chunk.append(text)
                
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                
                return chunks
            
            chunks = extract_chunks(temp_pdf_path)

           # My openAI key stored as an environment variable
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            #Generating embeddings for multiple chunks at once to reduce latency
            def get_embedding(texts, model="text-embedding-3-small"):
                print("Getting embedding for text...")
                # Handle both single text (query) and list of texts (chunks)
                if isinstance(texts, str):
                    texts = [texts]
                response = client.embeddings.create(
                    input=texts,
                    model=model
                )
                # Return single embedding for single text, list for multiple
                embeddings = [data.embedding for data in response.data]
                print("Embeddings generated")
                return embeddings[0] if len(embeddings) == 1 else embeddings
            
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            def get_top_chunks(query):
                # Get query embedding (will return single embedding)
                query_vector = get_embedding(query)
                
                # Get embeddings for all chunks at once
                # Process chunks in parallel using ThreadPoolExecutor
                # with ThreadPoolExecutor() as executor:
                #     chunk_embeddings = list(executor.map(lambda x: get_embedding([x])[0], chunks))
                chunk_embeddings = get_embedding(chunks)

                # Calculate similarities between query vector and each chunk embedding
                similarities = [cosine_similarity(query_vector, chunk_vector) for chunk_vector in chunk_embeddings]

                # Sort the chunks based on the similarity score
                relevant_chunks = sorted(zip(similarities, chunks), reverse=True, key=lambda x: x[0])

                top_chunks = [chunk for _, chunk in relevant_chunks[:3]]
                return top_chunks

            def ask_gpt(query):
                top_chunks = get_top_chunks(query)
                context = "\n\n".join(top_chunks)
                
                # Define the three specific queries for stock history
                queries = {
                    'total_gain_loss': 'What is the total gain/loss?',
                    'account_info': 'What is the name and type of account?',
                    'buy_sell_prices': 'What are the buy and sell prices of each stock?'
                }
                
                results = {}
                base_prompt = """You are an expert stock analysis assistant skilled at data extraction.
                You will receive relevant context that you will use to answer the question.
                
                {context}
                
                Answer the following question: {query}
                """
                
                # Run each query separately
                for query_type, query_text in queries.items():
                    print(f"Running LLM Model for {query_type}...")
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a crucial stock analysis assistant."},
                            {"role": "system", "content": "Data is extracted from a PDF document. The data is in the form of a table and text, and often for the holdings the tabular form is especially common. This means that holding text data was stripped of its vertical alignment, meaning that there are columns to locate and understanding how many columns there are can help identify which data points correspond to which column for each row."},
                            {"role": "user", "content": base_prompt.format(context=context, query=query_text)}
                        ]
                    )
                    results[query_type] = response.choices[0].message.content
                
                # Combine all results
                combined_response = f"""
Total Gain/Loss:
{results['total_gain_loss']}

Account Information:
{results['account_info']}

Buy and Sell Prices:
{results['buy_sell_prices']}
"""
                return combined_response.strip()

            # Clean up temporary file
            os.unlink(temp_pdf_path)
            
            #Render template with the response
            message = ask_gpt(query)
            return render(request, 'finrag/upload.html', {
                'message': message
            })


    return render(request, 'finrag/upload.html')