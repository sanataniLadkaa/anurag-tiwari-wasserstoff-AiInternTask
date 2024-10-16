# AI Intern Task - Wasserstoff

This project is part of my AI internship tasks at Wasserstoff. It involves the development of an AI-powered pipeline that performs PDF ingestion, text extraction, summarization, and keyword extraction, along with handling MongoDB storage and concurrency.

## Project Overview

This project is designed to handle PDF documents in a pipeline format. The tasks include:
- Extracting text from PDFs.
- Summarizing the text using BERT.
- Extracting keywords from the text.
- Storing the processed data in a MongoDB database.
- Implementing concurrency to handle multiple documents efficiently.
- Providing performance metrics to monitor the process.

## Features

- **PDF Text Extraction**: Extracts text from PDF files using the `PyMuPDF` library.
- **Summarization**: Summarizes extracted text using the BERT model.
- **Keyword Extraction**: Extracts significant keywords from the text.
- **MongoDB Integration**: Stores processed data in MongoDB.
- **Concurrency**: Handles multiple documents simultaneously for better performance.
- **Error Handling**: Captures and logs errors during the PDF processing pipeline.

## Prerequisites

- Python 3.8+
- MongoDB
- Git
- Virtual Environment (venv)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sanataniLadkaa/anurag-tiwari-wasserstoff-AiInternTask.git
   cd anurag-tiwari-wasserstoff-AiInternTask
