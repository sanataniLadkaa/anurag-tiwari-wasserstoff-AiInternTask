import os
import logging
import fitz  # PyMuPDF
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from nltk import download, word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import time  # For tracking performance
from statistics import mean  # For calculating average times

# Load BERT model for sentence embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Initialize NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

# MongoDB client setup
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_processing']
collection = db['documents']

# Ensure unique file paths in MongoDB (create index if not created)
collection.create_index("path", unique=True)

# Setup logging
logging.basicConfig(filename="performance_log.txt", level=logging.INFO)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.replace('\n', ' ').strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

def extract_keywords(text):
    """Extract domain-specific keywords from the text using stemming and lemmatization."""
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    filtered_words = [
        lemmatizer.lemmatize(stemmer.stem(word.lower())) 
        for word in words 
        if word.isalnum() and word.lower() not in stop_words
    ]

    fdist = FreqDist(filtered_words)
    keywords = fdist.most_common(10)
    domain_specific_keywords = [word for word, _ in keywords if len(word) > 3]
    return domain_specific_keywords

def bert_summary(text, top_n_sentences=5):
    """Summarize text using a BERT-based approach."""
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return None

        # Ensure that top_n_sentences does not exceed the number of available sentences
        top_n_sentences = min(top_n_sentences, len(sentences))

        embeddings = model.encode(sentences, convert_to_tensor=True)
        mean_embedding = embeddings.mean(dim=0)
        similarities = util.pytorch_cos_sim(embeddings, mean_embedding)
        top_indices = similarities.topk(k=top_n_sentences, largest=True).indices
        selected_sentences = [sentences[i] for i in top_indices]
        return ' '.join(selected_sentences).replace('\n', ' ').strip()

    except Exception as e:
        logging.error(f"Error creating BERT summary: {str(e)}")
        return None


def summarize_text(text):
    """Summarization using BERT only."""
    text_length = len(text)
    top_n_sentences = max(5, min(10, text_length // 500))
    try:
        summary = bert_summary(text, top_n_sentences=top_n_sentences)
        return summary if summary else text
    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}")
        return text

def process_pdf(pdf_path):
    """Process a single PDF document: extract metadata, text, summary, and keywords."""
    start_time = time.time()
    try:
        file_size_bytes = os.path.getsize(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        text = extract_text_from_pdf(pdf_path)
        if not text:
            return None

        summary = summarize_text(text)
        keywords = extract_keywords(text)

        # Normalize path for cross-platform compatibility
        normalized_path = os.path.normpath(pdf_path)

        document = {
            "name": pdf_name,
            "path": normalized_path,
            "size_mb": round(file_size_mb, 2),
            "summary": summary,
            "keywords": keywords
        }

        existing_document = collection.find_one({"path": normalized_path})

        if existing_document:
            update_result = collection.update_one({"path": normalized_path}, {"$set": document})
            if update_result.modified_count > 0:
                logging.info(f"Updated {pdf_name} successfully!")
        else:
            result = collection.insert_one(document)
            logging.info(f"Processed {pdf_name} successfully! Document inserted with ID: {result.inserted_id}")

        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        logging.info(f"Time taken to process {pdf_name}: {time_taken} seconds")

        return time_taken

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def process_pdfs_in_parallel(pdf_folder):
    """Process one or more PDFs concurrently, whether it's a folder or a single file."""
    if os.path.isfile(pdf_folder) and pdf_folder.endswith('.pdf'):
        pdf_files = [os.path.normpath(pdf_folder)]
    elif os.path.isdir(pdf_folder):
        pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    else:
        print(f"{pdf_folder} is not a valid file or directory.")
        return

    times = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        times = list(executor.map(process_pdf, pdf_files))

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    average_time = round(mean([t for t in times if t is not None]), 2)

    logging.info(f"Processed {len(pdf_files)} PDFs in parallel.")
    logging.info(f"Total time taken: {total_time} seconds")
    logging.info(f"Average time per PDF: {average_time} seconds")

    print(f"Processed {len(pdf_files)} PDFs.")
    print(f"Total time: {total_time} seconds")
    print(f"Average time per document: {average_time} seconds")

# if __name__ == "__main__":
#     pdf_folder_path = "pdf_downloads"
#     process_pdfs_in_parallel(pdf_folder_path)
import os
import logging
import fitz  # PyMuPDF
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from nltk import download, word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import time  # For tracking performance
from statistics import mean  # For calculating average times

# Load BERT model for sentence embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Initialize NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

# MongoDB client setup
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_processing']
collection = db['documents']

# Ensure unique file paths in MongoDB (create index if not created)
collection.create_index("path", unique=True)

# Setup logging
logging.basicConfig(filename="performance_log.txt", level=logging.INFO)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.replace('\n', ' ').strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

def extract_keywords(text):
    """Extract domain-specific keywords from the text using stemming and lemmatization."""
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    filtered_words = [
        lemmatizer.lemmatize(stemmer.stem(word.lower())) 
        for word in words 
        if word.isalnum() and word.lower() not in stop_words
    ]

    fdist = FreqDist(filtered_words)
    keywords = fdist.most_common(10)
    domain_specific_keywords = [word for word, _ in keywords if len(word) > 3]
    return domain_specific_keywords

def bert_summary(text, top_n_sentences=5):
    """Summarize text using a BERT-based approach."""
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return None

        # Ensure that top_n_sentences does not exceed the number of available sentences
        top_n_sentences = min(top_n_sentences, len(sentences))

        embeddings = model.encode(sentences, convert_to_tensor=True)
        mean_embedding = embeddings.mean(dim=0)
        similarities = util.pytorch_cos_sim(embeddings, mean_embedding)
        top_indices = similarities.topk(k=top_n_sentences, largest=True).indices
        selected_sentences = [sentences[i] for i in top_indices]
        return ' '.join(selected_sentences).replace('\n', ' ').strip()

    except Exception as e:
        logging.error(f"Error creating BERT summary: {str(e)}")
        return None


def summarize_text(text):
    """Summarization using BERT only."""
    text_length = len(text)
    top_n_sentences = max(5, min(10, text_length // 500))
    try:
        summary = bert_summary(text, top_n_sentences=top_n_sentences)
        return summary if summary else text
    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}")
        return text

def process_pdf(pdf_path):
    """Process a single PDF document: extract metadata, text, summary, and keywords."""
    start_time = time.time()
    try:
        file_size_bytes = os.path.getsize(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        text = extract_text_from_pdf(pdf_path)
        if not text:
            return None

        summary = summarize_text(text)
        keywords = extract_keywords(text)

        # Normalize path for cross-platform compatibility
        normalized_path = os.path.normpath(pdf_path)

        document = {
            "name": pdf_name,
            "path": normalized_path,
            "size_mb": round(file_size_mb, 2),
            "summary": summary,
            "keywords": keywords
        }

        existing_document = collection.find_one({"path": normalized_path})

        if existing_document:
            update_result = collection.update_one({"path": normalized_path}, {"$set": document})
            if update_result.modified_count > 0:
                logging.info(f"Updated {pdf_name} successfully!")
        else:
            result = collection.insert_one(document)
            logging.info(f"Processed {pdf_name} successfully! Document inserted with ID: {result.inserted_id}")

        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        logging.info(f"Time taken to process {pdf_name}: {time_taken} seconds")

        return time_taken

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def process_pdfs_in_parallel(pdf_folder):
    """Process one or more PDFs concurrently, whether it's a folder or a single file."""
    if os.path.isfile(pdf_folder) and pdf_folder.endswith('.pdf'):
        pdf_files = [os.path.normpath(pdf_folder)]
    elif os.path.isdir(pdf_folder):
        pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    else:
        print(f"{pdf_folder} is not a valid file or directory.")
        return

    times = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        times = list(executor.map(process_pdf, pdf_files))

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    average_time = round(mean([t for t in times if t is not None]), 2)

    logging.info(f"Processed {len(pdf_files)} PDFs in parallel.")
    logging.info(f"Total time taken: {total_time} seconds")
    logging.info(f"Average time per PDF: {average_time} seconds")

    print(f"Processed {len(pdf_files)} PDFs.")
    print(f"Total time: {total_time} seconds")
    print(f"Average time per document: {average_time} seconds")

# if __name__ == "__main__":
#     pdf_folder_path = "pdf_downloads"
#     process_pdfs_in_parallel(pdf_folder_path)
import os
import logging
import fitz  # PyMuPDF
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from nltk import download, word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import time  # For tracking performance
from statistics import mean  # For calculating average times

# Load BERT model for sentence embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Initialize NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

# MongoDB client setup
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_processing']
collection = db['documents']

# Ensure unique file paths in MongoDB (create index if not created)
collection.create_index("path", unique=True)

# Setup logging
logging.basicConfig(filename="performance_log.txt", level=logging.INFO)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.replace('\n', ' ').strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

def extract_keywords(text):
    """Extract domain-specific keywords from the text using stemming and lemmatization."""
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    filtered_words = [
        lemmatizer.lemmatize(stemmer.stem(word.lower())) 
        for word in words 
        if word.isalnum() and word.lower() not in stop_words
    ]

    fdist = FreqDist(filtered_words)
    keywords = fdist.most_common(10)
    domain_specific_keywords = [word for word, _ in keywords if len(word) > 3]
    return domain_specific_keywords

def bert_summary(text, top_n_sentences=5):
    """Summarize text using a BERT-based approach."""
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return None

        # Ensure that top_n_sentences does not exceed the number of available sentences
        top_n_sentences = min(top_n_sentences, len(sentences))

        embeddings = model.encode(sentences, convert_to_tensor=True)
        mean_embedding = embeddings.mean(dim=0)
        similarities = util.pytorch_cos_sim(embeddings, mean_embedding)
        top_indices = similarities.topk(k=top_n_sentences, largest=True).indices
        selected_sentences = [sentences[i] for i in top_indices]
        return ' '.join(selected_sentences).replace('\n', ' ').strip()

    except Exception as e:
        logging.error(f"Error creating BERT summary: {str(e)}")
        return None


def summarize_text(text):
    """Summarization using BERT only."""
    text_length = len(text)
    top_n_sentences = max(5, min(10, text_length // 500))
    try:
        summary = bert_summary(text, top_n_sentences=top_n_sentences)
        return summary if summary else text
    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}")
        return text

def process_pdf(pdf_path):
    """Process a single PDF document: extract metadata, text, summary, and keywords."""
    start_time = time.time()
    try:
        file_size_bytes = os.path.getsize(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        text = extract_text_from_pdf(pdf_path)
        if not text:
            return None

        summary = summarize_text(text)
        keywords = extract_keywords(text)

        # Normalize path for cross-platform compatibility
        normalized_path = os.path.normpath(pdf_path)

        document = {
            "name": pdf_name,
            "path": normalized_path,
            "size_mb": round(file_size_mb, 2),
            "summary": summary,
            "keywords": keywords
        }

        existing_document = collection.find_one({"path": normalized_path})

        if existing_document:
            update_result = collection.update_one({"path": normalized_path}, {"$set": document})
            if update_result.modified_count > 0:
                logging.info(f"Updated {pdf_name} successfully!")
        else:
            result = collection.insert_one(document)
            logging.info(f"Processed {pdf_name} successfully! Document inserted with ID: {result.inserted_id}")

        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        logging.info(f"Time taken to process {pdf_name}: {time_taken} seconds")

        return time_taken

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def process_pdfs_in_parallel(pdf_folder):
    """Process one or more PDFs concurrently, whether it's a folder or a single file."""
    if os.path.isfile(pdf_folder) and pdf_folder.endswith('.pdf'):
        pdf_files = [os.path.normpath(pdf_folder)]
    elif os.path.isdir(pdf_folder):
        pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    else:
        print(f"{pdf_folder} is not a valid file or directory.")
        return

    times = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        times = list(executor.map(process_pdf, pdf_files))

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    average_time = round(mean([t for t in times if t is not None]), 2)

    logging.info(f"Processed {len(pdf_files)} PDFs in parallel.")
    logging.info(f"Total time taken: {total_time} seconds")
    logging.info(f"Average time per PDF: {average_time} seconds")

    print(f"Processed {len(pdf_files)} PDFs.")
    print(f"Total time: {total_time} seconds")
    print(f"Average time per document: {average_time} seconds")

# if __name__ == "__main__":
#     pdf_folder_path = "pdf_downloads"
#     process_pdfs_in_parallel(pdf_folder_path)
