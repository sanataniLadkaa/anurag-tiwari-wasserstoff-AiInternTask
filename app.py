from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
import shutil
import logging

# Importing your PDF processing functions
from main import process_pdfs_in_parallel  # Ensure this import is correct

app = FastAPI()

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Setting up templates
templates = Jinja2Templates(directory="templates")

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Path where uploaded files will be saved
UPLOAD_DIRECTORY = "./uploaded_pdfs"

# Create the upload directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page with file upload form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_pdfs(request: Request, files: list[UploadFile] = File(...)):
    """Handle multiple PDF file uploads."""
    summaries = []
    
    # Check if any files were uploaded
    if not files:
        logging.error("No files uploaded.")
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        # Loop through each uploaded file
        for file in files:
            # Check if the uploaded file is a PDF
            if not file.filename.endswith('.pdf'):
                logging.error(f"Invalid file type: {file.filename}")
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")

            # Save each uploaded file
            file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
            with open(file_location, "wb+") as f:
                shutil.copyfileobj(file.file, f)
                logging.info(f"Saved file {file.filename} to {file_location}")

            # Process the uploaded PDF
            try:
                summary = process_pdfs_in_parallel(file_location)  # Ensure this function handles the PDF correctly
                summaries.append({"filename": file.filename, "summary": summary})
            except Exception as processing_error:
                logging.error(f"Error processing {file.filename}: {processing_error}")
                summaries.append({"filename": file.filename, "error": str(processing_error)})

        # Return result after processing all PDFs
        return templates.TemplateResponse("result.html", {"request": request, "summaries": summaries})

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": str(e)}
