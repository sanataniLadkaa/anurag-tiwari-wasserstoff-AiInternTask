from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
import shutil

# Assuming your PDF processing functions are imported
from main import process_pdfs_in_parallel  # Replace with the name of your script

app = FastAPI()

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
    try:
        # Loop through each uploaded file
        for file in files:
            # Save each uploaded file
            file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
            with open(file_location, "wb+") as f:
                shutil.copyfileobj(file.file, f)

            # Process the uploaded PDF
            summary = process_pdfs_in_parallel(file_location)  # Assuming this returns a summary
            summaries.append({"filename": file.filename, "summary": summary})

        # Return result after processing all PDFs
        return templates.TemplateResponse("result.html", {"request": request, "summaries": summaries})

    except Exception as e:
        return {"error": str(e)}
