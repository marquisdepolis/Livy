import PyPDF2

def read_pdf(file):
    file.seek(0)  # move the file cursor to the beginning
    pdf_reader = PyPDF2.PdfReader(file)
    
    # Check if the PDF is encrypted
    if pdf_reader.is_encrypted:
        print("Encrypted PDF file detected. Skipping...")
        return ""
    
    if len(pdf_reader.pages) == 0:
        raise ValueError("PDF file is empty")
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text