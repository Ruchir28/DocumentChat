import string
import PyPDF2
import spacy



def embed_pdf(file_path: string):
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(num_pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            extracted_text = page.extract_text()
            text += extracted_text.strip()  # Extract text from each page
    return text

def read_txt_file(file_path: string):
    nlp = spacy.load("en_core_web_sm")
    text = ''
    with open(file_path, 'r') as file:
        text = file.read()
    # split sentences so that it can be used for embedding
    text = text.replace("\n", " ")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences
