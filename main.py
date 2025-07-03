import fitz  
import os
import re

def extract_pdf(pdf_folder):
    output_txt_file = 'indian_laws_combined.txt'

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    pdf_files.sort()

    with open(output_txt_file, 'w', encoding='utf-8') as output_file:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f'Reading {pdf_path}...')

            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    output_file.write(text + '\n')
                else:
                    print(f" Blank or image-based page detected in {pdf_file}")
            doc.close()

    print(' All PDFs successfully merged into indian_laws_combined.txt!')

pdf_folder = 'data'
extract_pdf(pdf_folder)


def clean_document(text):
    # First, remove special characters but keep essential punctuation and newlines
    text = re.sub(r'[^\w\s\.\,\:\;\n]', '', text)
    
    # Split the text into lines, strip whitespace from each, and filter out empty lines
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Join the non-empty lines back together with a single newline
    text = '\n'.join(non_empty_lines)
    
    # Now, replace multiple spaces within lines with a single space
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

# Read the text file
with open('indian_laws_combined.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Clean the text
clean_text = clean_document(raw_text)

# Print a sample
print(clean_text[0:1000])

