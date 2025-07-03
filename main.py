import fitz  
import os

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