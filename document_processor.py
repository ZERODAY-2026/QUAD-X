#this was used just to check our document_processor module

import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.
    """

    document = pymupdf.open(pdf_path)
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF: ")

    pages = extract_text_from_pdf(pdf_path)

    for page in pages:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"])

    print("\nPDF extraction complete!")

