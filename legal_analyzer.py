
import ollama

MODEL = "gemma4:e4b"


def analyze_document(text):
    prompt = f"""
You are a legal document triage assistant.

Analyse the following text extracted from a legal document.

Identify potential issues involving:
1. Financial obligations, fees, and penalties.
2. Termination and cancellation conditions.
3. Confidentiality obligations.
4. Liability and indemnity.
5. Unclear, unusual, or potentially restrictive clauses.

For each potential issue:
- Give it a short title.
- Classify its category.
- Assign a preliminary severity: Low, Medium, or High.
- Quote the relevant text exactly.
- Explain why it may deserve attention.
- Suggest what the reader should review.

Important rules:
- Use only the supplied document text as evidence.
- Never invent quotations or legal provisions.
- Do not assume that an unusual clause is illegal.
- If you find no clear issues, say so.
- Distinguish explicit statements from your own inferences.
- Do not present this analysis as professional legal advice.

DOCUMENT TEXT:
{text}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]


if __name__ == "__main__":
    sample_text = """
    Sample Service Agreement

    1. The customer must pay a monthly fee of Rs. 2000.

    2. The customer may terminate this agreement by giving
    90 days written notice.

    3. If the customer terminates before the first year,
    an early termination fee of Rs. 15000 will apply.

    4. The customer must not disclose confidential business
    information to unauthorized persons.

    5. The service provider accepts no liability whatsoever
    for any loss suffered by the customer.
    """

    result = analyze_document(sample_text)

    print("\n========== LEGAL TRIAGE REPORT ==========\n")
    print(result)

