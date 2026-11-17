
import os
import csv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import generate_answer


evaluation_cases = [
    {
        "question": "What is a perceptron, and who introduced it?",
        "expected": "The perceptron is an early trainable artificial neuron introduced by Frank Rosenblatt in 1958.",
        "should_abstain": False
    },
    {
        "question": "How does a perceptron calculate its output?",
        "expected": "It calculates the weighted sum of inputs plus bias and applies a sign activation function.",
        "should_abstain": False
    },
    {
        "question": "What is the role of bias in a perceptron?",
        "expected": "Bias shifts the decision boundary and controls how easily the neuron activates.",
        "should_abstain": False
    },
    {
        "question": "Why can't a single perceptron implement XOR?",
        "expected": "XOR is not linearly separable, so one straight decision boundary cannot separate its classes.",
        "should_abstain": False
    },
    {
        "question": "What is the ADALINE delta rule?",
        "expected": "The Widrow-Hoff or LMS rule updates weights using the prediction error multiplied by the input and learning rate.",
        "should_abstain": False
    },
    {
        "question": "What is quantum computing?",
        "expected": "The uploaded document does not contain sufficient information.",
        "should_abstain": True
    },
    {
        "question": "Who invented the light bulb?",
        "expected": "The uploaded document does not contain sufficient information.",
        "should_abstain": True
    },
    {
        "question": "What is convolution?",
        "expected": "The uploaded document does not contain sufficient information.",
        "should_abstain": True
    }
]


def main():

    print("\nLoading document...")

    documents = load_pdf(
        "data/sample_documents/sample.pdf"
    )

    chunks = split_documents(documents)

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    print("Document ready.")
    print(f"Evaluation cases: {len(evaluation_cases)}")

    output_rows = []

    for index, case in enumerate(evaluation_cases, start=1):

        question = case["question"]

        print("\n" + "=" * 65)
        print(f"QUESTION {index}: {question}")
        print("=" * 65)

        result = generate_answer(
            vector_store,
            question,
            k=3
        )

        answer = result["answer"]

        sources = result.get("sources", [])

        pages = sorted({
            int(source["metadata"]["page"])
            for source in sources
        })

        print("\nGenerated answer:")
        print(answer)

        print("\nRetrieved pages:")
        print(pages)

        output_rows.append({
            "question": question,
            "expected_answer": case["expected"],
            "generated_answer": answer,
            "retrieved_pages": pages,
            "should_abstain": case["should_abstain"],
            "correctness_score": "",
            "faithfulness_score": "",
            "completeness_score": "",
            "citation_accuracy_score": ""
        })

    output_file = "answer_evaluation_results.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=output_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 65)
    print("ANSWER EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
