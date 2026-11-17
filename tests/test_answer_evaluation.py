
import os
import csv
import time

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
        "expected_pages": [2],
        "should_abstain": False
    },
    {
        "question": "How does a perceptron calculate its output?",
        "expected": "It calculates the weighted sum of inputs plus bias and applies a threshold or sign activation function.",
        "expected_pages": [2],
        "should_abstain": False
    },
    {
        "question": "What are the main components of a perceptron?",
        "expected": "A perceptron uses input features, weights, bias, a summation operation, and an activation function.",
        "expected_pages": [2],
        "should_abstain": False
    },
    {
        "question": "What is the role of bias in a perceptron?",
        "expected": "Bias shifts the decision boundary and affects the activation condition.",
        "expected_pages": [3],
        "should_abstain": False
    },
    {
        "question": "How is the decision boundary geometrically interpreted?",
        "expected": "It is the boundary separating the input regions classified into different classes.",
        "expected_pages": [3, 4],
        "should_abstain": False
    },
    {
        "question": "What does it mean for a dataset to be linearly separable?",
        "expected": "The classes can be separated by a linear decision boundary.",
        "expected_pages": [4],
        "should_abstain": False
    },
    {
        "question": "Why does the perceptron update rule correct a misclassified example?",
        "expected": "The update adjusts weights in the direction needed to reduce the classification error for the misclassified example.",
        "expected_pages": [4, 5],
        "should_abstain": False
    },
    {
        "question": "How does the perceptron learn the AND function?",
        "expected": "It learns weights and bias that classify the AND truth-table inputs correctly.",
        "expected_pages": [6, 7],
        "should_abstain": False
    },
    {
        "question": "What are the assumptions of the perceptron convergence theorem?",
        "expected": "The theorem assumes linearly separable training data and a suitable perceptron learning process.",
        "expected_pages": [7, 8],
        "should_abstain": False
    },
    {
        "question": "What does the mistake bound tell us about perceptron training?",
        "expected": "It provides an upper bound on the number of classification mistakes under the theorem's assumptions.",
        "expected_pages": [8],
        "should_abstain": False
    },
    {
        "question": "Why can't a single perceptron implement XOR?",
        "expected": "XOR is not linearly separable, so one linear decision boundary cannot separate its classes.",
        "expected_pages": [9],
        "should_abstain": False
    },
    {
        "question": "How do multilayer networks solve the XOR problem?",
        "expected": "Multilayer networks combine multiple computational units to represent decision boundaries that a single perceptron cannot represent.",
        "expected_pages": [9],
        "should_abstain": False
    },
    {
        "question": "What is the ADALINE delta rule?",
        "expected": "The Widrow-Hoff or LMS rule updates weights using prediction error, input values, and a learning rate.",
        "expected_pages": [9, 10, 11],
        "should_abstain": False
    },
    {
        "question": "What error measure is used by ADALINE during training?",
        "expected": "ADALINE uses a squared-error-based learning objective.",
        "expected_pages": [10],
        "should_abstain": False
    },
    {
        "question": "How do perceptron, logistic regression, and linear SVM differ?",
        "expected": "They differ in their learning objectives, output interpretation, and classification approaches.",
        "expected_pages": [10],
        "should_abstain": False
    },
    {
        "question": "What are the important mathematical formulas of the perceptron?",
        "expected": "The document presents formulas for the weighted sum, activation, and weight update.",
        "expected_pages": [11],
        "should_abstain": False
    },
    {
        "question": "What is quantum computing?",
        "expected": "The uploaded document does not contain sufficient information.",
        "expected_pages": [],
        "should_abstain": True
    },
    {
        "question": "Who invented the light bulb?",
        "expected": "The uploaded document does not contain sufficient information.",
        "expected_pages": [],
        "should_abstain": True
    },
    {
        "question": "What is convolution in image processing?",
        "expected": "The uploaded document does not contain sufficient information.",
        "expected_pages": [],
        "should_abstain": True
    },
    {
        "question": "Explain the architecture of a Transformer model.",
        "expected": "The uploaded document does not contain sufficient information.",
        "expected_pages": [],
        "should_abstain": True
    },
    {
        "question": "What is backpropagation in deep neural networks?",
        "expected": "The uploaded document does not contain sufficient information.",
        "expected_pages": [],
        "should_abstain": True
    }
]


def save_results(output_rows, output_file):

    fieldnames = [
        "question",
        "expected_answer",
        "generated_answer",
        "expected_pages",
        "retrieved_pages",
        "should_abstain",
        "correctness_score",
        "faithfulness_score",
        "completeness_score",
        "citation_accuracy_score"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)


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

    print("\nDocument ready.")
    print(f"Total evaluation cases: {len(evaluation_cases)}")

    output_file = "answer_evaluation_results.csv"

    output_rows = []

    for index, case in enumerate(evaluation_cases, start=1):

        question = case["question"]

        print("\n" + "=" * 65)
        print(f"QUESTION {index}: {question}")
        print("=" * 65)

        result = None

        for attempt in range(1, 4):

            try:

                result = generate_answer(
                    vector_store,
                    question,
                    k=3
                )

                break

            except Exception as error:

                print(f"Attempt {attempt} failed: {error}")

                if attempt < 3:

                    wait_time = 10 * attempt

                    print(f"Retrying in {wait_time} seconds...")

                    time.sleep(wait_time)

        if result is None:

            print(f"Skipping Question {index} after 3 failed attempts.")

            continue

        answer = result["answer"]

        sources = result.get("sources", [])

        pages = sorted({
            int(source["metadata"]["page"])
            for source in sources
        })

        print("\nGenerated answer:")
        print(answer)

        print("\nExpected pages:")
        print(case["expected_pages"])

        print("Retrieved pages:")
        print(pages)

        output_rows.append({
            "question": question,
            "expected_answer": case["expected"],
            "generated_answer": answer,
            "expected_pages": case["expected_pages"],
            "retrieved_pages": pages,
            "should_abstain": case["should_abstain"],
            "correctness_score": "",
            "faithfulness_score": "",
            "completeness_score": "",
            "citation_accuracy_score": ""
        })

        save_results(
            output_rows,
            output_file
        )

        print(f"\nCheckpoint saved: {len(output_rows)} questions")

    print("\n" + "=" * 65)
    print("ANSWER EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Successfully evaluated: {len(output_rows)}/{len(evaluation_cases)}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
