
import pandas as pd

df = pd.read_csv("answer_evaluation_results.csv")

score_columns = [
    "accuracy_score",
    "correctness_score",
    "faithfulness_score",
    "completeness_score",
    "citation_accuracy_score"
]

for column in score_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nANSWER QUALITY EVALUATION")
print("=" * 55)

for column in score_columns:
    average = df[column].mean()

    if pd.notna(average):
        percentage = (average / 2) * 100
        print(f"{column.replace('_score', '').title():<25}{percentage:.2f}%")

print("\nABSTENTION EVALUATION")
print("=" * 55)

abstention_cases = df[df["should_abstain"] == True]

correct_refusals = abstention_cases[
    abstention_cases["generated_answer"].str.contains(
        "does not contain|cannot answer|no mention|no information",
        case=False,
        na=False
    )
]

print(f"Out-of-document questions: {len(abstention_cases)}")
print(f"Correct refusals: {len(correct_refusals)}")

if len(abstention_cases) > 0:
    refusal_rate = len(correct_refusals) / len(abstention_cases) * 100
    print(f"Refusal rate: {refusal_rate:.2f}%")

print("\nEvaluation completed.")
