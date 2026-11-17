
evaluation_queries = [
    # Page 2: Introduction and perceptron architecture
    {
        "question": "What is a perceptron, and who introduced it?",
        "relevant_pages": [2]
    },
    {
        "question": "How does a perceptron calculate its output?",
        "relevant_pages": [2]
    },
    {
        "question": "What are the main components of a perceptron?",
        "relevant_pages": [2]
    },
    {
        "question": "How are input features and weights used in a perceptron?",
        "relevant_pages": [2]
    },
    {
        "question": "What activation mechanism does the basic perceptron use?",
        "relevant_pages": [2]
    },

    # Page 3: Bias and decision boundary
    {
        "question": "What is the role of the bias in a perceptron?",
        "relevant_pages": [3]
    },
    {
        "question": "How does bias affect the perceptron's decision boundary?",
        "relevant_pages": [3]
    },
    {
        "question": "How is the decision boundary geometrically interpreted?",
        "relevant_pages": [3, 4]
    },
    {
        "question": "What mathematical condition defines the perceptron's decision boundary?",
        "relevant_pages": [3]
    },

    # Pages 4–5: Linear separability and learning rule
    {
        "question": "What does it mean for a dataset to be linearly separable?",
        "relevant_pages": [4]
    },
    {
        "question": "What is the perceptron learning update rule?",
        "relevant_pages": [4, 5]
    },
    {
        "question": "Under what circumstances are the perceptron's weights updated?",
        "relevant_pages": [4, 5]
    },
    {
        "question": "What happens to the weights when an example is misclassified?",
        "relevant_pages": [5]
    },
    {
        "question": "Why does the perceptron update rule correct a misclassified example?",
        "relevant_pages": [5]
    },
    {
        "question": "What is the role of the learning rate in the perceptron update rule?",
        "relevant_pages": [5]
    },

    # Pages 6–7: AND function and convergence
    {
        "question": "How does the perceptron learn the AND function?",
        "relevant_pages": [6, 7]
    },
    {
        "question": "What are the input-output combinations in the AND truth table?",
        "relevant_pages": [6]
    },
    {
        "question": "How does training adjust the perceptron to classify AND inputs correctly?",
        "relevant_pages": [6, 7]
    },
    {
        "question": "What does convergence mean in perceptron learning?",
        "relevant_pages": [7, 8]
    },
    {
        "question": "What are the assumptions and mistake bound of the perceptron convergence theorem?",
        "relevant_pages": [7, 8]
    },

    # Page 8: Convergence theorem
    {
        "question": "Under what conditions does the perceptron convergence theorem guarantee learning?",
        "relevant_pages": [8]
    },
    {
        "question": "What does the mistake bound tell us about perceptron training?",
        "relevant_pages": [8]
    },

    # Page 9: XOR and multilayer networks
    {
        "question": "Why can't a single perceptron implement XOR?",
        "relevant_pages": [9]
    },
    {
        "question": "Why is the XOR problem not linearly separable?",
        "relevant_pages": [9]
    },
    {
        "question": "How do multilayer networks solve the XOR problem?",
        "relevant_pages": [9]
    },
    {
        "question": "What is the limitation of a single-layer perceptron demonstrated by XOR?",
        "relevant_pages": [9]
    },
    {
        "question": "What is ADALINE, and how is it related to the perceptron?",
        "relevant_pages": [9, 10]
    },

    # Pages 9–10: ADALINE and model comparison
    {
        "question": "How does the ADALINE delta rule differ from the perceptron learning rule?",
        "relevant_pages": [9, 10]
    },
    {
        "question": "What error measure is used by ADALINE during training?",
        "relevant_pages": [10]
    },
    {
        "question": "How does ADALINE update its weights during training?",
        "relevant_pages": [10, 11]
    },
    {
        "question": "How do perceptron, logistic regression, and linear SVM differ?",
        "relevant_pages": [10]
    },
    {
        "question": "How does the perceptron differ from logistic regression in terms of output and learning?",
        "relevant_pages": [10]
    },
    {
        "question": "What distinguishes a linear SVM from a perceptron?",
        "relevant_pages": [10]
    },

    # Page 11: Mathematical formulas
    {
        "question": "What are the important formulas of the perceptron?",
        "relevant_pages": [11]
    },
    {
        "question": "What is the mathematical expression for the perceptron's weighted sum?",
        "relevant_pages": [11]
    },
    {
        "question": "What mathematical formula describes the perceptron weight update?",
        "relevant_pages": [11]
    }
]
