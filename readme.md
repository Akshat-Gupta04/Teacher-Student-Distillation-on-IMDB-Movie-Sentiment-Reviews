# Teacher-Student Sentiment Distillation on IMDB Movie Reviews

This project demonstrates **knowledge distillation** for sentiment analysis trained on the IMDB movie reviews dataset. In knowledge distillation, a large, highly-trained teacher model is used to annotate unlabeled or weakly labeled data. A smaller, more efficient student model is then trained on these teacher-generated annotations to mimic the teacher's behavior. Here, we use the teacher model [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) (a robust sentiment classifier fine-tuned on Twitter data) to annotate a subset of the IMDB dataset. The student model, [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased), is then fine-tuned for 10 epochs on this teacher-annotated data. Finally, the project evaluates and visualizes both teacher and student performances against the original binary sentiment labels (0 for negative, 1 for positive).

---

## Requirements

- **Python 3.7+**
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- **PyTorch** (with CUDA support for GPU acceleration)

---

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. **(Optional) Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate
    ```

3. **Install required packages:**

    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets scikit-learn matplotlib
    ```

    > Adjust the PyTorch extra-index URL if your CUDA version differs.

---

## Usage

1. **Run the Script:**
    
    Run the run.ipynb file

    The script will:
    - **Load** a subset of the IMDB dataset (2,000 training and 1,000 test examples).
    - **Annotate** these reviews using the teacher model with proper truncation and padding to handle long reviews.
    - **Train** the student model for 10 epochs on the teacher-generated data.
    - **Evaluate** both teacher and student performance against the original IMDB ground truth.
    - **Plot**:
        - A bar chart comparing teacher vs. student accuracy.
        - A side-by-side bar chart of label distributions (negative/neutral/positive) on the test set.
        
2. **Results:**
    - Teacher-annotated CSV files are saved in the `data/` folder.
    - The fine-tuned student model is saved in `models/distilled_sentiment/`. but i didnt shared in this repo due to size constraint
    - Plots are saved in `results/figures/` as:
        - `teacher_vs_student_accuracy.png`
        - `teacher_vs_student_label_distribution.png`

---

## Concept Explanation

**Knowledge Distillation** is a technique where a smaller (student) model learns to approximate the behavior of a larger (teacher) model. In this project:
- The **teacher model** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) classifies movie reviews into three sentiment categories: **negative**, **neutral**, and **positive**.
- These predictions are then used as "soft labels" to train the **student model** (`distilbert-base-uncased`), which is more efficient and lightweight.
- The performance of both models is compared against the binary sentiment labels provided in the IMDB dataset (0 for negative, 1 for positive). For evaluation, the teacher’s output is mapped so that both neutral and positive are considered positive.
- Finally, the project visualizes both the overall accuracy and the label distribution from teacher and student models.

---

## Troubleshooting

- **GPU Memory:**  
  If you encounter GPU memory errors, try reducing the `batch_size` in both the teacher pipeline and training arguments.

- **CUDA/GPU Issues:**  
  Ensure that your GPU drivers and CUDA toolkit are properly installed. You can modify the `device` parameter in the pipeline calls if you need to run on CPU.

- **Label Mapping:**  
  The teacher model outputs textual labels ("negative", "neutral", "positive"). Ensure these are correctly mapped to the IMDB binary labels for evaluation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **IMDB Dataset:** Provided by [Hugging Face Datasets](https://huggingface.co/datasets/imdb).
- **Teacher Model:** [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- **Student Model:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased).