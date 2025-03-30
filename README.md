# cot-4500-as3b

This repository contains solutions for Assignment 3b, covering basic numerical methods:

1. **Gaussian Elimination** and **Backward Substitution**
2. **LU Factorization** (without using `scipy`)
3. **Check Diagonal Dominance**
4. **Check Positive Definiteness**

## Repository Structure

```bash
cOT-450d-a3b/
├─ src/
│ ├─ main/
│ │ └─ init.py
│ │ └─ assignment_3.py
│ └─ test/
│ └─ init.py
│ └─ test_assignment_3.py
├─ requirements.txt
└─ README.md
```

- `assignment_3.py` implements the core functions and a `main()` method demonstrating them.
- `test_assignment_3.py` contains unit tests using `unittest`.

## Requirements

- Python 3.x
- No external libraries are strictly required beyond the Python standard library.
- If you do decide to use `numpy`, you could add `numpy` to the `requirements.txt` file.

Example `requirements.txt` (if you choose to use numpy):

## How to Run

1. **Clone or download** this repository.
2. Navigate to the root folder `cot-4500-as3b`.
3. (Optional) Install dependencies:

```bash
    pip install -r requirements.txt
```

4. Run the main script:

```bash
    python3 src/main/assignment_3.py
```

5. Run tests:

```bash
    python3 -m unittest src.test.test_assignment_3
```
