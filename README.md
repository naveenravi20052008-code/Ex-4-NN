
<H3>ENTER YOUR NAME:Naveen R</H3> 
<H3>ENTER YOUR REGISTER NO:212225040276</H3> 
<H3>EX. NO.4</H3>
<H3>DATE:25/11/2026</H3>
<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>
<H3>Aim:</H3>
To implement a Multilayer Perceptron for Multi classification
<H3>Theory</H3>

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)

<H3>Algorithm:</H3>

1. Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5. In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6. Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7. In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

<H3>Program:</H3> 

from pathlib import Path
from time import perf_counter
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path("/content/Dry_Bean_Dataset.xlsx")
TARGET = "Class"
TEST_SIZE = 0.20
RANDOM_STATE = 42
HIDDEN_LAYERS = (10, 10, 10)
MAX_ITER = 1000


def evaluate_model(data: pd.DataFrame, feature_columns: list[str], target: pd.Series, class_names: list[str], title: str) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns],
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    # Fit scaling only on training data to avoid leaking test-set statistics.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, y_train)

    fit_seconds = perf_counter() - start
    predictions = model.predict(X_test_scaled)

    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(f"Features: {len(feature_columns)}")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Iterations: {model.n_iter_} / {MAX_ITER} | Fit time: {fit_seconds:.2f} seconds")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print("Confusion matrix (true rows, predicted columns; class order below):")
    print(class_names)
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=model.classes_, target_names=class_names, zero_division=0))

    if caught:
        print("Convergence warning:", str(caught[-1].message))

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "iterations": model.n_iter_,
        "fit_seconds": fit_seconds,
        "converged": not bool(caught),
    }


def main() -> None:
    if DATA_PATH.suffix == ".xlsx":
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    # Select original numeric columns
    original_features = [col for col in df.select_dtypes(include="number").columns if col != TARGET]

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    # Encode the bean names as integer classes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])

    print(f"Rows loaded: {before:,} | exact duplicates removed: {removed:,} | rows used: {len(df):,}")
    print("Class order:", label_encoder.classes_.tolist())

    class_names = label_encoder.classes_.tolist()
    evaluate_model(df, original_features, y, class_names, "Baseline: original numeric features")
	import matplotlib.pyplot as plt
import seaborn as sns

# Plot the confusion matrix using the 'cm' variable from the kernel
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix: Baseline MLP (10, 10, 10)", fontsize=14)
plt.ylabel("Actual Class", fontsize=12)
plt.xlabel("Predicted Class", fontsize=12)
plt.tight_layout()
plt.show()


<H3>Output:</H3>

<img width="723" height="686" alt="Screenshot 2026-11-26 061854" src="https://github.com/user-attachments/assets/c581ab3a-dbe9-4373-b6d7-9175a216b251" />

<img width="1087" height="907" alt="image" src="https://github.com/user-attachments/assets/269436bd-3e8f-4359-9d23-8fba8412a72e" />

<H3>Result:</H3>
Thus, MLP is implemented for multi-classification using python.
