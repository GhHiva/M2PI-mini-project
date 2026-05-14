
# What is Deep Learning?

Deep learning is a branch of machine learning that uses **neural networks with many layers** to learn patterns from data.

The word **“deep”** refers to the large number of layers between the input and the output.

![NN](image/NN.png)
---

# Simple Intuition

A deep learning model tries to learn a function:

$$
f(x) \approx y
$$

where:

* $x$ = input data
* $y$ = desired output

Here, we aim to minimize the error between the actual value and the predicted (approximated) value.

---

# Why Is It Called a Neural Network?

Neural networks were inspired by biological neurons.

A single artificial neuron performs:

$$
y = f(w^Tx+b)
$$

where:

* $x$ = input
* $w^T$ = weight
* $b$ = bias
* $f$ = activation function

This is essentially:

1. a linear transformation
2. followed by a nonlinear function

![NNM1](image/NNM1.png)
---

# Mathematical Perspective

Deep learning is mainly built from:

| Mathematics    | Role                        |
| -------------- | --------------------------- |
| Linear Algebra | matrix operations           |
| Calculus       | gradients and optimization  |
| Probability    | uncertainty and predictions |
| Optimization   | minimizing loss             |
| Statistics     | learning from data          |

---

# Core Idea of Training

The model starts with random parameters.

It predicts:

\[
\hat{y}=f(x;\theta)
\]

where:

* (\theta) = parameters (weights)

The prediction error is measured using a [loss function](https://www.ibm.com/think/topics/loss-function):

\[
L(\theta)
\]

Then optimization updates parameters using [gradient descent](https://www.ibm.com/think/topics/gradient-descent):

\[
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)
\]

The network gradually improves through repeated updates.

---

# Examples of Deep Learning Applications

* Object detection (YOLO)
* ChatGPT and language models
* Face recognition
* Medical imaging
* Self-driving cars
* Speech recognition
* Recommendation systems

---

## Connection to YOLO

YOLO is a deep learning model specialized for:

\[
\textbf{Object Detection}
\]

It uses deep neural networks to:

* understand image features
* detect objects
* predict locations
* classify objects simultaneously

---

# Final Intuition

Deep learning can be viewed as:

> A large mathematical function that learns patterns from data through optimization.


# Segmentation Concepts for YOLO and Computer Vision

This section introduces important segmentation techniques that are commonly used in modern computer vision systems. These methods help models not only detect objects with bounding boxes, but also understand the exact shape and region of objects inside an image.

In underwater computer vision tasks, segmentation is extremely useful because marine environments are visually complex. Objects may overlap, blend into the background, or have unclear boundaries. Segmentation helps models understand images at a much deeper level.

---

# 1. What is Segmentation?

Before discussing different segmentation methods.

Object detection answers:

* *What is the object?*
* *Where is the object approximately?*

Segmentation goes one step further and answers:

> “Which exact pixels belong to the object?”

Instead of drawing only a rectangle around an object, segmentation attempts to identify the precise region occupied by that object.

---

## 2. Semantic Segmentation

### Main Idea

Semantic segmentation classifies **every pixel** in an image into a category. Semantic segmentation does **not** distinguish between separate objects of the same class.

Example:

* 5 fish in an image
* all fish pixels receive the same label

The model knows:

* “these pixels belong to person”

but not:

* “this is person #1 and this is person #2”

---

## Mathematical View

Semantic segmentation is essentially:

\[
f(x,y) \rightarrow c
\]

where:

* $x,y$ = pixel location
* $c$ = class label

The model performs pixel-wise classification.

---

## Applications

* Medical imaging
* Satellite analysis
* Road detection
* Underwater habitat analysis


![ss](image/semantic segmentation.png)
---

## 3. Instance Segmentation

### Main Idea

Instance segmentation combines:

* object detection
* semantic segmentation

It not only classifies pixels, but also separates individual objects.

---

## Example

Suppose an image contains:

* 3 people

Semantic segmentation:

* labels all person pixels together

Instance segmentation:

* separates person #1, person #2, and person #3

Each object receives its own mask. This is matter when:

* objects overlap
* multiple similar objects exist
* counting objects is necessary

![is](image/instance segmentation.png)

---

# 4. Binary Masks

## Main Idea

A binary mask is the simplest form of segmentation.

Each pixel can only take two values:

[
0 \quad \text{or} \quad 1
]

where:

* 1 → object pixel
* 0 → background pixel

---

## Example

A fish mask may look like:

[
\begin{bmatrix}
0 & 0 & 1 & 1 \
0 & 1 & 1 & 1 \
0 & 0 & 1 & 0
\end{bmatrix}
]

This means:

* white region → fish
* black region → background

---

## Why Binary Masks Are Useful

Binary masks:

* simplify segmentation tasks
* isolate objects
* improve training quality
* help compute object boundaries

---

## Geometry Connection

A binary mask can be interpreted as:

* a region indicator function
* a discrete geometric shape

---

# 5. Soft Masks

## Main Idea

Soft masks extend binary masks.

Instead of only:

* 0 or 1

pixels can take continuous values:

[
0 \le p \le 1
]

representing confidence levels.

---

## Example

[
\begin{bmatrix}
0.1 & 0.2 & 0.8 \
0.3 & 0.9 & 0.95 \
0.1 & 0.7 & 0.2
\end{bmatrix}
]

Interpretation:

* 0.95 → very likely object pixel
* 0.1 → likely background

---

## Why Soft Masks Matter

Soft masks:

* preserve uncertainty
* provide smoother boundaries
* improve gradient flow during training

They are often used internally during neural network optimization.

---

## Probability Connection

Soft masks are closely related to probability theory.

Each pixel prediction can be viewed as:

[
P(\text{object at pixel})
]

---

# 6. Promptable Segmentation (SAM / SAM2 / SAM3)

## Main Idea

Promptable segmentation means:

> The model performs segmentation based on a user prompt.

Instead of segmenting everything automatically, the user provides guidance.

---

# What is SAM?

SAM stands for:

[
\textbf{Segment Anything Model}
]

Developed for highly flexible segmentation tasks.

Modern versions include:

* SAM
* SAM2
* SAM3

---

## Types of Prompts

The user can provide:

* a point
* a bounding box
* text
* clicks
* rough regions

The model then predicts the segmentation mask.

---

## Example

Suppose the user clicks on a fish.

SAM identifies:

* the entire fish region
* object boundaries
* precise mask

without manual annotation.

---

# Why SAM Is Powerful

Traditional segmentation models:

* require fixed training objectives

Promptable segmentation:

* adapts dynamically
* supports interactive annotation
* accelerates dataset labeling

---

# Connection to YOLO

YOLO mainly predicts:

* bounding boxes

SAM-style models predict:

* detailed masks

Together they can improve:

* annotation quality
* localization accuracy
* fine-grained understanding

---

# 7. Relationship Between Detection and Segmentation

| Task                    | Output                  |
| ----------------------- | ----------------------- |
| Classification          | class label             |
| Object Detection        | bounding box            |
| Semantic Segmentation   | class for every pixel   |
| Instance Segmentation   | separate object masks   |
| Promptable Segmentation | masks guided by prompts |

---

# 8. Why These Techniques Matter for Underwater Vision

Underwater images are difficult because of:

* low visibility
* lighting distortion
* overlapping objects
* camouflage
* blurry boundaries

Segmentation techniques help models:

* isolate marine creatures
* separate background from objects
* improve object localization
* enhance training quality

---

# 9. Mathematical Connections

These techniques connect strongly to mathematics.

| Mathematics    | Connection                    |
| -------------- | ----------------------------- |
| Linear Algebra | image tensors                 |
| Geometry       | object regions and boundaries |
| Probability    | soft masks and confidence     |
| Optimization   | mask prediction learning      |
| Calculus       | backpropagation               |
| Set Theory     | overlap regions and masks     |

---

# 10. Final Intuition for Students

Bounding boxes are only a rough approximation of objects.

Segmentation allows the model to move from:

> “The object is somewhere here”

to:

> “These exact pixels belong to the object.”

This is one of the major steps toward more human-like visual understanding in modern AI systems.

