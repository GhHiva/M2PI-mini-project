# FathomNet Underwater Vision Tutorial — All Text Sections (Master Notebook)


## Preread: look up these topics

Before the workshop, skim short explanations or official docs for the items below. They are the vocabulary the notebook assumes you can recognize.

- **computer vision**
- **machine learning**
- **deep learning**
- **supervised learning**
- **image classification**
- **object detection**
- **instance segmentation**
- **segmentation (semantic vs instance)**
- **YOLO & Ultralytics**
- **FathomNet**
- **COCO annotation format**
- **bounding box**
- **polygon & instance mask**
- **logits**
- **softmax**
- **cross-entropy loss**
- **training / validation / test splits**
- **overfitting**
- **learning rate & optimizer**
- **epochs & batch size**
- **confusion matrix**
- **precision & recall**
- **IoU (intersection over union)**
- **mAP, mAP50, mAP50-95**
- **confidence threshold**
- **fine-tuning & pretrained weights**
- **SAM3 & promptable segmentation**
- **Hugging Face (weights & tokens)**
- **GPU & CUDA**
- **Jupyter & Google Colab**
- **pseudo-labels**
- **gradient descent (intuition)**

---

# Underwater **Computer Vision** With **FathomNet** (Master Notebook)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/blob/main/notebooks/fathomnet_underwater_vision_tutorial_master.ipynb)

This notebook introduces **computer vision** workflows for underwater imagery using **FathomNet**-derived data. You will train and compare **image classification**, **object detection**, **instance segmentation**, and promptable **segmentation** workflows.

You only need a mathematical foundation to start. You can follow the guided path from **classification** to **segmentation**, or jump into the section that matches your comfort level.

You will work with images derived from the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/), an expert-annotated underwater image database designed to support marine science and **machine learning**. The problem is simple to state and hard to solve: given an underwater image, decide what organisms or biological structures are present, where they are, and sometimes which pixels belong to each object.

Underwater imagery is not just "ImageNet, but blue." Vehicle lights, **water-column attenuation**, turbidity, motion blur, scale changes, partial animals, transparent bodies, and **long-tailed taxonomy** all make this a useful stress test for **computer vision**.

Reference links:

- **FathomNet** Database: https://database.fathomnet.org/fathomnet/#/
- **FathomNet** data use policy: https://www.fathomnet.org/datause
- **Ultralytics** training guide: https://docs.ultralytics.com/modes/train/
- **Ultralytics** **classification** task: https://docs.ultralytics.com/tasks/classify/
- **Ultralytics** **detection** format: https://docs.ultralytics.com/datasets/detect/
- **Ultralytics** **segmentation** format: https://docs.ultralytics.com/datasets/segment/
- **SAM3** repository: https://github.com/facebookresearch/sam3
- **SAM3** image predictor example: https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb

This instructor copy includes worked code solutions and short answer-key notes. Use the non-master notebook for the live participant version.

---

## Notebook Map

Start with setup and the first image grid so the visual problem is concrete. After that, you can follow the notebook in order or jump to the section that matches your comfort level. Each later section starts with a short **section bootstrap** cell that loads the variables and helpers needed for that section.

What you will do:

- **Setup and first look:** load the compact tutorial bundle, check the runtime, and immediately look at real underwater images.
- **Dataset exploration:** inspect the same imagery through different label views: image-level crop classes, **bounding boxes**, and **segmentation** polygons.
- **Pretrained YOLO warmup:** use an existing **YOLO** model before training anything, so prediction outputs and confidence scores have a concrete meaning.
- **Classification:** train a small crop classifier and connect **softmax**, **cross-entropy**, **learning rate**, and confusion matrices to the observed results.
- **Object detection:** fine-tune **YOLO** on binary organism boxes and study box parameterization, **intersection over union** (IoU), **precision**, **recall**, and **mean average precision** (mAP).
- **Instance segmentation:** fine-tune **YOLO** **segmentation** on binary **masks** and compare box-level and **mask**-level evaluation.
- **SAM3 promptable segmentation:** use **text prompts** such as `"fish"` or `"small crab"` to request **masks**, boxes, and scores, with cached outputs available for every runtime.

The default notebook behavior is:

- use a prebuilt compact data bundle,
- run live **YOLO** training only when a **GPU** is available,
- fall back to cached **training curves** and cached **SAM3**-style outputs when a **GPU**, model weights, or **Hugging Face** access are unavailable.

Vocabulary you will see:

- **YOLO** means "You Only Look Once"; in this notebook it refers to **Ultralytics** **YOLO** models for **classification**, **detection**, and **segmentation**.
- A **bounding box** is a rectangle around an object, usually represented by center point, width, and height.
- **Segmentation** means predicting pixels or regions, not just a class or rectangle.
- **Instance segmentation** means predicting a separate **mask** for each object instance.
- **COCO** means "**Common Objects in Context**"; here it mostly refers to a widely used JSON annotation format for images, categories, **bounding boxes**, and segmentations.
- **SAM3** means "**Segment Anything** Model 3"; in this notebook it refers to Meta's promptable **segmentation** model for detecting and segmenting objects from **text prompts**.

---

### Suggested Instructor Timing

For a three-hour live session, a reasonable pacing target is:

1. `0-15 min`: setup, runtime check, and first look at **FathomNet** imagery.
2. `15-30 min`: **YOLO** warm-up on a familiar image.
3. `30-50 min`: dataset and annotation exploration.
4. `50-75 min`: **classification** from organism crops.
5. `75-115 min`: binary **object detection** with **YOLO**.
6. `115-150 min`: **instance segmentation** with **YOLO**.
7. `150-175 min`: **SAM3** text-prompt **segmentation**.
8. `175-180 min`: wrap-up and extension ideas.

---

## Setup And Runtime Check

First locate the repository, optionally install dependencies, load the small bundle, and inspect the **GPU**.

GitHub repository for this tutorial:

https://github.com/Spiffical/fathomnet-underwater-vision-tutorial

The notebook works in two common modes:

- **Colab from GitHub:** the setup cell clones the repository into `/content/fathomnet_underwater_vision_tutorial`.
- **Local Jupyter:** start **Jupyter** from the cloned repository, or from a child directory inside it.

Data storage model for the session:

- The canonical data bundle is `data/fathomnet_underwater_tutorial_bundle.zip` in the repository.
- For **Colab**, the default `BUNDLE_URL` points to the zip in this GitHub repository.
- In **Colab**, you download your own copy into the temporary runtime, then the notebook extracts it under `data/fathomnet_underwater_tutorial_bundle`.
- Nothing is downloaded live from **FathomNet** during the session.

---

### Dependencies And Runtime

Now check the Python dependencies, inspect the **GPU**/runtime, and set the random seed. In **Colab** this cell installs missing packages; locally it only checks what is already importable.

---

### Download Or Unpack The Tutorial Bundle

This cell prepares the compact **FathomNet**-derived data bundle. In **Colab**, each participant gets a temporary runtime copy. Locally, the notebook uses the zip already stored in the repository when available.

---

## First Look At **FathomNet** Imagery

Before training anything, look at the data. These are full underwater images from the compact **FathomNet**-derived bundle. Some organisms are obvious, some are tiny, and some are visually ambiguous even for a human.

As you scan the grid, ask:

- What would count as an "object"?
- Which organisms are easy to crop and classify?
- Which organisms need a **bounding box**?
- Which organisms would really need a pixel **mask**?

---

### Master Notes: First Look

- Let participants look before naming every task. Good early prompts are: "What is easy to see?", "What is small?", and "What would be hard to annotate consistently?"
- This section intentionally uses full images before crops, boxes, or **masks** so the modeling choices feel motivated by the visual problem.
- If a participant says the image has many unlabeled things, that is a useful opening for incomplete labels and distribution shift.

---

## **YOLO** Warm-Up: Predict Before Training

The official **Ultralytics** **Colab** tutorial starts with the shortest useful workflow: install/check the package, run `predict` on a normal image, then move to validation, training, and export. You can open that reference here:

https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb

This notebook uses the same idea, but keeps the warm-up small. First, load a pretrained **YOLO** model and ask it to predict objects in a familiar image. After that, move to underwater imagery where the pretrained vocabulary and the visual statistics are much less convenient.

The official tutorial currently uses newer **YOLO** model families. This workshop defaults to `yolo11n.pt` and `yolo11n-seg.pt` because they match the prepared underwater experiments and run quickly in free **Colab**. You can swap model names later.

---

### **YOLO** Warm-Up Exercises

Beginner:

- Change `YOLO_WARMUP_CONF` from `0.25` to `0.6`.
- Which detections disappear first?

Intermediate:

- Change `YOLO_WARMUP_SOURCE` to `https://ultralytics.com/images/bus.jpg`.
- Read the printed class names and decide whether the model is solving **classification**, **detection**, or **segmentation**.

Advanced:

- Run the same pretrained model on one underwater validation image before **fine-tuning**.
- What does the model miss, and what does it hallucinate?

---

### Master Notes: **YOLO** Warm-Up

- Raising `YOLO_WARMUP_CONF` from `0.25` to `0.6` should remove lower-confidence detections first; this is a thresholding operation after the model has already scored candidate boxes.
- `bus.jpg` is still **detection**: the model predicts classes and boxes for multiple objects in one image.
- A generic **COCO**-pretrained detector often misses underwater organisms because many taxa are outside its label vocabulary, and it may hallucinate familiar **COCO** classes from shapes or textures.
- This is the same basic pattern as the **Ultralytics** tutorial: `YOLO(weights)`, then `.predict(...)`, then inspect results.

---

## Dataset Exploration

You have now seen the images. Next, inspect how the same underlying problem is represented for different **machine learning** tasks.

The bundle has four useful views:

- `classification_crops`: organism crops grouped by source-level **FathomNet** concept label.
- `yolo_detect_binary`: one-class **object detection** labels.
- `yolo_segment_binary`: one-class **instance segmentation** **polygon** labels.
- `sam3_cached_outputs`: **SAM3**-like prompt outputs for the fallback lab.

The full source **COCO**-style annotation file in the local **YOLO** project has `119,096` images, `280,118` annotations, and `1,897` categories. **COCO** means "**Common Objects in Context**"; the name comes from a benchmark dataset, but people also use "**COCO** format" to mean a JSON annotation schema with image records, category records, and annotation records. This tutorial uses a compact subset so you can work in **Colab** without spending the session on data transfer.

For the live **detection** and **segmentation** exercises, the **YOLO** labels deliberately omit extremely tiny annotations. That keeps the training signal about visible organisms rather than letting dense fragments dominate the classroom metrics. The original **COCO** subset is still included for discussion and after-session extensions.

The images and annotations are **FathomNet**-derived. The bundle includes `licenses/attribution.csv`; visual content is subject to contributor-selected Creative Commons terms and is used here for machine-learning training and development.

---

### View 1: Full Images With Truth Category Labels

This is a full-image, **classification**-style view of the **COCO**-format truth labels. The title above each image lists categories annotated as present in that image. This view hides boxes and **masks** on purpose, so you can separate the question "what is present?" from the question "where is it?"

---

### View 2: **Classification** Crops With Truth Labels

The training classifier uses organism crops grouped by source-level **FathomNet** concept labels. Each crop has one truth label from the folder name. There is still no geometry target here: no center point, no **bounding box**, no **polygon**.

---

### View 3: **YOLO** **Object Detection** Truth Labels

**Object detection** asks for a class and a rectangle for each object. The **YOLO** **detection** rows here have the form

`class_id x_center y_center width height`

where the four geometry values are **normalized** to `[0, 1]` by the image width and height. The examples below are selected because they contain multiple labeled objects.

---

### View 4: **YOLO** **Instance Segmentation** Truth Labels

**Instance segmentation** asks for a separate region for each object. In **YOLO** **segmentation** format, each row starts with the class id and is followed by **normalized** **polygon coordinates**:

`class_id x1 y1 x2 y2 ... xK yK`

The target is more informative than a box, but it is also more expensive to annotate and usually harder to learn.

---

### Annotation Counts And Category Mix

You have now seen the truth labels visually. The next two cells summarize the same bundle numerically: first the **COCO**-style categories and annotation counts, then the object-size distribution used by the **YOLO** **detection** labels.

---

### Object Size Distribution

This histogram shows how large the **detection** boxes are as a fraction of image area. It matters because tiny objects are harder to localize, and the tutorial labels intentionally filter out extremely tiny boxes for the live training exercises.

The bundle uses a **normalized** bounding-box area threshold of `0.005`: an object is kept only if its box covers at least `0.5%` of the image area. For **segmentation** labels, the same idea is applied to the **bounding box** around the **polygon**.

This is a teaching choice, not a scientific law. It makes short training runs easier to interpret because the labels focus on visible organisms. The tradeoff is that the live task becomes "find larger, visually salient objects," not "find every organism in the scene."

---

### Advanced Exercise: Audit The Train/Validation Split

A validation metric is only useful if the validation split is independent and reasonably representative. In this exercise, audit the **YOLO** **detection** split before trusting its mAP.

Implement `audit_yolo_split_exercise(...)` so it reports:

- image counts in `train` and `val`;
- whether any image stems appear in both splits;
- object instances per split;
- median box area per split.

Hints:

- labels live under `labels/train` and `labels/val`;
- image IDs can be compared by filename stem;
- **YOLO** **detection** rows store **normalized** `width` and `height` in columns 4 and 5;
- a suspicious audit result is a modeling problem, not just a data-loading problem.

---

### Master Notes: Split Audit

- The expected overlap between train and validation image stems is zero.
- This audit is intentionally simple: leakage, object counts, and box-size summaries catch many common split problems.
- A representative **validation set** matters more than a polished training curve. If validation differs sharply from training, mAP can mislead.

---

### Dataset Exploration Questions

Beginner:

- Pick one image where the object is visually obvious and one where it is subtle.
- Which source of difficulty is most visible: contrast, scale, clutter, partial view, or taxonomy?

Intermediate:

- Compare the `coco/subset.json` category counts with the binary **YOLO** labels.
- What information is lost when many biological categories are collapsed into `object`?
- Compare the category-only full-image view with the box and **mask** views.
- What extra information do you gain when the truth labels include geometry?

Advanced:

- The live **YOLO** labels drop extremely tiny boxes. Predict how the histogram and mAP would change if those boxes were restored.
- The **segmentation** labels contain more geometry than boxes. Where would that extra information matter scientifically?

---

### Master Notes: Dataset Exploration

- Good obvious examples usually have high contrast, a large organism, and a clean background. Subtle examples often involve small objects, partial bodies, or organism-background camouflage.
- The visual views are intentionally distinct: full images with category labels show "what is present," **classification** crops show one source-level concept label per crop, **detection** labels show multiple boxes per full image, and **segmentation** labels show multiple object-shaped regions per full image.
- Keep this section about truth labels only. Save model predictions for the **YOLO** warm-up and training/evaluation sections.
- Collapsing many categories into `object` removes biological identity, taxonomy, and ecological meaning, but makes a short **detection** exercise easier and more stable.
- Restoring tiny boxes would push the area histogram left, increase the number of labels per image, and likely lower short-run mAP for a small model because localization becomes much harder.
- The **COCO** subset is useful for reasoning about categories and conversion; the prepared **YOLO** folders are useful for short live training.

---

# Part 1: **Classification** From Organism Crops

**Classification** means predicting one label for an input. This is the easiest entry point because each organism crop has one label and no geometric target. The crop labels in this bundle are common **FathomNet** source concepts, so there are enough classes for top-5 accuracy to be meaningful instead of automatically perfect.

Here the image crop is a tensor

$$x \in [0,1]^{H \times W \times 3},$$

and the model produces one score, or logit, for each class. For `K` classes,

$$z=f_\theta(x)\in \mathbb{R}^K.$$

**$y$** is the **ground-truth class label** for this crop: the integer index of the correct category among the `K` classes. One labeled example is the pair $(x,y)$. In this notebook the **cross-entropy** helper uses **zero-based** labels $y\in\{0,\dots,K-1\}$, matching Python; some textbooks write $y\in\{1,\dots,K\}$ for the same idea with indices shifted by one.

The **softmax** turns **logits** into class probabilities,

$$p_k=\frac{\exp(z_k)}{\sum_j \exp(z_j)},$$

and **cross-entropy** penalizes the model when it gives low probability to the true class:

$$\ell(z,y)=-\log p_y.$$

The live exercise is to change one training knob, rerun the small model, and interpret whether validation accuracy moved in a meaningful direction.

---

### Train, Validation, And Test Splits

Before training, define the data splits. Most supervised machine-learning workflows keep separate sets:

- **Training set:** examples used to update model parameters by **gradient descent**.
- **Validation set:** examples used during development to choose **hyperparameters**, compare runs, tune thresholds, and select the best **checkpoint**.
- **Test set:** examples held back until the end for a final estimate of performance on data you did not optimize against.

In symbols, training chooses parameters

$$\hat{\theta}=\arg\min_\theta \frac{1}{n_{\mathrm{train}}}\sum_{i\in \mathrm{train}}\ell(f_\theta(x_i),y_i),$$

while validation estimates whether those parameters are useful away from the data that supplied the gradients. **Ultralytics** saves `weights/best.pt` as the **checkpoint** that performs best on the **validation set** during training. That is usually the **checkpoint** you evaluate or fine-tune from next, while `weights/last.pt` is simply the final epoch.

This compact tutorial bundle uses train/validation splits for live exercises. In a real project, keep a separate test split untouched until your modeling choices are fixed.

---

### Section Bootstrap: Load The Crop Dataset

Start this section by loading the **classification** paths and printing the split/class counts. If you skipped here from above, this bootstrap cell also restores the common setup variables.

---

### Inspect The **Classification** Inputs

Before training, look at one crop from each class. These are the actual inputs for the classifier: one cropped image, one truth label.

---

### Visible Helper: **Cross-Entropy**

This is a small, editable version of the loss behind multi-class **classification**. It is not meant to replace PyTorch; it is here so the math is visible.

---

### Visible Helper: Training Arguments

You are encouraged to edit this function before running a training cell. It is deliberately small: the goal is to connect a few knobs to training behavior.

---

### Train Or Load A Small **Classification** Run

Now move from the hand-computed loss to an actual model. This cell trains `yolo11n-cls.pt` when a **GPU** is available. If not, it loads the cached **classification** curve so the interpretation exercise still works.

Focus on the training arguments: `n_epochs`, `imgsz`, `batch`, `optimizer`, and `lr0`. Those are the knobs you will modify in the exercises. When live training runs, **Ultralytics** saves the validation-best **checkpoint** at `weights/best.pt`.

The first plot compares training loss with validation loss. Training loss is computed on examples that supply gradients; validation loss is computed on held-out examples. If training loss keeps improving while validation loss gets worse, you are seeing **overfitting**.

The second plot reports validation top-1 and top-5 accuracy when those columns are available. Top-1 asks whether the highest-probability class is correct. Top-5 asks whether the truth appears anywhere in the five highest-probability classes, which is useful here because the classifier has more than five possible labels.

---

### Mini-Lab: **Learning Rate** As An Optimization Knob

The **learning rate** controls the scale of each gradient step. In a small workshop run, you usually cannot prove one setting is globally best. You can still learn to recognize three useful patterns:

- too small: the curve barely moves;
- useful: validation accuracy improves without wild instability;
- too large: loss or validation accuracy jumps around or degrades.

Try one or more learning rates from `lr0 = 1e-4`, `1e-3`, and `1e-2`. The default mini-lab uses several epochs because one epoch is often too short to reveal the difference between slow, useful, and unstable optimization. The cached curves give you a baseline comparison, and a live **GPU** run lets you see how much the result changes from one run to the next.

This mini-lab plots validation accuracy only so the comparison across learning rates stays uncluttered. In other cells, `plot_training_curves(..., include_training=True)` can add matching **training curves** for train/validation loss comparisons.

---

### Master Notes: **Learning Rate** Lab

- `1e-4` is likely to look slow in a one-epoch or two-epoch workshop run.
- `1e-3` is a reasonable default for this tiny **classification** **fine-tuning** exercise.
- `1e-2` may improve quickly or become unstable, depending on batch order and initialization.
- The teaching point is not that one run proves the best **learning rate**. It is that optimization behavior is visible in curves, and validation accuracy is a noisy estimate on this small split.

---

### **Confusion Matrix** Reading Practice

The next cell uses a small discussion matrix rather than requiring live predictions. Read it row by row: each row is a true class, and each column is the predicted class. The goal is to connect metric summaries to specific biological or visual confusions.

---

### Advanced Exercise: Build A **Confusion Matrix** From Predictions

The toy matrix above is useful for reading practice. A real **confusion matrix** should come from model predictions on validation images.

Use the **Ultralytics** **classification** prediction docs and `sklearn.metrics.confusion_matrix` to implement `build_confusion_matrix_from_predictions(...)`.

Hints:

- validation crops are stored in `CLASSIFY_ROOT / "val" / class_name`;
- the true class comes from the folder name;
- an **Ultralytics** **classification** result has class probabilities in `result.probs`;
- use the trained `classification_best_model_path` if it exists, otherwise this exercise is best run after live training.

---

### Master Notes: Prediction **Confusion Matrix**

- Keep this advanced exercise gated until a trained **classification** **checkpoint** exists.
- If a generic ImageNet classifier is used instead of the tutorial-trained **checkpoint**, predicted class names may not match the tutorial's **FathomNet** concept folders.
- The useful teaching point is the mapping from validation folder names to true labels and from model outputs to predicted labels.

---

### **Classification** Exercises

Beginner:

- Change `CLASSIFY_N_EPOCHS` from `10` to another value, then rerun the training cell.
- Run or reason through one learning-rate trial.
- Decide whether the validation curve is improving, noisy, or **overfitting**.

Intermediate:

- Use the **Ultralytics** **classification** docs to find the direct `YOLO(...).train(...)` pattern for **classification**.
- Compare one `lr0` result with another `lr0` result.
- Change `imgsz` from `224` to `320`, then decide whether the extra computation was worth it.
- Pick one class and inspect three crops that seem visually ambiguous.

Advanced:

- Design a fair pretrained-vs-random-initialization comparison. What would you keep fixed?
- Complete `build_confusion_matrix_from_predictions(...)` and replace the toy matrix with predictions from your live model.
- Which two classes are most confusable, and what visual ambiguity might explain it?
- Propose a better class grouping for this small crop dataset.

---

### Master Notes: **Classification**

- The direct **Ultralytics** pattern is `from ultralytics import YOLO`, `model = YOLO("yolo11n-cls.pt")`, then `model.train(data=str(CLASSIFY_ROOT), ...)`.
- Increasing `n_epochs` should usually improve the tiny validation run at first, but the curve can be noisy because the **validation set** is intentionally small.
- In a 5-epoch **GPU** sweep with this 12-class crop bundle, `lr0=1e-4` reached about `0.61` top-1, `lr0=1e-3` peaked near `0.69` top-1, and `lr0=1e-2` reached about `0.58` top-1. The exact values can move, but the pattern is useful.
- In the toy **confusion matrix**, adjacent off-diagonal entries are deliberate. Good discussion targets are visual similarity, partial views, label granularity, and the difference between a true biological ambiguity and a model error.

---

# Part 2: Binary **Object Detection** With **YOLO**

**Detection** asks for *where* the animals or biological structures are, not only what crop class they belong to. In this section the target is a **bounding box**, or a rectangle that encloses an object.

Instead of one label per crop, the output is a set of boxes:

$$\{(c_i,b_i,s_i)\}_{i=1}^m,\qquad b_i=(c_x,c_y,w,h).$$

Here `c_i` is the class, `b_i` is a **normalized** box, and `s_i` is the confidence score. **YOLO** stores `c_x`, `c_y`, `w`, and `h` in `[0,1]` relative to image width and height.

The workshop bundle uses a binary **object** target and filters very small boxes. This makes the first detector behave like a useful mathematical object during a short tutorial: you can see IoU, confidence thresholds, **precision**, **recall**, and AP move for interpretable reasons.

During training, the model parameters are chosen by minimizing **empirical risk** on the training split:

$$
\hat{\theta}\in\arg\min_\theta \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i).
$$

The validation split is the quick estimate of whether those parameters are useful outside the examples that supplied the gradients.

For one ground-truth box `A` and one predicted box `B`, the intersection-over-union is

$$\operatorname{IoU}(A,B)=\frac{|A \cap B|}{|A \cup B|}.$$

**Precision** and **recall** depend on a **confidence threshold**:

$$\mathrm{precision}=\frac{TP}{TP+FP}, \qquad \mathrm{recall}=\frac{TP}{TP+FN}.$$

**Average precision**, or **AP**, summarizes the **precision-recall** curve as that threshold changes. **mAP** means **mean average precision**. In **YOLO** reports, `mAP50` uses an IoU threshold of `0.50`, while `mAP50-95` averages over several IoU thresholds from `0.50` to `0.95`, making it a stricter localization metric.

---

### Mini-Lab: Matching Predictions To Labels

**Detection** metrics are built from a matching rule. A predicted box becomes a **true positive** only if it has enough overlap with an unmatched ground-truth box. Everything else is a **false positive**, and every unmatched ground-truth box is a **false negative**.

Use this toy example before looking at **YOLO** outputs. It is small enough to calculate by hand.

---

### Master Notes: Matching Predictions

- At `conf >= 0.25`, the toy example keeps all three predictions: one **true positive** and two **false positives**, so **precision** is `1/3` and **recall** is `1`.
- At `conf >= 0.5`, it keeps the first two predictions: one **true positive** and one **false positive**, so **precision** is `1/2` and **recall** is `1`.
- At `conf >= 0.8`, it keeps only the best prediction: one **true positive**, zero **false positives**, so **precision** is `1` and **recall** is `1`.
- In real multi-object AP, matching is repeated over many objects and thresholds, but this one-object case is the core logic.

---

### Section Bootstrap: Load The **Detection** Dataset

The toy example above shows the metric logic. Now switch to the real **YOLO** **detection** dataset. This bootstrap cell loads the dataset **YAML**, validates image/label pairing, and defines the cached result path used when live training is unavailable.

---

### **Detection** Truth Example: Several Objects In One Image

This plot shows one validation image from the **YOLO** **detection** dataset. The green rectangles are **ground-truth labels**, not model predictions. Each row in the matching `.txt` label file says: class id, **normalized** box center, **normalized** box width, and **normalized** box height.

The example is selected to contain multiple labeled objects so the **detection** task is visually clear.

---

### Train Or Load A Small **Detection** Run

Now we train a binary object detector. The model starts from `yolo11n.pt`, an **Ultralytics** **YOLO** **detection** **checkpoint**, and the dataset comes from `DETECT_YAML`.

If a **GPU** is available, this cell runs a short **fine-tuning** job. Otherwise, it loads a cached training curve. Either way, the output you should inspect is the same: **precision**, **recall**, `mAP50`, and `mAP50-95`. For a live run, the best validation **checkpoint** is saved as `weights/best.pt`.

---

### Mini-Lab: **Confidence Threshold** Sweep

Training gives the model parameters. The **confidence threshold** is a decision rule you choose after the model scores candidate detections.

Sweep a few thresholds on the same underwater image. Your job is to describe the tradeoff in plain language before you look at the metric names:

- low threshold: more detections, more possible **false positives**;
- high threshold: fewer detections, more possible missed objects.

---

### Master Notes: Threshold Sweep

- Lower thresholds tend to increase the number of detections and increase **recall** opportunities.
- Higher thresholds tend to suppress weak boxes, which often improves **precision** but can remove real objects.
- The model weights do not change during this lab. Only the decision rule changes.
- If the generic **COCO** detector finds little in underwater images, that is itself a domain-shift result.

---

### Mini-Lab: Overfit A Tiny Dataset

A powerful debugging move is to ask whether the model can memorize a tiny **training set**. If it cannot overfit 8-12 examples, something may be wrong with the labels, model wiring, optimization settings, or data path.

This cell prepares the tiny dataset for you. Training is off by default because it is a live **GPU** exercise.

---

### Master Notes: Tiny Overfit

- A successful tiny-overfit run should drive training loss down sharply.
- Validation mAP may remain noisy because the **validation set** is also tiny and may contain different-looking objects.
- If training loss does not drop, first suspect data paths, label format, **learning rate**, batch size, or whether the labels actually align with the images.
- This is a debugging test, not a final model-quality test.

---

### Advanced Exercise: Start From **FathomNet** Megalodon

The detector above starts from a general **YOLO** **checkpoint**. A stronger underwater baseline is [FathomNet Megalodon](https://huggingface.co/FathomNet/megalodon), a YOLOv8x object detector fine-tuned by MBARI/**FathomNet** to detect one class: `object`. The model card reports that it was trained from publicly available **FathomNet** localizations and is meant for post-processing underwater images and video.

This is an advanced path because the **checkpoint** is larger than `yolo11n.pt`, and **fine-tuning** it can be slow on a small free **GPU**. The key question is worth the trouble:

> Does a domain-specific **FathomNet** **checkpoint** give you a better starting point than a generic image **checkpoint**?

Useful references:

- **FathomNet** Megalodon model card: https://huggingface.co/FathomNet/megalodon
- **Hugging Face** Hub download docs: https://huggingface.co/docs/huggingface_hub/guides/download
- **Ultralytics** prediction mode: https://docs.ultralytics.com/modes/predict/
- **Ultralytics** training and **fine-tuning** mode: https://docs.ultralytics.com/modes/train/

You will not get the full solution here. The cells below give constants, checks, and hints. Your job is to use the documentation to fill in the missing pieces:

1. download the **checkpoint** from **Hugging Face**;
2. load it as a **YOLO** model;
3. run prediction on `first_detect_image`;
4. optionally fine-tune it on `DETECT_YAML`.

---

### Master Notes: Megalodon

- This should be framed as an advanced optional path. The **checkpoint** is much larger than `yolo11n.pt`, and YOLOv8x **fine-tuning** can be slow on small GPUs.
- The useful comparison is not only metric-vs-metric. Ask whether Megalodon finds underwater object-like regions that generic **COCO** weights miss.
- Keep `n_epochs` modest at first. The first **fine-tuning** question is whether the run is wired correctly and whether early validation behavior looks plausible.
- Because Megalodon is already a one-class **FathomNet** detector, it is a good match for the tutorial's binary `object` labels.
- The participant notebook intentionally gives hints rather than complete code here. For a quick solution, use `hf_hub_download(...)`, then `YOLO(str(path)).predict(...)`, then `YOLO(str(path)).train(data=str(DETECT_YAML), ...)`.

---

#### Prediction Task

After the **checkpoint** path exists, use the **Ultralytics** prediction documentation to run Megalodon on `first_detect_image`. Compare the predicted boxes with the ground-truth boxes from the earlier **detection**-truth cell.

---

#### Optional **Fine-Tuning** Scaffold

**Fine-tuning** means continuing optimization from an existing **checkpoint** instead of starting from generic weights. For this small tutorial dataset, use a small **learning rate** and very few epochs first. Your goal is not to produce a publishable model in one run; your goal is to learn whether the domain-specific starting point changes the early training behavior.

Before running the cell, read the **Ultralytics** training examples and identify:

- where the **checkpoint** path enters `YOLO(...)`;
- where the dataset **YAML** enters `model.train(data=...)`;
- which arguments control image size, batch size, **learning rate**, and epoch count.

---

### Exercise: **COCO** Boxes To **YOLO** Boxes

The bundle already includes **YOLO** labels so you can spend most of the session on modeling. The real conversion step is deliberately left as an exercise.

**COCO** stores a **bounding box** as

```text
[x_min, y_min, width, height]
```

in pixel coordinates. **YOLO** **detection** stores one row as

```text
class_id center_x center_y width height
```

with all four geometry values **normalized** to `[0,1]`.

---

### Mini-Lab: Error Taxonomy

Metrics tell you *how much* error there is. A failure taxonomy tells you *what kind* of error it is. For underwater imagery, common categories are:

- good prediction;
- missed small object;
- poor localization;
- duplicate **detection**;
- object-like background texture;
- ambiguous or incomplete label.

Look at a few predictions and assign one category to each. This is how you turn a metric into a modeling decision.

---

### Master Notes: Error Taxonomy

- The most productive discussion is usually not "the model is bad," but "which failure mode is dominant?"
- Missed small objects suggest resolution, label filtering, or data quantity issues.
- Object-like background **false positives** suggest thresholding, hard-negative examples, or stronger domain-specific pretraining.
- Poor localization suggests label geometry, image size, or training duration.

---

### **Detection** Exercises

Beginner:

- Use the **Ultralytics** training guide to identify the two lines that load a model and start training.
- Run the **confidence threshold** sweep and describe what changes from `0.10` to `0.80`.
- Complete the toy matching exercise: what is TP, FP, and FN at each **confidence threshold**?

Intermediate:

- Change `imgsz` from `320` to `416`.
- Change `DETECT_N_EPOCHS`, `lr0`, or `batch`, then compare `mAP50` and **recall**.
- Complete `coco_bbox_to_yolo_exercise(...)` above.
- Use the error taxonomy on two predicted images.

Advanced:

- Turn on `RUN_TINY_OVERFIT_LAB` and check whether the model can memorize 8 training images.
- Turn on the Megalodon advanced path, compare its predictions with `yolo11n.pt`, and decide whether the domain-specific **checkpoint** changes the failure modes.
- Fine-tune Megalodon for one or two epochs, then compare its early metrics with the generic **checkpoint** run.
- Explain how the **precision-recall** curve would move if the model became more conservative.
- Sketch how you would convert the whole `coco/subset.json` file into **YOLO** **detection** labels.

---

### Master Notes: **Detection**

- The two essential training lines are `model = YOLO("yolo11n.pt")` and `model.train(data=str(DETECT_YAML), **DETECT_ARGS)`.
- **COCO** `[x_min, y_min, width, height]` maps to **YOLO** `(x_min + width/2)/W`, `(y_min + height/2)/H`, `width/W`, `height/H`.
- A more conservative **confidence threshold** usually raises **precision** and lowers **recall**.
- Raising `imgsz` can help small objects because the resized image preserves more spatial detail, but it costs more memory and time.
- A whole-file **COCO** converter needs to group annotations by image, map category ids to contiguous **YOLO** class ids, normalize geometry by each image size, skip invalid boxes, and write one `.txt` label file per image.

---

# Part 3: **Instance Segmentation** With **YOLO**

**Segmentation** means predicting regions or pixels. **Instance segmentation** upgrades a rectangle to a separate shape for each object instance. A **mask** can be viewed as a set of pixels

$$M \subseteq \Omega,\qquad \Omega=\{1,\dots,H\}\times\{1,\dots,W\}.$$

For multiple objects, the model predicts a set of class-**mask**-score triples:

$$\{(c_i,M_i,s_i)\}_{i=1}^m.$$

As in the **detection** section, the live labels focus on larger visible instances. The omitted tiny polygons are a useful reminder that label design is part of the modeling problem, not a boring preprocessing footnote.

The same IoU idea applies:

$$\operatorname{maskIoU}(M,N)=\frac{|M \cap N|}{|M \cup N|}.$$

In the **YOLO** **segmentation** text format, one row is

```text
class x1 y1 x2 y2 ... xn yn
```

where **polygon coordinates** are **normalized** to `[0,1]`.

---

### Section Bootstrap: Load The **Segmentation** Dataset

Start the **segmentation** section by loading the **YOLO** **segmentation** **YAML** and validating its labels. This is the same pattern as **detection**, but each label row now stores a **polygon** instead of a rectangle.

---

### **Segmentation** Truth Example: Same Image, More Geometry

This plot shows **ground-truth polygon labels** from the **YOLO** **segmentation** dataset. When possible, it uses the same image as the **detection** example above so you can compare boxes with **masks** directly. The colored regions are annotation polygons, not model predictions.

---

### Visible Helper: **Mask** IoU

This helper is intentionally simple. It assumes binary **masks** with equal shape. You can modify it to handle soft **masks**, ignored pixels, or batches.

---

### Advanced Exercise: **Rasterize** A **YOLO** **Polygon**

**YOLO** **segmentation** labels store polygons, but **mask** IoU is defined on pixels. To connect the two, **rasterize** one **normalized** **polygon** row into a binary **mask**.

Implement `yolo_polygon_row_to_mask(...)`.

Hints:

- a **YOLO** **segmentation** row is `class_id x1 y1 x2 y2 ...`;
- multiply **normalized** x coordinates by image width and y coordinates by image height;
- `PIL.ImageDraw.Draw(...).polygon(...)` can fill a **polygon**;
- the output can be a NumPy boolean array or nested list of `0/1` values.

---

### Master Notes: **Rasterization**

- This exercise connects the set notation for **masks** to the **polygon** text file format.
- **Rasterization** introduces pixel-grid choices: rounding, boundary inclusion, and holes can all matter in production.
- Comparing **mask** area to bounding-box area helps explain why **mask** metrics and box metrics need not move together.

---

### Train Or Load A Small **Segmentation** Run

Now train the **segmentation** model, or load cached curves if live training is unavailable. The important difference from **detection** is that the report contains both box metrics and **mask** metrics. Compare them: a model can learn reasonable boxes before it learns precise **masks**. For a live run, the best validation **checkpoint** is saved as `weights/best.pt`.

---

### Exercise: **COCO** Polygons To **YOLO** **Segmentation** Rows

**COCO** **segmentation** annotations often store **polygon** vertices as one flat list:

```text
[x1, y1, x2, y2, ..., xn, yn]
```

**YOLO** **segmentation** wants the same geometry **normalized** by image width and height, preceded by the class id.

You do not get the full converter here. The point is to reason through the coordinate map first, then decide what edge cases a production converter needs to handle: multiple polygons, holes, tiny regions, invalid polygons, and categories you might merge.

---

### **Segmentation** Exercises

Beginner:

- Inspect two **masks**: one clean and one messy.
- Why can box mAP be higher than **mask** mAP?
- Use the **Ultralytics** **segmentation** docs to identify the minimum number of **polygon** points per object.

Intermediate:

- Change the **confidence threshold** and inspect the visual result.
- Complete `mask_iou(...)` for one extra edge case: two empty **masks**, two disjoint **masks**, or two identical **masks**.
- Complete `coco_polygon_to_yolo_row_exercise(...)` above.

Advanced:

- Complete `yolo_polygon_row_to_mask(...)` and compare the rasterized **mask** area with the **polygon**'s bounding-box area.
- Use the coarse biological label plan from the source **YOLO** repo as an after-session extension.
- Ask whether binary "object" **segmentation** is a scientifically useful target, or only a stepping stone.
- Propose a rule for dropping or keeping tiny polygons, then predict how that rule will affect **recall** and mAP.

---

### Master Notes: **Segmentation**

- **YOLO** **segmentation** requires at least three `(x, y)` **polygon** points per object.
- Box mAP can be higher than **mask** mAP because a predicted rectangle can be acceptable even when the boundary shape is poor.
- **COCO** **polygon coordinates** are **normalized** pairwise: `x/W`, `y/H`, preceded by the **YOLO** class id.
- Dropping tiny polygons often improves short workshop metrics by reducing ambiguous labels, but it can reduce ecological **recall** for small organisms.
- Binary object **segmentation** is useful as a stepping stone and for generic saliency, but it is less scientifically expressive than taxonomic or functional-group labels.

---

# Part 4: **SAM3** Text-Prompt **Segmentation** Lab

**SAM3** stands for **Segment Anything Model 3**. It is Meta's promptable **segmentation** **foundation model** for images and videos. "Promptable" means the model does not only take an image as input; it also takes a prompt that tells the model what kind of object or region you want. **SAM3** can use **text prompts**, visual prompts such as points and boxes, and exemplar prompts. In this notebook you will focus on **text prompts**.

**SAM3** changes the interface compared with the supervised **YOLO** models above. Instead of training a new model for every label set, you can ask for a concept:

```text
image + "fish" -> masks, boxes, scores
```

Mathematically, the text prompt becomes part of the input:

$$g_\phi(x,q)\rightarrow \{(M_i,b_i,s_i)\}_{i=1}^m.$$

That changes the main question from "how do you train a new supervised head?" to "which prompt and threshold define the object concept you actually want?"

This is useful for rapid exploration and for proposing **pseudo-labels**, but it is not magic. A prompt like `"fish"` may miss small or unusual fish; a broad prompt like `"marine organism"` may include objects that are not useful for your scientific question. You will inspect those failures directly.

The default path below uses cached **SAM3**-style outputs so everyone can complete the lab. If your runtime passes the live **SAM3** checks and you have **checkpoint** access, install **SAM3**, enter a **Hugging Face** token, and let the notebook set `USE_LIVE_SAM3 = True`.

---

### Section Bootstrap: Load **SAM3** Cache And Runtime Checks

This cell checks whether live **SAM3** is possible and lists the cached prompt outputs available in the tutorial bundle. The rest of the section works even when live **SAM3** is unavailable.

---

### Optional: Install **SAM3** For Live **Inference**

The cached **SAM3** path works without installing **SAM3**. Live **SAM3** needs the official **SAM3** package, Python 3.12 or newer, PyTorch 2.7 or newer, a **CUDA**-compatible **GPU** with **CUDA** 12.6 or newer, and **checkpoint** access. The bootstrap status above tells you which checks are failing. If `sam3_importable` is `false`, **SAM3** is not installed in the current runtime.

Set `INSTALL_SAM3_FOR_LIVE = True` only if you want to try live **SAM3**. This can take several minutes in **Colab**. The install source is the official **SAM3** GitHub repository: https://github.com/facebookresearch/sam3

---

### Optional: **Hugging Face** Token For Live **SAM3**

Live **SAM3** **checkpoint** loading may require a **Hugging Face** account, **checkpoint** access, and a token. The cached path above does not need a token.

To create a token:

1. Sign in to **Hugging Face**.
2. Open https://huggingface.co/settings/tokens.
3. Click **New token**.
4. Choose a `read` token, or a fine-grained token with read access to the **SAM3** model/**checkpoint** you plan to use.
5. Copy the token and paste it into the hidden input when you run the cell below.

Official **Hugging Face** token documentation: https://huggingface.co/docs/hub/security-tokens

Do not paste a real token into a saved notebook cell, markdown cell, chat, or GitHub issue. The cell below uses hidden input and stores the token only in this runtime's environment variables.

---

### Visible Helper: Try A **SAM3** Prompt

Edit `prompt` and `confidence_threshold`. The same function works with cached fallback outputs or live **SAM3**.

---

### Compare Several Prompts

After one prompt works, try several prompts on the same cached image. This cell does not train anything; it only changes the text query and reports how many **masks** were returned for each concept.

---

### Advanced Exercise: Export **SAM3** **Pseudo-Labels**

Promptable **segmentation** can be used to create **pseudo-labels**: labels produced by a model instead of a human. This can speed up annotation, but it can also copy the prompt model's mistakes into your supervised **training set**.

Implement `sam3_result_to_yolo_pseudo_labels(...)` so it converts cached **SAM3**-style polygons into **YOLO** **segmentation** rows.

Hints:

- `sam3_result["polygons"]` is a list of polygons;
- `sam3_result["scores"]` is aligned with those polygons;
- the cached polygons are already **normalized** to `[0, 1]`;
- filter out **masks** below `score_threshold`;
- each output row should look like `[class_id, x1, y1, x2, y2, ...]`.

---

### Master Notes: **SAM3** **Pseudo-Labels**

- The cached polygons are **normalized**, so the conversion to **YOLO** rows is mostly filtering and flattening.
- The hard part is not the file format; it is deciding whether prompt-generated labels are trustworthy enough to train on.
- Good discussion prompt: which **pseudo-label** errors would a supervised model amplify?

---

### **SAM3** Exercises

Beginner:

- Try `PROMPT = "fish"`, `"sponge"`, `"gelatinous animal"`, and `"small crab"`.
- Increase `CONFIDENCE_THRESHOLD` from `0.5` to `0.8`.
- Which prompts are too broad? Which are too specific?

Advanced:

- If live **SAM3** works, compare cached fallback with live **masks**.
- Complete `sam3_result_to_yolo_pseudo_labels(...)`, then reason about when **pseudo-label** noise helps or hurts supervised training.

---

### Master Notes: **SAM3**

- Broad prompts such as `marine organism` tend to increase **recall** and **false positives**.
- Specific prompts such as `small crab` can improve **precision** when the concept is visually present, but often return nothing when the image lacks the target or the prompt misses the model's visual vocabulary.
- Raising `CONFIDENCE_THRESHOLD` filters low-score **masks**; expect fewer detections, higher apparent **precision**, and lower **recall**.
- **SAM3** **pseudo-labels** are useful when they reduce annotation effort, but noisy **pseudo-labels** can teach a supervised model the prompt model's mistakes.

---

# Wrap-Up

You moved through four levels of supervision and geometry:

- **classification**: one image crop, one class;
- **detection**: objects as **normalized** boxes;
- **segmentation**: objects as **masks** or polygons;
- **SAM3** prompting: **segmentation** conditioned on text.

Good after-session projects:

- train on the 500-image or full dataset after this notebook,
- convert the coarse biological label plan into a multiclass **detection** task,
- compare **YOLO** training against **FathomNet** pretrained checkpoints,
- use **SAM3** prompt outputs to propose labels for active learning,
- extend from images to video tracking.

The main lesson is not that one model wins everywhere. The real modeling choice is the pair:

$$\text{scientific question} \quad + \quad \text{label geometry}.$$

Once that pair is clear, the model choice becomes much easier to reason about.
