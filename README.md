# Detecting User Actions from Mouse Events

## Introduction

This project aims to classify a user's current activity (e.g., web browsing, chatting, watching videos, reading) based solely on their mouse events. This approach enhances productivity by enabling features like sharing status updates on platforms like Slack or activating Focus mode, while preserving user privacy by avoiding more invasive monitoring methods such as screen captures or keystroke logging.

---

## Sections

### 1. Introduction/Background

- **Literature Review:** Relevant research on activity detection through non-invasive means such as mouse events.
- **Dataset Description:** The dataset will contain mouse event data (e.g., clicks, x-y coordinates, movement speed, scrolling events) in a time-series of 0.1sec interval.

### 2. Problem Definition

- **Problem:** Automatically identifying user activity based on mouse events, which avoids privacy concerns.
- **Motivation:** Enhancing user productivity and privacy by providing non-invasive activity recognition.

### 3. Methods

- **Data Preprocessing Methods Identified:** 
  - Sliding window technique for managing time-series data of irregular duration.
  - Feature engineering to extract additional insights like scroll patterns, click patterns, and idle times.
- **Machine Learning Algorithms/Models Identified:** 
  - Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Temporal Convolutional Networks (TCNs) for time-series classification.
  - Data augmentation with synthetic data generation to address small dataset issues.
- **Relevant Courses and Methods:**
  - CS 7641: Unsupervised and Supervised Learning, focusing on deep learning approaches for time-series data.

### 4. (Potential) Results and Discussion

- **Quantitative Metrics:**
  - Accuracy, precision, and recall for evaluating classification performance.
- **Project Goals:**
  - Ensure high privacy standards while maintaining accurate user activity detection.
- **Expected Results:**
  - A model capable of distinguishing user activity with high accuracy while preserving privacy by using non-invasive data collection methods.

### 5. References

- TODO

---

## Gantt Chart

Below is the Gantt chart outlining each group member’s responsibilities for the project.

![Gantt Chart](https://docs.google.com/spreadsheets/d/14TtwuTkYRx8cqvmaVrm9Yi3lG_yQL1HBeeJYwnyOWzk/edit?usp=sharing)

---

## Contribution Table

Each group member's specific contributions to the project proposal are outlined in the table below.

| Name           | Proposal Contributions                                 |
|----------------|--------------------------------------------------------|
| Ji Min Park    | Dataset preparation, Data preprocessing methods (e.g., sliding window technique) |
| Hyunju Ji      | Model selection (Boosting, LSTM, TCN), Designing experiments, Initial model training |
| Woohyun Noh    | Literature review on non-invasive user activity detection, Documentation of results and discussions |
| Jungwoo Park   | Feature engineering (click patterns, scroll patterns, idle times), Data augmentation |
| Minsuk Chang   | Evaluation metrics (accuracy, precision, recall), Writing report and final presentation |


---

## Video Presentation

- **Link to 3-minute YouTube Unlisted video:** [YouTube Link]()

This video provides a concise summary of our proposal, explaining the problem, methods, and expected outcomes.

---

## Repository Setup

This is a private repository created using GT GitHub Enterprise or regular GitHub. All group members and the assigned mentor are collaborators. The repository contains the proposal, resources, and related documentation.

---

## Conclusion

Thank you for reviewing our project proposal. We are excited to continue developing our methods and aim to deliver impactful results.