# Detecting User Actions from Mouse Events

## Introduction

This project aims to classify a user's current activity (e.g., web browsing, chatting, watching videos, reading) based solely on their mouse events. This approach enhances productivity by enabling features like sharing status updates on platforms like Slack or activating Focus mode, while preserving user privacy by avoiding more invasive monitoring methods such as screen captures or keystroke logging.  

### Related Work
Previous research has attempted to identify individuals based on their mouse usage pattern [3] or predict the next mouse event of the user [2]. However, we are approaching the problem by classifying the current user's high-level activity in real time. Kuric et al. [1] have suggested several features (such as clicks, velocity, acceleration, etc.) that can be used to classify current actions, which can support our project.be used to classify current action, which can support our project.  

### Dataset Description
We aim to collect 5 to 10 min action recordings from 5 participants. The dataset will be split by frames and recorded in 0.1 sec per frame with corresponding mouse events. For better convergence and higher performance of our model, we will use MacBook Pro/Air for the data collection, which has a uniform device (Track Pad) that records the mouse events.

---

### Problem Definition

- **Problem:** Automatically identifying user activity purely based on mouse events, which avoids privacy concerns.
- **Motivation:** Enhancing user productivity and privacy by providing non-invasive activity recognition.

### Methods

- **Data Preprocessing Methods Identified:** 
  - Regularizing the x, y coordinates based on the individual screen size. 
  - Forming continuous event stream into a fixed timeslot(about 0.1 sec) for more structurized time series.
  - Feature engineering to extract additional insights like scroll patterns, click patterns, idle times, or mouse velocity.
- **Machine Learning Algorithms/Models Identified:** 
  1. Long Short-Term Memory (LSTM), and Temporal Convolutional Networks (TCNs) for time-series classification.
  2. Converting time-series into windows and run boosting or tree-based algorithms
  3. Time series analysis algorithms (ARIMA, ETS) for exploring individual features in the data.
- **Relevant Courses and Methods:**
  - CS 7641: Unsupervised and Supervised Learning, focusing on machine learning approaches for plain data.
  - CS 7643: Deep learning models and algorithms including time-series.

### Results and Discussion

- **Quantitative Metrics:**
  - Accuracy, precision/recall, and f1-score for evaluating classification performance. For each window, 
- **Project Goals:**
  - Ensure high privacy standards while maintaining accurate user activity detection.
- **Expected Results:**
  - A model capable of distinguishing user activity with high accuracy while preserving privacy by using non-invasive data collection methods.

### References

- [1] E. Kuric, P. Demcak, M. Krajcovic, and P. Nemcek, “Is mouse dynamics information credible for user behavior research? An empirical investigation,” Computer Standards & Interfaces, vol. 90, p. 103849, 2024.
- [2] E. Y. Fu, T. C. K. Kwok, E. Y. Wu, H. V. Leong, G. Ngai, and S. C. F. Chan, “Your mouse reveals your next activity: towards predicting user intention from mouse interaction,” in 2017 IEEE 41st Annual Computer Software and Applications Conference (COMPSAC), vol. 1, pp. 869-874, 2017.
- [3] J. J. Matthiesen and U. Brefeld, “Assessing user behavior by mouse movements,” in HCI International 2020-Posters: 22nd International Conference, HCII 2020, Copenhagen, Denmark, July 19-24, 2020, Proceedings, Part I, pp. 68-75, 2020.

---

### Gantt Chart

Below is the Gantt chart outlining each group member’s responsibilities for the project.

![Gantt Chart](https://docs.google.com/spreadsheets/d/14TtwuTkYRx8cqvmaVrm9Yi3lG_yQL1HBeeJYwnyOWzk/edit?usp=sharing)

---

### Contribution Table

Each group member's specific contributions to the project proposal are outlined in the table below.

| Name           | Proposal Contributions                                 |
|----------------|--------------------------------------------------------|
| Ji Min Park    | Dataset preparation, Data preprocessing methods (e.g., sliding window technique) |
| Hyunju Ji      | Model selection (Boosting, LSTM, TCN), Designing experiments, Initial model training |
| Woohyun Noh    | Literature review on non-invasive user activity detection, Documentation of results and discussions |
| Jungwoo Park   | Feature engineering (click patterns, scroll patterns, idle times), Data augmentation |
| Minsuk Chang   | Evaluation metrics (accuracy, precision, recall), Writing report and final presentation |
