 1. Project Overview

This repository hosts the implementation for the Market Mood & Moves project. It contains the complete codebase that operationalises the theoretical ideas discussed in the end-term report. The project demonstrates how sentiment signals derived from financial text can be combined with historical stock market data to model and analyse price behaviour.

The primary objective of this repository is to present an end-to-end implementation pipeline, covering data acquisition, sentiment extraction, sequential modelling, and performance evaluation. All theoretical discussions, mathematical formulations, and conceptual reasoning are documented separately in the accompanying report. This repository is dedicated solely to practical implementation.

 2. Project Structure

The codebase is organised in a modular fashion, mirroring the sequential stages of the project workflow:

Data Collection and Preprocessing

Scripts and notebooks for acquiring financial news and historical stock prices

Utilities for data cleaning, temporal alignment, and preprocessing

Sentiment Analysis Module

Implementation of sentiment extraction using a pre-trained FinBERT model

Transformation of unstructured financial text into numerical sentiment representations

Feature Engineering

Creation of time-series features from price data

Alignment and aggregation of sentiment scores with corresponding market data

Time-Series Modelling

Sequence models designed to learn temporal patterns in financial data

Integration of sentiment features with numerical market variables as multimodal inputs

Training and Evaluation

Model training procedures

Performance assessment and result visualisation

Each module is developed to function independently while remaining fully compatible with the rest of the pipeline, enabling seamless sequential execution.

 3. Component Integration

The project adopts a pipeline-based architecture, where each stage builds upon the output of the previous one:

Financial news articles and stock price data are collected and preprocessed

Sentiment scores are extracted from textual data using FinBERT

Sentiment information is temporally aligned with historical market data

The combined feature set is passed to a time-series prediction model

Model predictions are evaluated using appropriate performance metrics

This design reflects the integration of ideas from behavioural finance, natural language processing, and sequential learning, as outlined in the end-term report.

 4. Execution Guidelines

Code files should be executed in the order of the defined pipeline stages

Jupyter Notebooks are used for step-wise execution and visual analysis

Python scripts support modularity and code reuse

Required library imports are specified within individual files

No code modifications are necessary for evaluation, assuming all dependencies are correctly installed

 5. Dependencies and Environment

The project is implemented in Python and relies on commonly used data science and machine learning libraries, including:

Libraries for numerical computation and data manipulation

Natural Language Processing frameworks

Deep learning libraries for sequence modelling

Specific library versions and installation instructions are provided within the relevant scripts or notebooks where applicable.

 6. Notes for Evaluation

Detailed theoretical explanations of models, methods, and formulas are available in the end-term report

This repository represents the practical realisation of those concepts

The code prioritises clarity, readability, and modular design over production-level optimisation

Generated outputs and visualisations are intended to aid interpretation and evaluation of model behaviour

 7. Conclusion

This repository completes the Market Mood & Moves project by converting theoretical ideas into a functional implementation. When viewed alongside the end-term report, it demonstrates a comprehensive understanding of sentiment-driven financial modelling, spanning data collection, model development, and evaluation.
