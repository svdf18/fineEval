# LLM Model Evaluator

A tool for evaluating and comparing the performance of different language models, with visualization capabilities.

## Overview

This project provides a framework to:
- Run comparative evaluations of different Local LLM models through the llama.cpp API
- Measure response times, lengths and quality
- Generate visualizations of the results
- Support analysis of model performance across different question types

## Dependencies

- reqquirements.txt
- spacy en_core_web_sm

## Output Files

The project generates several output files:

### evaluation_results.csv
Contains raw evaluation data including:
- Model name
- Prompts tested
- Model responses
- Response times
- Response lengths

### Visualization Files
Generated by visualize_results.py:
- response_times.png - Box plots comparing response times across models
- response_lengths.png - Box plots comparing response lengths
- time_vs_length.png - Scatter plot showing relationship between response time and length
- question_type_analysis.png - Bar chart showing performance by question type


### evaluator.py
Main evaluation engine that:
- Handles model API calls
- Collects timing and response data
- Manages test case execution
- Calculates performance metrics

### visualize_results.py  
Visualization tool that:
- Creates statistical plots
- Generates performance comparisons
- Provides summary statistics
- Helps identify patterns in model behavior
