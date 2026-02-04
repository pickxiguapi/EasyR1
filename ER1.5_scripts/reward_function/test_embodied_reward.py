"""
Test suite for embodied_reward.py

Tests all supported problem types:
- multiple choice
- numerical
- open-ended
- math
- spatial grounding
- trace
- trace_3d
- point
"""
import json
import re

import pytest
from embodied_reward import (
    accuracy_reward,
    accuracy_reward_point,
    accuracy_reward_trace,
    accuracy_reward_trace_3d,
    compute_score,
    format_structure_reward_check,
)


# class TestFormatStructureCheck:
#     """Test format and structure validation"""

#     def test_valid_format_multiple_choice(self):
#         """Test valid format for multiple choice"""
#         response = "<think>Let me analyze this question.</think><answer>A</answer>"
#         score = format_structure_reward_check(response, "multiple choice")
#         assert score == 1.0

#     def test_invalid_format_missing_think(self):
#         """Test invalid format - missing think tags"""
#         response = "<answer>A</answer>"
#         score = format_structure_reward_check(response, "multiple choice")
#         assert score == 0.0

#     def test_invalid_format_missing_answer(self):
#         """Test invalid format - missing answer tags"""
#         response = "<think>Let me think.</think>"
#         score = format_structure_reward_check(response, "multiple choice")
#         assert score == 0.0

#     def test_invalid_format_wrong_order(self):
#         """Test invalid format - wrong tag order"""
#         response = "<answer>A</answer><think>Let me think.</think>"
#         score = format_structure_reward_check(response, "multiple choice")
#         assert score == 0.0

#     def test_valid_format_with_whitespace(self):
#         """Test valid format with extra whitespace"""
#         response = "<think>  Let me think.  </think>  <answer>  A  </answer>"
#         score = format_structure_reward_check(response, "multiple choice")
#         assert score == 1.0

#     def test_valid_format_with_whitespace2(self):
#         """Test valid format with extra whitespace"""
#         response = "< think >  Let me think.  </ think>  <answer>  A  </answer>"
#         response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
#         score = format_structure_reward_check(response, "multiple choice")
#         assert score == 1.0


# class TestMultipleChoice:
#     """Test multiple choice problem type"""

#     def test_correct_answer_letter(self):
#         """Test correct answer with letter option"""
#         reward_inputs = [{
#             "response": "<think>The answer is B.</think><answer>B</answer>",
#             "response_length": 100,
#             "ground_truth": "B",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "What is the answer?",
#             "problem_id": "mc_1"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0
#         assert results[0]["format_structure"] == 1.0

#     def test_incorrect_answer(self):
#         """Test incorrect answer"""
#         reward_inputs = [{
#             "response": "<think>I think it's A.</think><answer>A</answer>",
#             "response_length": 100,
#             "ground_truth": "B",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "What is the answer?",
#             "problem_id": "mc_2"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 0.0
#         assert results[0]["format_structure"] == 1.0

#     def test_answer_with_period(self):
#         """Test answer format: A.dog"""
#         reward_inputs = [{
#             "response": "<think>The answer is A.</think><answer>A.dog</answer>",
#             "response_length": 100,
#             "ground_truth": "A",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Choose the correct option",
#             "problem_id": "mc_3"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_answer_with_colon(self):
#         """Test answer format: A: dog"""
#         reward_inputs = [{
#             "response": "<think>The answer is A.</think><answer>A: dog</answer>",
#             "response_length": 100,
#             "ground_truth": "A",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Choose the correct option",
#             "problem_id": "mc_4"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_answer_with_parenthesis(self):
#         """Test answer format: A) dog"""
#         reward_inputs = [{
#             "response": "<think>The answer is A.</think><answer>A) dog</answer>",
#             "response_length": 100,
#             "ground_truth": "A",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Choose the correct option",
#             "problem_id": "mc_5"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_answer_with_space(self):
#         """Test answer format: A dog"""
#         reward_inputs = [{
#             "response": "<think>The answer is A.</think><answer>A dog</answer>",
#             "response_length": 100,
#             "ground_truth": "A",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Choose the correct option",
#             "problem_id": "mc_6"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_text_option_exact_match(self):
#         """Test text-based option (not letter)"""
#         reward_inputs = [{
#             "response": "<think>The answer is dog.</think><answer>dog</answer>",
#             "response_length": 100,
#             "ground_truth": "dog",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "What animal is this?",
#             "problem_id": "mc_7"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_text_option_incorrect(self):
#         """Test incorrect text-based option"""
#         reward_inputs = [{
#             "response": "<think>The answer is cat.</think><answer>cat</answer>",
#             "response_length": 100,
#             "ground_truth": "dog",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "What animal is this?",
#             "problem_id": "mc_8"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 0.0

#     def test_yes_no_lowercase(self):
#         """Test yes/no question with lowercase"""
#         reward_inputs = [{
#             "response": "<think>This is correct.</think><answer>yes</answer>",
#             "response_length": 100,
#             "ground_truth": "yes",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Is this correct?",
#             "problem_id": "mc_9"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_yes_no_uppercase(self):
#         """Test yes/no question with uppercase"""
#         reward_inputs = [{
#             "response": "<think>This is correct.</think><answer>YES</answer>",
#             "response_length": 100,
#             "ground_truth": "yes",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Is this correct?",
#             "problem_id": "mc_10"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_yes_no_mixed_case(self):
#         """Test yes/no question with mixed case"""
#         reward_inputs = [{
#             "response": "<think>This is correct.</think><answer>Yes</answer>",
#             "response_length": 100,
#             "ground_truth": "yes",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Is this correct?",
#             "problem_id": "mc_11"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_no_lowercase(self):
#         """Test no answer with lowercase"""
#         reward_inputs = [{
#             "response": "<think>This is incorrect.</think><answer>no</answer>",
#             "response_length": 100,
#             "ground_truth": "no",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Is this correct?",
#             "problem_id": "mc_12"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_no_uppercase(self):
#         """Test no answer with uppercase"""
#         reward_inputs = [{
#             "response": "<think>This is incorrect.</think><answer>NO</answer>",
#             "response_length": 100,
#             "ground_truth": "no",
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "Is this correct?",
#             "problem_id": "mc_13"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0


# class TestNumerical:
#     """Test numerical problem type"""

#     def test_correct_number(self):
#         """Test correct numerical answer"""
#         reward_inputs = [{
#             "response": "<think>Calculating...</think><answer>42.50</answer>",
#             "response_length": 100,
#             "ground_truth": "42.50",
#             "data_type": "text",
#             "problem_type": "numerical",
#             "problem": "Calculate the result",
#             "problem_id": "num_1"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_number_with_rounding(self):
#         """Test number with rounding (2 decimal places)"""
#         reward_inputs = [{
#             "response": "<think>Result is 42.501</think><answer>42.501</answer>",
#             "response_length": 100,
#             "ground_truth": "42.50",
#             "data_type": "text",
#             "problem_type": "numerical",
#             "problem": "Calculate the result",
#             "problem_id": "num_2"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0  # Should round to 42.50

#     def test_incorrect_number(self):
#         """Test incorrect numerical answer"""
#         reward_inputs = [{
#             "response": "<think>Result is 40</think><answer>40</answer>",
#             "response_length": 100,
#             "ground_truth": "42.50",
#             "data_type": "text",
#             "problem_type": "numerical",
#             "problem": "Calculate the result",
#             "problem_id": "num_3"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 0.0


# class TestMath:
#     """Test math problem type"""

#     def test_correct_math_answer(self):
#         """Test correct mathematical answer"""
#         reward_inputs = [{
#             "response": "<think>Solving equation...</think><answer>x = 5</answer>",
#             "response_length": 100,
#             "ground_truth": "x = 5",
#             "data_type": "text",
#             "problem_type": "math",
#             "problem": "Solve for x",
#             "problem_id": "math_1"
#         }]
#         results = compute_score(reward_inputs)
#         assert results[0]["accuracy"] == 1.0

#     def test_equivalent_math_expression(self):
#         """Test mathematically equivalent expressions"""
#         reward_inputs = [{
#             "response": "<think>Simplifying...</think><answer>2 + 3</answer>",
#             "response_length": 100,
#             "ground_truth": "5",
#             "data_type": "text",
#             "problem_type": "math",
#             "problem": "Simplify the expression",
#             "problem_id": "math_2"
#         }]
#         results = compute_score(reward_inputs)
#         # This depends on math_verify implementation
#         assert results[0]["accuracy"] in [0.0, 1.0]


# class TestMixedProblemTypes:
#     """Test batch processing with mixed problem types including open-ended"""

#     def test_mixed_types_with_open_ended(self):
#         """Test batch with multiple choice, numerical, and open-ended"""
#         reward_inputs = [
#             {
#                 "response": "<think>Answer is B</think><answer>B</answer>",
#                 "response_length": 100,
#                 "ground_truth": "B",
#                 "data_type": "text",
#                 "problem_type": "multiple choice",
#                 "problem": "Choose the answer",
#                 "problem_id": "mixed_mc_1"
#             },
#             {
#                 "response": "<think>Result is 42</think><answer>42.00</answer>",
#                 "response_length": 100,
#                 "ground_truth": "42.00",
#                 "data_type": "text",
#                 "problem_type": "numerical",
#                 "problem": "Calculate the value",
#                 "problem_id": "mixed_num_1"
#             },
#             {
#                 "response": "<think>Explanation</think><answer>The cat is sleeping.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The cat is sleeping.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "What is happening?",
#                 "problem_id": "mixed_1"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         assert len(results) == 3
#         # Multiple choice should be exact match
#         assert results[0]["accuracy"] == 1.0
#         # Numerical should be exact match
#         assert results[1]["accuracy"] == 1.0
#         # Open-ended should have high score (identical text)
#         assert results[2]["accuracy"] > 0.8

#     def test_multiple_open_ended_in_batch(self):
#         """Test multiple open-ended questions mixed with other types"""
#         reward_inputs = [
#             {
#                 "response": "<think>A</think><answer>A</answer>",
#                 "response_length": 100,
#                 "ground_truth": "A",
#                 "data_type": "text",
#                 "problem_type": "multiple choice",
#                 "problem": "Choose A",
#                 "problem_id": "mixed_mc_2"
#             },
#             {
#                 "response": "<think>Open 1</think><answer>First answer.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "First answer.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Question 1",
#                 "problem_id": "q1"
#             },
#             {
#                 "response": "<think>B</think><answer>B</answer>",
#                 "response_length": 100,
#                 "ground_truth": "B",
#                 "data_type": "text",
#                 "problem_type": "multiple choice",
#                 "problem": "Choose B",
#                 "problem_id": "mixed_mc_3"
#             },
#             {
#                 "response": "<think>Open 2</think><answer>Second answer.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "Second answer.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Question 2",
#                 "problem_id": "q2"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         assert len(results) == 4
#         # Multiple choice questions
#         assert results[0]["accuracy"] == 1.0
#         assert results[2]["accuracy"] == 1.0
#         # Open-ended questions
#         assert results[1]["accuracy"] > 0.8
#         assert results[3]["accuracy"] > 0.8


# class TestOpenEndedEdgeCases:
#     """Test edge cases for open-ended evaluation"""

#     def test_open_ended_empty_answer(self):
#         """Test open-ended with empty answer"""
#         reward_inputs = [{
#             "response": "<think>I don't know</think><answer></answer>",
#             "response_length": 100,
#             "ground_truth": "The answer is here.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "What is the answer?",
#             "problem_id": "empty_test"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Empty answer should have very low accuracy
#         assert results[0]["accuracy"] < 0.1
#         assert results[0]["format_structure"] == 0.0

#     def test_open_ended_invalid_format(self):
#         """Test open-ended with invalid format"""
#         reward_inputs = [{
#             "response": "No tags here",
#             "response_length": 100,
#             "ground_truth": "The answer.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "Question",
#             "problem_id": "invalid_format"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Invalid format should result in 0 scores
#         assert results[0]["accuracy"] == 0.0
#         assert results[0]["format_structure"] == 0.0
#         assert results[0]["overall"] == 0.0

#     def test_open_ended_missing_problem_field(self):
#         """Test open-ended without problem field"""
#         reward_inputs = [{
#             "response": "<think>Answer</think><answer>The answer.</answer>",
#             "response_length": 100,
#             "ground_truth": "The answer.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             # Missing "problem" field
#             "problem_id": "no_problem"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should still work (prompt will be None but ROUGE fallback should handle it)
#         assert len(results) == 1
#         assert results[0]["accuracy"] == 0.0

#     def test_open_ended_missing_problem_id(self):
#         """Test open-ended without problem_id field"""
#         reward_inputs = [{
#             "response": "<think>Answer</think><answer>The answer.</answer>",
#             "response_length": 100,
#             "ground_truth": "The answer.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "What is it?"
#             # Missing "problem_id" field
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should still work (will be grouped under None problem_id)
#         assert len(results) == 1
#         assert results[0]["accuracy"] == 0.0  # Identical text


# class TestOpenEndedBatchSize:
#     """Test batch size handling for open-ended questions"""

#     def test_batch_size_multiple_batches(self):
#         """Test processing across multiple batches"""
#         # Create enough questions to span multiple batches
#         reward_inputs = []
#         for i in range(10):
#             reward_inputs.append({
#                 "response": f"<think>Answer {i}</think><answer>Answer {i}.</answer>",
#                 "response_length": 100,
#                 "ground_truth": f"Answer {i}.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": f"Question {i}",
#                 "problem_id": f"q{i}"
#             })

#         results = compute_score(reward_inputs, format_weight=0.1)

#         assert len(results) == 10
#         # All should have high accuracy
#         for r in results:
#             assert r["accuracy"] > 0.8


# class TestFormatWeight:
#     """Test format_weight parameter"""

#     def test_format_weight_default(self):
#         """Test default format weight (0.1)"""
#         reward_inputs = [{
#             "response": "<think>Answer</think><answer>A</answer>",
#             "response_length": 100,
#             "ground_truth": "A",
#             "data_type": "image",
#             "problem_type": "multiple choice",
#             "problem": "What is the answer?",
#             "problem_id": "fw_1"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # overall = 0.9 * accuracy + 0.1 * format_structure
#         # = 0.9 * 1.0 + 0.1 * 1.0 = 1.0
#         assert results[0]["overall"] == 1.0

#     def test_format_weight_custom(self):
#         """Test custom format weight"""
#         reward_inputs = [{
#             "response": "<think>Answer</think><answer>B</answer>",
#             "response_length": 100,
#             "ground_truth": "A",  # Wrong answer
#             "data_type": "text",
#             "problem_type": "multiple choice",
#             "problem": "What is the answer?",
#             "problem_id": "fw_2"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.3)

#         # overall = 0.7 * 0.0 + 0.3 * 1.0 = 0.3
#         assert results[0]["overall"] == 0.3
#         assert results[0]["accuracy"] == 0.0
#         assert results[0]["format_structure"] == 1.0


# class TestOpenEndedBatchProcessing:
#     """Test open-ended batch processing with reward model"""

#     def test_open_ended_basic(self):
#         """Test basic open-ended evaluation (will use ROUGE as fallback)"""
#         reward_inputs = [{
#             "response": "<think>Let me explain.</think><answer>The cat sat on the mat.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat sat on the mat.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see.",
#             "problem_id": "test_1"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should have high accuracy for identical text
#         assert results[0]["accuracy"] > 0.8
#         assert results[0]["format_structure"] == 1.0
#         assert 0.0 <= results[0]["overall"] <= 1.0

#     def test_open_ended_batch_multiple(self):
#         """Test batch processing with multiple open-ended questions"""
#         reward_inputs = [
#             {
#                 "response": "<think>Answer 1</think><answer>The sky is blue.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The sky is blue.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "What color is the sky?",
#                 "problem_id": "q1"
#             },
#             {
#                 "response": "<think>Answer 2</think><answer>Water is wet.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "Water is wet.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Describe water.",
#                 "problem_id": "q2"
#             },
#             {
#                 "response": "<think>Answer 3</think><answer>Fire is hot.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "Fire is hot.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Describe fire.",
#                 "problem_id": "q3"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         assert len(results) == 3
#         # All should have high accuracy (identical answers)
#         for i in range(3):
#             assert results[i]["accuracy"] > 0.8
#             assert results[i]["format_structure"] == 1.0

#     def test_open_ended_partial_match(self):
#         """Test open-ended with partially matching answer"""
#         reward_inputs = [{
#             "response": "<think>Explaining...</think><answer>The dog sat on the floor.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat sat on the mat.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "Describe the scene.",
#             "problem_id": "test_2"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should have partial match (some words overlap)
#         assert 0.0 < results[0]["accuracy"] < 1.0
#         assert results[0]["format_structure"] == 1.0

#     def test_open_ended_no_match(self):
#         """Test open-ended with completely different answer"""
#         reward_inputs = [{
#             "response": "<think>My answer</think><answer>Elephants are large animals.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat sat on the mat.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "Describe the scene.",
#             "problem_id": "test_3"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should have very low accuracy (no overlap)
#         assert results[0]["accuracy"] < 0.3
#         assert results[0]["format_structure"] == 1.0


# class TestOpenEndedNormalization:
#     """Test normalization logic for open-ended questions"""

#     def test_normalization_same_problem_id(self):
#         """Test normalization groups by problem_id correctly"""
#         # Create multiple responses for the same problem
#         reward_inputs = [
#             {
#                 "response": "<think>Good answer</think><answer>This is a very detailed and comprehensive answer.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "This is a detailed answer.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Question 1",
#                 "problem_id": "same_problem"
#             },
#             {
#                 "response": "<think>Medium answer</think><answer>This is an answer.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "This is a detailed answer.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Question 1",
#                 "problem_id": "same_problem"
#             },
#             {
#                 "response": "<think>Poor answer</think><answer>Answer.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "This is a detailed answer.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Question 1",
#                 "problem_id": "same_problem"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # After normalization, scores should be spread across [0, 1]
#         # Best answer should have highest score
#         assert results[0]["accuracy"] > results[1]["accuracy"]
#         assert results[1]["accuracy"] > results[2]["accuracy"]

#         # All scores should be in valid range
#         for r in results:
#             assert 0.0 <= r["accuracy"] <= 1.0

#     def test_normalization_different_problem_ids(self):
#         """Test normalization is applied per problem_id"""
#         reward_inputs = [
#             {
#                 "response": "<think>Answer</think><answer>Sky is blue.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "Sky is blue.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Q1",
#                 "problem_id": "problem_1"
#             },
#             {
#                 "response": "<think>Answer</think><answer>Water is wet.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "Water is wet.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Q2",
#                 "problem_id": "problem_2"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Both should have high scores (perfect matches)
#         # Normalization should be applied separately per problem_id
#         assert results[0]["accuracy"] > 0.8
#         assert results[1]["accuracy"] > 0.8


# class TestPointWithCount:
#     """Test point localization with count field"""

#     def test_point_with_count_correct(self):
#         """Test point with correct count and positions"""
#         gt_data = [
#             {"count": 2},
#             {"point_2d": [410, 582]},
#             {"point_2d": [363, 601]}
#         ]
#         pred_data = [
#             {"point_2d": [412, 580]},  # Close to first point
#             {"point_2d": [365, 600]}   # Close to second point
#         ]

#         reward_inputs = [{
#             "response": f'<think>Finding points</think><answer>{json.dumps(pred_data)}</answer>',
#             "response_length": 100,
#             "ground_truth": json.dumps(gt_data),
#             "data_type": "image",
#             "problem_type": "point",
#             "problem": "Find the points",
#             "problem_id": "point_count_1"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should have high accuracy (correct count, close positions)
#         assert results[0]["accuracy"] > 0.8

#     def test_point_with_count_wrong_count(self):
#         """Test point with wrong count"""
#         gt_data = [
#             {"count": 2},
#             {"point_2d": [410, 582]},
#             {"point_2d": [363, 601]}
#         ]
#         pred_data = [
#             {"point_2d": [412, 580]},
#             {"point_2d": [365, 600]},
#             {"point_2d": [400, 500]}  # Extra point
#         ]

#         reward_inputs = [{
#             "response": f'<think>Finding points</think><answer>{json.dumps(pred_data)}</answer>',
#             "response_length": 100,
#             "ground_truth": json.dumps(gt_data),
#             "data_type": "image",
#             "problem_type": "point",
#             "problem": "Find the points",
#             "problem_id": "point_count_2"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Should have penalty for wrong count
#         assert results[0]["accuracy"] < 0.8


# class TestPointWithSegmentation:
#     """Test point localization with segmentation (polygon)"""

#     def test_point_in_polygon_all_inside(self):
#         """Test all points inside polygon"""
#         gt_data = [{
#             "segmentation": [
#                 [475, 433],
#                 [444, 533],
#                 [469, 525],
#                 [506, 617],
#                 [531, 625],
#                 [556, 433]
#             ]
#         }]
#         pred_data = [
#             {"point_2d": [500, 500]},  # Inside
#             {"point_2d": [480, 450]}   # Inside
#         ]

#         reward_inputs = [{
#             "response": f'<think>Finding points</think><answer>{json.dumps(pred_data)}</answer>',
#             "response_length": 100,
#             "ground_truth": json.dumps(gt_data),
#             "data_type": "image",
#             "problem_type": "point",
#             "problem": "Find points in region",
#             "problem_id": "point_seg_1"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # All points inside, should be 1.0
#         assert results[0]["accuracy"] == 1.0

#     def test_point_in_polygon_partial(self):
#         """Test partial points inside polygon"""
#         gt_data = [{
#             "segmentation": [
#                 [475, 433],
#                 [444, 533],
#                 [469, 525],
#                 [506, 617],
#                 [531, 625],
#                 [556, 433]
#             ]
#         }]
#         pred_data = [
#             {"point_2d": [500, 500]},  # Inside
#             {"point_2d": [100, 100]},  # Outside
#             {"point_2d": [480, 450]}   # Inside
#         ]

#         reward_inputs = [{
#             "response": f'<think>Finding points</think><answer>{json.dumps(pred_data)}</answer>',
#             "response_length": 100,
#             "ground_truth": json.dumps(gt_data),
#             "data_type": "image",
#             "problem_type": "point",
#             "problem": "Find points in region",
#             "problem_id": "point_seg_2"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # 2 out of 3 points inside, should be ~0.67
#         assert 0.6 < results[0]["accuracy"] < 0.7


# class TestPointWithBox:
#     """Test point localization with box_2d"""

#     def test_point_in_box_all_inside(self):
#         """Test all points inside box"""
#         gt_data = [{
#             "box_2d": [624, 392, 714, 564]
#         }]
#         pred_data = [
#             {"point_2d": [650, 450]},  # Inside
#             {"point_2d": [700, 500]}   # Inside
#         ]

#         reward_inputs = [{
#             "response": f'<think>Finding points</think><answer>{json.dumps(pred_data)}</answer>',
#             "response_length": 100,
#             "ground_truth": json.dumps(gt_data),
#             "data_type": "image",
#             "problem_type": "point",
#             "problem": "Find points in box",
#             "problem_id": "point_box_1"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # All points inside, should be 1.0
#         assert results[0]["accuracy"] == 1.0

#     def test_point_in_box_partial(self):
#         """Test partial points inside box"""
#         gt_data = [{
#             "box_2d": [624, 392, 714, 564]
#         }]
#         pred_data = [
#             {"point_2d": [650, 450]},  # Inside
#             {"point_2d": [800, 800]},  # Outside
#             {"point_2d": [700, 500]}   # Inside
#         ]

#         reward_inputs = [{
#             "response": f'<think>Finding points</think><answer>{json.dumps(pred_data)}</answer>',
#             "response_length": 100,
#             "ground_truth": json.dumps(gt_data),
#             "data_type": "image",
#             "problem_type": "point",
#             "problem": "Find points in box",
#             "problem_id": "point_box_2"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # 2 out of 3 points inside, should be ~0.67
#         assert 0.6 < results[0]["accuracy"] < 0.7


# class TestOpenEndedModelEvaluation:
#     """Test open-ended evaluation with reward model"""

#     def test_perfect_answer(self):
#         """Test identical answer to ground truth"""
#         reward_inputs = [{
#             "response": "<think>Let me describe this.</think><answer>The cat is sitting on the mat.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat is sitting on the mat.",
#             "data_type": "image",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see in the image.",
#             "problem_id": "oe_perfect"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Perfect match should have very high accuracy
#         assert results[0]["accuracy"] > 0.9
#         assert results[0]["format_structure"] == 1.0
#         assert results[0]["overall"] > 0.9

#     def test_good_answer(self):
#         """Test semantically similar answer"""
#         reward_inputs = [{
#             "response": "<think>Analyzing the scene.</think><answer>A cat is resting on a mat.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat is sitting on the mat.",
#             "data_type": "image",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see in the image.",
#             "problem_id": "oe_good"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Good answer should have high accuracy
#         assert results[0]["accuracy"] > 0.7
#         assert results[0]["format_structure"] == 1.0

#     def test_partial_answer(self):
#         """Test partially correct answer"""
#         reward_inputs = [{
#             "response": "<think>I see an animal.</think><answer>There is a cat in the image.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat is sitting on the mat.",
#             "data_type": "image",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see in the image.",
#             "problem_id": "oe_partial"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         # Partial answer should have medium accuracy
#         assert 0.3 < results[0]["accuracy"] < 0.9
#         assert results[0]["format_structure"] == 1.0

#     def test_poor_answer(self):
#         """Test incorrect or irrelevant answer"""
#         reward_inputs = [{
#             "response": "<think>Looking at this.</think><answer>The sky is blue and birds are flying.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat is sitting on the mat.",
#             "data_type": "image",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see in the image.",
#             "problem_id": "oe_poor"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         # Poor answer should have low accuracy
#         assert results[0]["accuracy"] < 0.4
#         assert results[0]["format_structure"] == 1.0

#     def test_very_short_answer(self):
#         """Test very short answer"""
#         reward_inputs = [{
#             "response": "<think>Brief response.</think><answer>Cat.</answer>",
#             "response_length": 100,
#             "ground_truth": "The cat is sitting on the mat.",
#             "data_type": "image",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see.",
#             "problem_id": "oe_short"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         # Short answer should have low to medium accuracy
#         assert 0.0 < results[0]["accuracy"] < 0.9
#         assert results[0]["format_structure"] == 1.0

#     def test_very_long_answer(self):
#         """Test very long detailed answer"""
#         long_answer = "The image shows a domestic cat, which appears to be a tabby cat with distinctive markings. The cat is positioned on what looks like a mat or rug, sitting in a relaxed posture. The setting appears to be indoors, and the cat seems calm and comfortable in its environment."
#         reward_inputs = [{
#             "response": f"<think>Providing detailed description.</think><answer>{long_answer}</answer>",
#             "response_length": 200,
#             "ground_truth": "The cat is sitting on the mat.",
#             "data_type": "image",
#             "problem_type": "open-ended",
#             "problem": "Describe what you see in the image.",
#             "problem_id": "oe_long"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         # Long detailed answer should have good accuracy if relevant
#         assert results[0]["accuracy"] > 0.5
#         assert results[0]["format_structure"] == 1.0


# class TestOpenEndedBatchEvaluation:
#     """Test batch processing for open-ended questions with reward model"""

#     def test_batch_same_quality(self):
#         """Test batch with similar quality answers"""
#         reward_inputs = [
#             {
#                 "response": "<think>Answer 1</think><answer>The dog is running in the park.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The dog is running in the park.",
#                 "data_type": "video",
#                 "problem_type": "open-ended",
#                 "problem": "What is happening?",
#                 "problem_id": "batch_1"
#             },
#             {
#                 "response": "<think>Answer 2</think><answer>The bird is flying in the sky.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The bird is flying in the sky.",
#                 "data_type": "video",
#                 "problem_type": "open-ended",
#                 "problem": "What is happening?",
#                 "problem_id": "batch_2"
#             },
#             {
#                 "response": "<think>Answer 3</think><answer>The fish is swimming in the water.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The fish is swimming in the water.",
#                 "data_type": "video",
#                 "problem_type": "open-ended",
#                 "problem": "What is happening?",
#                 "problem_id": "batch_3"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         assert len(results) == 3
#         # All perfect matches should have high accuracy
#         for r in results:
#             assert r["accuracy"] > 0.9
#             assert r["format_structure"] == 1.0

#     def test_batch_mixed_quality(self):
#         """Test batch with varying quality answers"""
#         reward_inputs = [
#             {
#                 "response": "<think>Perfect</think><answer>The cat is sleeping.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The cat is sleeping.",
#                 "data_type": "image",
#                 "problem_type": "open-ended",
#                 "problem": "What is the cat doing?",
#                 "problem_id": "mix_1"
#             },
#             {
#                 "response": "<think>Good</think><answer>A cat is resting.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The cat is sleeping.",
#                 "data_type": "image",
#                 "problem_type": "open-ended",
#                 "problem": "What is the cat doing?",
#                 "problem_id": "mix_1"
#             },
#             {
#                 "response": "<think>Poor</think><answer>There is furniture.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The cat is sleeping.",
#                 "data_type": "image",
#                 "problem_type": "open-ended",
#                 "problem": "What is the cat doing?",
#                 "problem_id": "mix_1"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         assert len(results) == 3
#         # Quality should decrease: perfect > good > poor
#         assert results[0]["accuracy"] > results[1]["accuracy"]
#         assert results[1]["accuracy"] > results[2]["accuracy"]


# class TestOpenEndedMixedBatch:
#     """Test open-ended mixed with other problem types in batch"""

#     def test_mixed_with_multiple_choice(self):
#         """Test batch with open-ended and multiple choice"""
#         reward_inputs = [
#             {
#                 "response": "<think>Choice</think><answer>A</answer>",
#                 "response_length": 50,
#                 "ground_truth": "A",
#                 "data_type": "text",
#                 "problem_type": "multiple choice",
#                 "problem": "Choose A",
#                 "problem_id": "mc_1"
#             },
#             {
#                 "response": "<think>Description</think><answer>The person is walking.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The person is walking.",
#                 "data_type": "video",
#                 "problem_type": "open-ended",
#                 "problem": "What is happening?",
#                 "problem_id": "oe_1"
#             },
#             {
#                 "response": "<think>Choice</think><answer>B</answer>",
#                 "response_length": 50,
#                 "ground_truth": "B",
#                 "data_type": "text",
#                 "problem_type": "multiple choice",
#                 "problem": "Choose B",
#                 "problem_id": "mc_2"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         assert len(results) == 3
#         # Multiple choice should be exact
#         assert results[0]["accuracy"] == 1.0
#         assert results[2]["accuracy"] == 1.0
#         # Open-ended should be high (perfect match)
#         assert results[1]["accuracy"] > 0.9

#     def test_mixed_with_numerical(self):
#         """Test batch with open-ended and numerical"""
#         reward_inputs = [
#             {
#                 "response": "<think>Calculating</think><answer>42.5</answer>",
#                 "response_length": 50,
#                 "ground_truth": "42.5",
#                 "data_type": "text",
#                 "problem_type": "numerical",
#                 "problem": "Calculate",
#                 "problem_id": "num_1"
#             },
#             {
#                 "response": "<think>Explaining</think><answer>The result is obtained by adding the values.</answer>",
#                 "response_length": 100,
#                 "ground_truth": "The result is obtained by adding the values.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": "Explain the process",
#                 "problem_id": "oe_2"
#             }
#         ]
#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         assert len(results) == 2
#         assert results[0]["accuracy"] == 1.0  # Numerical exact
#         assert results[1]["accuracy"] > 0.9  # Open-ended perfect match


# class TestOpenEndedEdgeCasesModel:
#     """Test edge cases for open-ended with model evaluation"""

#     def test_empty_answer_content(self):
#         """Test open-ended with empty answer content"""
#         reward_inputs = [{
#             "response": "<think>I don't know.</think><answer></answer>",
#             "response_length": 50,
#             "ground_truth": "The answer is here.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "What is the answer?",
#             "problem_id": "edge_empty"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Empty answer should have very low accuracy
#         assert results[0]["accuracy"] < 0.1
#         assert results[0]["format_structure"] == 0.0

#     def test_whitespace_only_answer(self):
#         """Test open-ended with whitespace-only answer"""
#         reward_inputs = [{
#             "response": "<think>Thinking...</think><answer>   </answer>",
#             "response_length": 50,
#             "ground_truth": "The answer is here.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "What is the answer?",
#             "problem_id": "edge_whitespace"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Whitespace-only should have very low accuracy
#         assert results[0]["accuracy"] < 0.1
#         assert results[0]["format_structure"] == 0.0

#     def test_invalid_format_no_tags(self):
#         """Test open-ended with invalid format (no tags)"""
#         reward_inputs = [{
#             "response": "Just plain text without tags",
#             "response_length": 50,
#             "ground_truth": "The answer.",
#             "data_type": "text",
#             "problem_type": "open-ended",
#             "problem": "Question",
#             "problem_id": "edge_no_tags"
#         }]
#         results = compute_score(reward_inputs, format_weight=0.1)

#         # Invalid format should result in 0 scores
#         assert results[0]["accuracy"] == 0.0
#         assert results[0]["format_structure"] == 0.0
#         assert results[0]["overall"] == 0.0


# class TestOpenEndedLargeBatch:
#     """Test large batch processing for open-ended questions"""

#     def test_large_batch_50_questions(self):
#         """Test processing 50 open-ended questions in one batch"""
#         reward_inputs = []
#         for i in range(50):
#             reward_inputs.append({
#                 "response": f"<think>Answer {i}</think><answer>This is answer number {i}.</answer>",
#                 "response_length": 100,
#                 "ground_truth": f"This is answer number {i}.",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": f"Question {i}",
#                 "problem_id": f"large_q"
#             })

#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         assert len(results) == 50
#         # All should have high accuracy (perfect matches)
#         for r in results:
#             assert r["accuracy"] > 0.9
#             assert r["format_structure"] == 1.0

#     def test_large_batch_mixed_quality(self):
#         """Test large batch with varying quality answers"""
#         reward_inputs = []
#         # Add 30 questions with varying quality
#         for i in range(10):
#             # Perfect matches
#             reward_inputs.append({
#                 "response": f"<think>Perfect</think><answer>Answer {i}</answer>",
#                 "response_length": 100,
#                 "ground_truth": f"Answer {i}",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": f"Q{i}",
#                 "problem_id": f"batch_perfect_{i}"
#             })
#         for i in range(10):
#             # Good matches
#             reward_inputs.append({
#                 "response": f"<think>Good</think><answer>The answer is {i}</answer>",
#                 "response_length": 100,
#                 "ground_truth": f"Answer {i}",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": f"Q{i}",
#                 "problem_id": f"batch_good_{i}"
#             })
#         for i in range(10):
#             # Poor matches
#             reward_inputs.append({
#                 "response": f"<think>Poor</think><answer>Something else entirely</answer>",
#                 "response_length": 100,
#                 "ground_truth": f"Answer {i}",
#                 "data_type": "text",
#                 "problem_type": "open-ended",
#                 "problem": f"Q{i}",
#                 "problem_id": f"batch_poor_{i}"
#             })

#         results = compute_score(reward_inputs, format_weight=0.1, normalize_model_reward_by_problem_id=False)

#         assert len(results) == 30
#         # Perfect matches should have highest scores
#         perfect_scores = [results[i]["accuracy"] for i in range(10)]
#         good_scores = [results[i]["accuracy"] for i in range(10, 20)]
#         poor_scores = [results[i]["accuracy"] for i in range(20, 30)]

#         # Average scores should decrease: perfect > good > poor
#         assert sum(perfect_scores) / len(perfect_scores) > sum(good_scores) / len(good_scores)
#         assert sum(good_scores) / len(good_scores) > sum(poor_scores) / len(poor_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
