#!/usr/bin/env python3
"""Test all malformed JSON cases from real model outputs"""

import sys
sys.path.insert(0, '/home/claude/downloads/723_project')

from scripts.eval_probe_optimized import extract_answer_and_prob

print("="*70)
print("TESTING ALL MALFORMED JSON CASES")
print("="*70)

test_cases = [
    # 1.5B MMLU - Extra closing brace
    ('{"answer":"4","p_true":"0.85"}}', "4", 0.85, "Extra closing brace"),
    
    # 4B TriviaQA - Single quotes + comma decimal
    ('{"answer\': \'Ross Bagdasarian\', \'p_true\': 0,95}', "Ross Bagdasarian", 0.95, "Single quotes + comma decimal"),
    
    # 4B TriviaQA - Comma decimal
    ('{"answer": "Scorpio", "p_true": 0,85}', "Scorpio", 0.85, "Comma decimal"),
    
    # 4B TriviaQA - Wrong key name
    ('{"answer": "Cats", "pTrue": 1}', "Cats", 1.0, "pTrue instead of p_true"),
    
    # 4B GSM8K - Space in key name
    ('{"answer": "2", " p_true": "0.85"}', "2", 0.85, "Space in key name"),
    
    # 4B GSM8K - Single quotes + comma
    ('{"answer\': \'3\', \'p_true\': 0,95}', "3", 0.95, "Single quotes + comma"),
    
    # 4B GSM8K - Confidence instead of p_true
    ('{"answer": "300", "confidence": 1}', "300", 1.0, "confidence key"),
    
    # 4B MMLU - Space in key
    ('{"answer": "6", " p_true": "0.85"}', "6", 0.85, "Space in p_true"),
    
    # 4B MMLU - Single quote + missing opening quote
    ('{"answer\': "2", "confidence": 1}', "2", 1.0, "Mangled quotes"),
    
    # Valid JSON (should still work)
    ('{"answer": "Paris", "p_true": 0.92}', "Paris", 0.92, "Valid JSON"),
]

passed = 0
failed = 0

for text, expected_answer, expected_prob, description in test_cases:
    try:
        answer, prob = extract_answer_and_prob(text)
        
        # Check answer
        if answer != expected_answer:
            print(f"\n✗ FAIL: {description}")
            print(f"  Input: {text}")
            print(f"  Expected answer: '{expected_answer}'")
            print(f"  Got answer: '{answer}'")
            failed += 1
            continue
        
        # Check probability
        if prob is None and expected_prob is not None:
            print(f"\n✗ FAIL: {description}")
            print(f"  Input: {text}")
            print(f"  Expected prob: {expected_prob}")
            print(f"  Got prob: None")
            failed += 1
            continue
        
        if prob is not None and abs(prob - expected_prob) > 0.01:
            print(f"\n✗ FAIL: {description}")
            print(f"  Input: {text}")
            print(f"  Expected prob: {expected_prob}")
            print(f"  Got prob: {prob}")
            failed += 1
            continue
        
        print(f"✓ PASS: {description}")
        passed += 1
        
    except Exception as e:
        print(f"\n✗ ERROR: {description}")
        print(f"  Input: {text}")
        print(f"  Exception: {e}")
        failed += 1

print("\n" + "="*70)
print(f"Results: {passed} passed, {failed} failed")
print("="*70)

if failed == 0:
    print("\n✅ ALL TESTS PASSED!")
    print("Malformed JSON parsing is now robust.")
else:
    print(f"\n⚠️  {failed} tests failed. Check the output above.")