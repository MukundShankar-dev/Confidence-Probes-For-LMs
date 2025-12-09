# import re

# text = '{"answer": "6", " p_true": "0.85"}'
# print(f"Input: {text}")

# # Step 6: Remove space in key
# result = re.sub(r'"\s+(\w+)":', r'"\1":', text)
# print(f"After step 6: {result}")

# # Step 7: Your exact line
# result = re.sub(r'"p_true":\s*"([0-9.]+)"', r'"p_true": ' + r'\1', result)
# print(f"After step 7: {result}")

# # Try to parse
# import json
# try:
#     obj = json.loads(result)
#     print(f"SUCCESS: {obj}")
# except Exception as e:
#     print(f"FAILED: {e}")
    
# # Also show what the replacement actually is
# replacement = r'"p_true": ' + r'\1'
# print(f"\nReplacement string: {repr(replacement)}")

import json
s = '{"answer": "6", "p_true": 0.85}'
print(f"String: {s}")
try:
    result = json.loads(s)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")