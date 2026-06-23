import os
import sys
import torch
from transformers import AutoTokenizer

print("\n" + "=" * 60)
print("  LAUNCHING LLM/RLHF WORKFLOW VALIDATION WORKER ")
print("=" * 60 + "\n")

# 1. Hardware Verification
print("[1/3] Querying Cluster Hardware...")
if torch.cuda.is_available():
    print(f"  --> Success! CUDA is active.")
    print(f"  --> Active GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print(f"  --> [ERROR] CUDA is NOT visible to PyTorch.")
    print(f"      Ensure you used the '--nv' flag with Singularity.")
    sys.exit(1)

# 2. Dependency & Functional Verification
print("\n[2/3] Initializing Hugging Face Tokenizer inside Container...")
try:
    # Using a tiny public model tokenizer to verify networking and code execution
    model_id = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dummy_text = "Singularity containers make cluster scaling seamless."
    tokens = tokenizer(dummy_text, return_tensors="pt").to(device)

    print("  --> Tokenizer loaded successfully!")
    print(f"  --> Input Text: \"{dummy_text}\"")
    print(f"  --> Generated Tensor IDs: {tokens['input_ids'].tolist()}")
except Exception as e:
    print(f"  --> [ERROR] Failed to run transformer workflow: {e}")
    sys.exit(1)

# 3. Host File System Verification
print("\n[3/3] Verifying Cluster Storage Write Access...")
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "summary.log")

try:
    with open(output_file, "w") as f:
        f.write("WORKFLOW STATUS: SUCCESS\n")
        f.write(f"GPU Utilized: {torch.cuda.get_device_name(0)}\n")
    print(f"  --> Success! Results exported safely to host path: {output_file}")
except Exception as e:
    print(f"  --> [ERROR] Host file system write failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  ALL SYSTEM CHECKS PASSED SUCCESSFULLY! ")
print("=" * 60 + "\n")