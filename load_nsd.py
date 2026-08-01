from datasets import load_dataset

print("Downloading the NSD-Flat visual cortex dataset...")
# This command bypasses any web forms and downloads the data directly
dataset = load_dataset("clane9/NSD-Flat", split="train")

print(f"\nSuccess! Downloaded {len(dataset)} samples.")

# Let's look at the first sample
sample = dataset[0]

# The dataset contains the image and the corresponding brain activity
print("\nSample Keys Available:")
print(list(sample.keys()))

# If you want to see the image:
# sample['image'].show()