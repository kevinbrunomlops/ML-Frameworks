"""
Lektion 3 - Laboration 1: Miljobygge och verifiering
Assignment: GPU/CPU validation

Instructions:
1. Write and run the verification script
2. Save outputs and short notes in this file
"""

# Task 1: Verification script
# TODO: Create a script that prints:
# - Python version
# - OS info (platform module)
# - Installed versions of torch / tensorflow / jax (if available)

# Task 2: GPU checks
# TODO: If torch is installed:
# - Print torch.cuda.is_available()
# - Print torch.cuda.get_device_name(0) if available
# TODO: If tensorflow is installed:
# - Print tf.config.list_physical_devices("GPU")
# TODO: If jax is installed:
# - Print jax.devices()

# Task 3: Troubleshooting notes
# TODO: If GPU is NOT detected, write 3-6 comments about:
# - driver version
# - CUDA version
# - any error messages

print("Done! You can now validate your ML runtime.")
