from prediction_core import prediction_pipeline

# Replace with the actual path to your test image
test_image_path = "D:\Dataset\ISIC_0000013.jpg"

# Test the pipeline
results = prediction_pipeline(test_image_path, "male", 45, "back")
print(results)