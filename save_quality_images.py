import os
import shutil
import csv
import json
from datetime import datetime

def save_quality_images(image_path, metadata, prediction_result):
    """
    Save images with good quality and that likely contain skin lesions to a separate folder,
    along with their metadata in a CSV file.
    
    Args:
        image_path (str): Path to the original image
        metadata (dict): Dictionary containing sex, age, anatomical site
        prediction_result (dict): The prediction result dict containing quality assessment
    
    Returns:
        bool: True if image was saved, False otherwise
    """
    # Create directory if it doesn't exist
    save_dir = "New_Images"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define CSV path
    csv_path = os.path.join(save_dir, "metadata.csv")
    
    # Check if this is a good quality image that likely contains a skin lesion
    if is_good_quality_skin_lesion(prediction_result):
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        extension = os.path.splitext(image_path)[1]
        new_filename = f"image_{timestamp}{extension}"
        save_path = os.path.join(save_dir, new_filename)
        
        # Copy the image to the new directory
        shutil.copy2(image_path, save_path)
        
        # Extract relevant data
        image_data = {
            'image_name': new_filename,
            'sex': metadata.get('sex', ''),
            'age': metadata.get('age_approx', ''),
            'anatomical_site': metadata.get('anatom_site', ''),
            'prediction': prediction_result.get('prediction', ''),
            'confidence': prediction_result.get('confidence_score', ''),
            'robustness': prediction_result.get('robustness', '')
        }
        
        # Save metadata to CSV
        save_metadata_to_csv(csv_path, image_data)
        
        return True
    
    return False

def is_good_quality_skin_lesion(prediction_result):
    """
    Determine if the image is good quality and likely contains a skin lesion.
    
    Args:
        prediction_result (dict): The prediction result dict
    
    Returns:
        bool: True if image is good quality and likely contains a skin lesion
    """
    # Check if we have agent decision info
    if 'agent_decision' not in prediction_result:
        return False
    
    agent_decision = prediction_result['agent_decision']
    
    # Look for quality indicators in the agent's feedback
    quality_indicators = [
        'good quality', 'acceptable', 'high quality', 'adequate'
    ]
    
    # Look for skin lesion content indicators
    skin_lesion_indicators = [
        'skin lesion', 'valid skin', 'proper skin', 'contains skin'
    ]
    
    # Check image feedback for quality
    image_feedback = agent_decision.get('image_feedback', '').lower()
    good_quality = any(indicator in image_feedback for indicator in quality_indicators)
    
    # Check content assessment for skin lesion
    content_assessment = agent_decision.get('content_assessment', '').lower()
    if not content_assessment:  # If content_assessment is empty, check detailed analyses
        if 'detailed_analyses' in agent_decision:
            content_analysis = agent_decision['detailed_analyses'].get('content', '').lower()
            is_skin_lesion = any(indicator in content_analysis for indicator in skin_lesion_indicators)
            
            # Also check for explicit negative indicators
            negative_indicators = ['not a skin lesion', 'unlikely to be a skin lesion']
            is_not_skin_lesion = any(indicator in content_analysis for indicator in negative_indicators)
            
            # If we found explicit negative indicators, override the positive check
            if is_not_skin_lesion:
                is_skin_lesion = False
    else:
        is_skin_lesion = any(indicator in content_assessment for indicator in skin_lesion_indicators)
        
        # Also check for explicit negative indicators
        negative_indicators = ['not a skin lesion', 'unlikely to be a skin lesion']
        is_not_skin_lesion = any(indicator in content_assessment for indicator in negative_indicators)
        
        # If we found explicit negative indicators, override the positive check
        if is_not_skin_lesion:
            is_skin_lesion = False
    
    # If both conditions are met, return True
    return good_quality and is_skin_lesion

def save_metadata_to_csv(csv_path, image_data):
    """
    Save image metadata to CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        image_data (dict): Dictionary containing image metadata
    """
    # Define CSV columns
    columns = ['image_name', 'sex', 'age', 'anatomical_site', 'prediction', 'confidence', 'robustness']
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Open file in append mode
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write row with data
        writer.writerow(image_data)