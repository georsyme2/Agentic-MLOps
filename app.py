import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import time
import traceback

from prediction_core import prediction_pipeline

def main():
    st.title("Melanoma Detection System")
    
    # Add debugging info
    st.sidebar.title("Debug Info")
    debug_info = st.sidebar.empty()

    # Input form
    uploaded_file = st.file_uploader("Upload image of skin lesion", type=["jpg", "jpeg", "png"])

    col1, col2, col3 = st.columns(3)
    with col1:
        sex = st.selectbox("Sex", ["male", "female"])
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col3:
        anatom_site = st.selectbox("Anatomical Site",
                                   ["back", "chest", "face", "scalp", "abdomen",
                                    "upper extremity", "lower extremity", "neck",
                                    "hand", "foot", "ear", "genital", "unknown"])

    analyze_button = st.button("Analyze Image")

    if analyze_button and uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            try:
                debug_info.info("Starting analysis process...")
                
                # Save uploaded file to temporary location
                debug_info.info("Saving temporary file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                debug_info.info(f"Temporary file saved to: {temp_file_path}")
                
                # Display the uploaded image
                st.image(uploaded_file, caption="Uploaded Image", width=300)
                
                # Call prediction pipeline with timeouts and progress updates
                debug_info.info("Calling prediction pipeline...")
                start_time = time.time()
                
                # Create a placeholder for progress updates
                progress_placeholder = st.empty()
                progress_text = "Analysis in progress: Preprocessing image..."
                progress_placeholder.info(progress_text)
                
                # Call your prediction pipeline
                result = prediction_pipeline(temp_file_path, sex, age, anatom_site)
                
                total_time = time.time() - start_time
                debug_info.info(f"Prediction completed in {total_time:.2f} seconds")
                progress_placeholder.empty()
                
                # Display prediction results
                st.header("Analysis Results")

                # Main prediction
                prediction = "Potential Melanoma" if result['prediction'] == 1 else "Likely Benign"
                confidence = result['confidence_score']

                # Color-coded results
                if result['prediction'] == 1:
                    st.markdown(f"### Prediction: <span style='color:red'>{prediction}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### Prediction: <span style='color:green'>{prediction}</span>", unsafe_allow_html=True)

                st.markdown(f"**Confidence Score:** {confidence:.2%}")

                # Enhanced robustness display
                if 'robustness_details' in result:
                    robustness = result['robustness']
                    robustness_details = result['robustness_details']
                    
                    # Display basic robustness status with appropriate color
                    if robustness == "Stable":
                        st.markdown(f"**Robustness:** <span style='color:green'>{robustness}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Robustness:** <span style='color:red'>{robustness}</span>", unsafe_allow_html=True)
                    
                    # Display detailed robustness metrics
                    with st.expander("View Robustness Details"):
                        st.markdown(f"**Stability Score:** {robustness_details['stability_score']:.2%} of augmented images received the same prediction")
                        st.markdown(f"**Augmentations Tested:** {robustness_details['augmentations_tested']}")
                        
                        # Create a bar chart of prediction counts
                        pred_counts = robustness_details['prediction_counts']
                        st.subheader("Prediction Distribution Across Augmentations")
                        
                        # Convert prediction counts to a format suitable for plotting
                        pred_df = pd.DataFrame({
                            'Prediction': [("Melanoma" if int(k) == 1 else "Benign") for k in pred_counts.keys()],
                            'Count': list(pred_counts.values())
                        })
                        
                        # Plot bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['red' if pred == 'Melanoma' else 'green' for pred in pred_df['Prediction']]
                        pred_df.plot.bar(x='Prediction', y='Count', ax=ax, color=colors)
                        ax.set_title('Prediction Distribution Across Augmented Images')
                        ax.set_ylabel('Number of Images')
                        st.pyplot(fig)
                else:
                    # Fallback for old format without details
                    st.markdown(f"**Robustness:** {result['robustness']}")

                
                # In app.py, update the "AI Agent Feedback" section:

                # Agent feedback section
                debug_info.info("Displaying agent feedback...")
                st.subheader("AI Agent Feedback")

                # Check if agent_decision exists and has expected structure
                if 'agent_decision' in result and isinstance(result['agent_decision'], dict):
                    agent_decision = result['agent_decision']
                    
                    # Handle different agent decision formats
                    if 'image_feedback' in agent_decision:
                        st.markdown("**Image Quality Assessment:**")
                        st.info(agent_decision.get('image_feedback', 'No feedback available'))
                    
                    if 'content_assessment' in agent_decision:
                        st.markdown("**Content Assessment:**")
                        st.info(agent_decision.get('content_assessment', 'No assessment available'))
                    
                    if 'prediction_feedback' in agent_decision:
                        st.markdown("**Prediction Assessment:**")
                        st.info(agent_decision.get('prediction_feedback', 'No feedback available'))
                    
                    if 'user_suggestion' in agent_decision:
                        st.markdown("**Recommendation:**")
                        st.info(agent_decision.get('user_suggestion', 'No recommendations available'))
                    
                    # Handle override decision
                    override_info = agent_decision.get('override_decision', {})
                    if isinstance(override_info, dict) and override_info.get('override', False):
                        st.warning("The AI agent has overridden the initial prediction due to quality or reliability issues.")
                    
                    # Add detailed analyses in expandable sections
                    if 'detailed_analyses' in agent_decision:
                        with st.expander("View Detailed Analysis"):
                            detailed = agent_decision['detailed_analyses']
                            if isinstance(detailed, dict):
                                if 'technical_quality' in detailed:
                                    st.subheader("Technical Quality Analysis")
                                    st.text(detailed.get('technical_quality', 'Not available'))
                                
                                if 'content' in detailed:
                                    st.subheader("Content Analysis") 
                                    st.text(detailed.get('content', 'Not available'))
                                
                                if 'reliability' in detailed:
                                    st.subheader("Prediction Reliability Analysis")
                                    st.text(detailed.get('reliability', 'Not available'))
                else:
                    st.warning("AI agent feedback not available for this prediction.")

                # Display image metrics
                debug_info.info("Displaying image metrics...")
                st.subheader("Image Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Sharpness', 'Brightness', 'Contrast'],
                    'Value': [
                        result['image_metrics']['sharpness'],
                        result['image_metrics']['brightness'],
                        result['image_metrics']['contrast']
                    ]
                })
                st.dataframe(metrics_df)

                # Simple bar chart for metrics
                fig, ax = plt.subplots(figsize=(10, 4))
                metrics_df.plot.bar(x='Metric', y='Value', ax=ax)
                ax.set_title('Image Quality Metrics')
                st.pyplot(fig)

                # Raw data in expandable section
                with st.expander("View Raw Results"):
                    st.json(result)

                # Clean up the temporary file
                debug_info.info("Cleaning up temporary file...")
                os.unlink(temp_file_path)
                debug_info.success("Analysis completed successfully!")

            except Exception as e:
                error_message = f"Error during analysis: {str(e)}"
                st.error(error_message)
                debug_info.error(error_message)
                debug_info.error(traceback.format_exc())
                
                # Try to clean up temp file if it exists
                try:
                    if 'temp_file_path' in locals():
                        os.unlink(temp_file_path)
                except:
                    pass

    elif analyze_button:
        st.warning("Please upload an image before analyzing.")

    # Disclaimer
    st.markdown("---")
    st.caption("This tool is for informational purposes only and is not a substitute for professional medical advice.")

if __name__ == "__main__":
    main()