import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from prediction_core import prediction_pipeline

def main():
    st.title("Melanoma Detection System")

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
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            # Call your prediction pipeline
            result = prediction_pipeline(temp_file_path, sex, age, anatom_site)

            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=300)

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
            st.subheader("AI Agent Feedback")

            st.markdown("**Image Quality Assessment:**")
            st.info(result['agent_decision']['image_feedback'])

            st.markdown("**Content Assessment:**")
            st.info(result['agent_decision']['content_assessment'])

            st.markdown("**Prediction Assessment:**")
            st.info(result['agent_decision']['prediction_feedback'])

            st.markdown("**Recommendation:**")
            st.info(result['agent_decision']['user_suggestion'])

            if result['agent_decision']['override_decision']['override']:
                st.warning("The AI agent has overridden the initial prediction due to quality or reliability issues.")

            # Add detailed analyses in expandable sections
            if 'detailed_analyses' in result['agent_decision']:
                with st.expander("View Detailed Analysis"):
                    st.subheader("Technical Quality Analysis")
                    st.text(result['agent_decision']['detailed_analyses']['technical_quality'])
                    
                    st.subheader("Content Analysis") 
                    st.text(result['agent_decision']['detailed_analyses']['content'])
                    
                    st.subheader("Prediction Reliability Analysis")
                    st.text(result['agent_decision']['detailed_analyses']['reliability'])

            # Display image metrics
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
            os.unlink(temp_file_path)

    elif analyze_button:
        st.warning("Please upload an image before analyzing.")

    # Disclaimer
    st.markdown("---")
    st.caption("This tool is for informational purposes only and is not a substitute for professional medical advice.")

if __name__ == "__main__":
    main()