import streamlit as st

def main():
    
    st.set_page_config(
        page_title="Intelligent System Project", 
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main application header
    st.title("Intelligent System Project")
    st.markdown(""" --- """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Machine Learning")
        if st.button("Machine learning explaination", use_container_width=1):
            st.switch_page("pages/1_MachineLearning-Explaination.py")
        if st.button("Machine learning Model", use_container_width=1):
            st.switch_page("pages/2_MachineLearning-Model.py")
        
    
    with col2:
        st.subheader("Neural Networks")
        if st.button("Neural network explaination", use_container_width=1):
            st.switch_page("pages/3_NeuralNetwork-Explaination.py")
        if st.button("Neural network Model", use_container_width=1):
            st.switch_page("pages/4_NeuralNetwork-Model.py")


if __name__ == "__main__":
    main()