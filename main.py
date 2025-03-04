import streamlit as st
import pathlib
# from src.pages import overview, visualization, prediction

def main():
    st.set_page_config(page_title="Intelligent System Project", page_icon=":robot_face:")

    sidebar()

    st.title("Intelligent System Project")
    st.write("Navigate through the pages to explore different ML models and explanations.")

def sidebar() -> None:
    st.sidebar.title("Pages")

    # Section 1
    st.sidebar.subheader("1. Machine Learning")
    st.sidebar.link_button(label="Explaination", url="https://openai.com", type="tertiary")
    st.sidebar.link_button(label="Model", url="#credit-card", type="tertiary")
    
    st.sidebar
    # Section 2
    st.sidebar.subheader("2. Neural Networks")
    st.sidebar.link_button("Explaination", "https://openai.com", type="tertiary")
    st.sidebar.link_button("Model", "https://openai.com", type="tertiary")
    st.sidebar.markdown("[Credit Card](#credit-card)")
    st.sidebar.markdown("[PayPal](#paypal)")

    st.sidebar.markdown(" ---------------------- ")


if __name__ == "__main__":
    main()