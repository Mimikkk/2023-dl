import streamlit as st

from src.mod.st_tensorboard import st_tensorboard

def main():
    st.set_page_config(layout="wide")
    st.title("Main Page")

    st_tensorboard(height=780)

if __name__ == '__main__':
    main()
