import streamlit as st
import time
import joblib


def load_results():
    st.session_state['form_visible'] = False
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.session_state['result_visible'] = True


def predict(file, query):
    model = joblib.load(file)
    result = model.predict(query)[0]
    if result == "satisfied":
        st.markdown(f"<div align='center'><b>Result of prediction: <span style='color: green;'>"
                    f"Satisfied</span></b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div align='center'><b>Result of prediction: <span style='color: yellow;'>"
                    f"Neutral or Dissatisfied</span></b></div>", unsafe_allow_html=True)
