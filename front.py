import streamlit as st
import pandas as pd
import joblib
from predict import load_results, predict


def create_front():
    st.title("Satisfaction poll")

    if 'form_visible' not in st.session_state:
        st.session_state['form_visible'] = True

    if 'result_visible' not in st.session_state:
        st.session_state['result_visible'] = False

    if st.session_state['form_visible']:
        st.session_state['customer_type'] = (
            st.selectbox("Customer type:", options=["Loyal Customer", "disloyal Customer"]))
        st.session_state['type_of_travel'] = (
            st.radio("Type of travel:", options=["Business travel", "Personal Travel"],
                                                      horizontal=True))
        st.session_state['class'] = (
            st.radio("Class:",options=["Business", "Eco", "Eco Plus"], horizontal=True))
        st.session_state['gender'] = st.radio("Gender:", options=["Male", "Female"],
                                              horizontal=True)
        st.session_state['age'] = st.slider("Age:", 5, 100, 30)
        st.session_state['flight_distance'] = (
            st.number_input("Flight distance (km):", min_value=0, max_value=15000))
        st.session_state['delay_sum'] = (
            st.number_input("Delay (minutes):", min_value=0, max_value=3000))
        st.session_state['survey_sum'] = (
            st.number_input("Survey sum:", min_value=0, max_value=100, value=50))

        st.button("Send!", on_click=load_results)

    if st.session_state['result_visible']:
        query = pd.DataFrame({
            'customer_type': st.session_state['customer_type'],
            'type_of_travel': st.session_state['type_of_travel'],
            'class': st.session_state['class'],
            'gender': st.session_state['gender'],
            'age': st.session_state['age'],
            'flight_distance': st.session_state['flight_distance'],
            'delay_sum': st.session_state['delay_sum'],
            'survey_sum': st.session_state['survey_sum']
        }, index=[0]
        )

        predict('model.pkl', query)
