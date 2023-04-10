import streamlit as st


def streamlit_header():
    st.set_page_config(
        page_title="My app",
        page_icon="🧊",
        initial_sidebar_state="auto",
        menu_items={
            # 'Get Help': 'https://www.extremelycoolapp.com/help',
            # 'Report a bug': "https://www.extremelycoolapp.com/bug",
            "About": "Made by https://github.com/matousidc!"
        },
    )
    st.title("My app")
    st.title("Bruh")
    st.markdown("### Pages: Cycling statistics")


if __name__ == "__main__":
    streamlit_header()

# excluded pywin32 from requirements because streamlit cloud runs on linux and thus cant install
# sidebar
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method", ("Standard (5-15 days)", "Express (2-5 days)")
#     )
#     print(add_radio)
