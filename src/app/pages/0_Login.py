import streamlit as st
import os
import sys

# =========================================================
# IMPORT AUTH MODULE
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from auth import authenticate_user


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Login - Student Performance System",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .login-title {
        text-align: center;
        font-size: 34px;
        font-weight: bold;
        margin-top: 30px;
    }

    .login-subtitle {
        text-align: center;
        color: #666666;
        font-size: 16px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ALREADY LOGGED IN
# =========================================================

if st.session_state.logged_in:

    user = st.session_state.user

    st.success(
        f"Welcome {user['username']}! "
        f"You are logged in as {user['role']}."
    )

    st.write("Role:", user["role"])
    if user["role"] == "Student":

        if st.button("Open Student Dashboard", use_container_width=True):
            st.switch_page("pages/5_Student_Dashboard.py")

    if st.button("🚪 Logout", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.user = None

        st.rerun()

    st.stop()


# =========================================================
# LOGIN HEADER
# =========================================================

st.markdown(
    '<div class="login-title">🎓 Student Performance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="login-subtitle">'
    'Login to access the Academic Performance Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOGIN FORM
# =========================================================

with st.form("login_form"):

    st.subheader("🔐 User Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    login_button = st.form_submit_button(
        "Login",
        use_container_width=True
    )

    if login_button:

        if not username or not password:

            st.warning("Please enter both username and password.")

        else:

            user = authenticate_user(
                username.strip(),
                password
            )

            if user is not None:

                st.session_state.logged_in = True
                st.session_state.user = user

                st.success("Login successful!")

                st.rerun()

            else:

                st.error("Invalid username or password.")


# =========================================================
# DEMO CREDENTIALS
# =========================================================

with st.expander("View Demo Login Credentials"):

    st.markdown(
        """
        **Admin**
        - Username: `admin`
        - Password: `admin123`

        **Faculty**
        - Username: `faculty`
        - Password: `faculty123`

        **Student**
        - Username: `student`
        - Password: `student123`
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Intelligent Academic Performance Monitoring & Recommendation System"
)