import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="College Data Upload",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from access_control import require_role, logout_button

user = require_role(["Admin", "Faculty"])
logout_button()

# ============================================================
# PAGE CONFIG
# ============================================================

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #172554 0%,
            #1e3a8a 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #172554,
            #1d4ed8,
            #2563eb
        );
        padding: 32px;
        border-radius: 22px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.20);
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin: 0;
        font-weight: 750;
    }

    .hero p {
        color: #dbeafe;
        font-size: 17px;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .section-header {
        background: white;
        padding: 20px 24px;
        border-radius: 16px;
        margin-top: 20px;
        margin-bottom: 18px;
        border-left: 6px solid #2563eb;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.07);
    }

    .section-header h2 {
        margin: 0;
        color: #172554;
        font-size: 25px;
    }

    .section-header p {
        margin: 5px 0 0 0;
        color: #64748b;
        font-size: 14px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 7px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        min-height: 130px;
    }

    .metric-icon {
        font-size: 27px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
    }

    .metric-value {
        color: #172554;
        font-size: 29px;
        font-weight: 750;
        margin-top: 5px;
    }

    .metric-subtitle {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 4px;
    }

    .upload-card {
        background: white;
        padding: 28px;
        border-radius: 20px;
        border: 2px dashed #93c5fd;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    }

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .info-card h3 {
        color: #172554;
        margin-top: 0;
    }

    .info-card p {
        color: #64748b;
        line-height: 1.6;
    }

    .warning-card {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 20px;
        border-radius: 15px;
    }

    .warning-card h3 {
        color: #9a3412;
        margin-top: 0;
    }

    .warning-card p {
        color: #7c2d12;
    }

    .success-card {
        background: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 15px;
    }

    .success-card h3 {
        color: #166534;
        margin-top: 0;
    }

    .success-card p {
        color: #166534;
    }

    .footer {
        text-align: center;
        padding: 30px;
        color: #64748b;
        font-size: 13px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📁 College Data Upload</h1>
        <p>
            Upload student data and generate instant academic analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏫 College Data")

    st.markdown("---")

    st.markdown("### 📂 Upload Settings")

    show_preview = st.checkbox(
        "Show Data Preview",
        value=True
    )

    show_statistics = st.checkbox(
        "Show Statistics",
        value=True
    )

    st.markdown("---")

    st.info(
        "Upload a CSV or XLSX file containing student academic information."
    )

# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📤 Upload Student Dataset</h2>
        <p>
            Select a CSV or Excel file containing your college student records
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel file containing student information."
)

# ============================================================
# SAMPLE DATA OPTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🧪 Demo Dataset</h2>
        <p>
            You can also use the existing project dataset for testing
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

use_project_data = st.checkbox(
    "Use the project academic dataset (student_academic_data.csv)",
    value=False
)

# ============================================================
# LOAD DATA
# ============================================================

data = None
if uploaded_file is not None:

    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".csv":
            data = pd.read_csv(uploaded_file)

        elif file_extension == ".xlsx":
            data = pd.read_excel(uploaded_file, engine="openpyxl")

        else:
            st.error("Unsupported file format. Please upload CSV or XLSX.")
            st.stop()

        st.success(
            f"File uploaded successfully: {uploaded_file.name}"
        )

    except Exception as e:
        st.error("Unable to read the uploaded file.")
        st.code(str(e))
        st.stop()

elif use_project_data:

    # Get the project root directory
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", ".."
        )
    )

    # Set demo dataset path
    project_data_path = os.path.join(
        base_dir,
        "data",
        "student_academic_data.csv"
    )

    try:
        data = pd.read_csv(project_data_path)

        # Convert demo marks to college assessment scale
        demo_mark_conversions = {
            "Internal_Marks": 20,
            "Assignment_Marks": 10,
            "Quiz_Marks": 10
        }

        for column, maximum in demo_mark_conversions.items():
            if column in data.columns:
                data[column] = (
                    pd.to_numeric(data[column], errors="coerce")
                    * maximum / 100
                )

        st.success(
            "Demo academic dataset loaded and converted "
            "to the college assessment scale."
        )

    except Exception as e:
        st.error("Unable to load project dataset.")
        st.exception(e)
# ============================================================
# PROCESS DATA
# ============================================================

if data is not None:

    data = data.copy()

    # --------------------------------------------------------
    # BASIC CLEANING
    # --------------------------------------------------------

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )
        # --------------------------------------------------------
    # COLUMN MAPPING
    # --------------------------------------------------------

    st.markdown("### College Dataset Column Mapping")

    column_options = ["-- Not Mapped --"] + list(data.columns)

    field_aliases = {
        "student_id": ["student_id", "studentid", "roll_no", "roll_number"],
        "name": ["name", "student_name", "studentname"],
        "department": ["department", "branch", "course"],
        "semester": ["semester", "sem"],
        "attendance": ["attendance", "attendance_percentage"],
        "internal_marks": ["internal_marks", "internal"],
        "assignment_marks": ["assignment_marks", "assignment"],
        "quiz_marks": ["quiz_marks", "quiz"],
        "study_hours": ["study_hours", "study_time"],
        "previous_cgpa": ["previous_cgpa", "prev_cgpa"],
        "current_cgpa": ["current_cgpa", "cgpa"],
        "backlogs": ["backlogs", "failures"],
        "assignment_submission_rate": [
            "assignment_submission_rate",
            "submission_rate"
        ],
        "learning_activity": ["learning_activity", "activity"],
        "final_marks": ["final_marks", "final_score"]
    }

    column_mapping = {}

    with st.form("column_mapping_form"):

        st.write("Match your college file columns with Trekverse fields.")

        for target_field, aliases in field_aliases.items():

            default_column = next(
                (
                    column for column in aliases
                    if column in data.columns
                ),
                "-- Not Mapped --"
            )

            default_index = column_options.index(default_column)

            selected_column = st.selectbox(
                target_field.replace("_", " ").title(),
                column_options,
                index=default_index,
                key=f"mapping_{target_field}"
            )

            column_mapping[target_field] = selected_column

        mapping_submitted = st.form_submit_button(
            "Apply Column Mapping"
        )

    if mapping_submitted:

        selected_columns = [
            value for value in column_mapping.values()
            if value != "-- Not Mapped --"
        ]

        if len(selected_columns) != len(set(selected_columns)):

            st.error("The same source column cannot be mapped to multiple fields.")

        elif column_mapping["student_id"] == "-- Not Mapped --":

            st.error("Student ID mapping is required.")

        else:

            rename_mapping = {
                source: target
                for target, source in column_mapping.items()
                if source != "-- Not Mapped --"
            }

            data = data.rename(columns=rename_mapping)

            st.session_state["mapped_student_data"] = data.copy()

            st.success("Column mapping applied successfully!")

            st.dataframe(
                data.head(10),
                use_container_width=True,
                hide_index=True
            )

    elif "mapped_student_data" in st.session_state:

        data = st.session_state["mapped_student_data"].copy()
        # --------------------------------------------------------
    # STUDENT DATA VALIDATION
    # --------------------------------------------------------

    validation_errors = []

    # Check required Student_ID column
    if "student_id" not in data.columns:

        validation_errors.append({
            "Row": "All",
            "Column": "student_id",
            "Error": "Required Student_ID column is missing"
        })

    else:

        # Check missing Student IDs
        missing_ids = data["student_id"].isna() | (
            data["student_id"].astype(str).str.strip() == ""
        )

        for index in data[missing_ids].index:
            validation_errors.append({
                "Row": index + 2,
                "Column": "student_id",
                "Error": "Student ID is missing"
            })

        # Check duplicate Student IDs
        duplicate_ids = (
            data["student_id"].notna()
            & data["student_id"].duplicated(keep=False)
        )

        for index in data[duplicate_ids].index:
            validation_errors.append({
                "Row": index + 2,
                "Column": "student_id",
                "Error": "Duplicate Student ID"
            })

    # Validate numeric fields when available
    numeric_rules = {
    "attendance": (0, 100),
    "internal_marks": (0, 20),
    "assignment_marks": (0, 10),
    "quiz_marks": (0, 10),
    "study_hours": (0, None),
    "previous_cgpa": (0, 10),
    "current_cgpa": (0, 10),
    "backlogs": (0, None),
    "final_marks": (0, 100)
}
    # --------------------------------------------------------
# DEPARTMENT VALIDATION
# --------------------------------------------------------

if data is not None and "department" in data.columns:

    allowed_departments = [
        "CSE-AI",
        "Computer Science",
        "ECE",
        "Information Technology"
    ]

    invalid_departments = (
        data["department"].notna()
        & ~data["department"].astype(str).str.strip().isin(
            allowed_departments
        )
    )

    for index in data[invalid_departments].index:
        validation_errors.append({
            "Row": index + 2,
            "Column": "department",
            "Error": "Invalid department name"
        })


# --------------------------------------------------------
# SEMESTER VALIDATION
# --------------------------------------------------------

if data is not None and "semester" in data.columns:
    semester_values = pd.to_numeric(
        data["semester"],
        errors="coerce"
    )

    allowed_semesters = [5, 6, 7, 8]

    invalid_semesters = (
        semester_values.notna()
        & ~semester_values.isin(allowed_semesters)
    )

    for index in data[invalid_semesters].index:
        validation_errors.append({
            "Row": index + 2,
            "Column": "semester",
            "Error": "Semester must be 5, 6, 7 or 8"
        })
    # --------------------------------------------------------
# BACKLOGS WHOLE NUMBER VALIDATION
# --------------------------------------------------------

if data is not None and "backlogs" in data.columns:

    backlog_values = pd.to_numeric(
        data["backlogs"],
        errors="coerce"
    )

    invalid_backlogs = (
        backlog_values.notna()
        & (backlog_values % 1 != 0)
    )

    for index in data[invalid_backlogs].index:
        validation_errors.append({
            "Row": index + 2,
            "Column": "backlogs",
            "Error": "Backlogs must be a whole number"
        })

    for column, limits in numeric_rules.items():

        if column not in data.columns:
            continue

        numeric_values = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        invalid_numeric = data[column].notna() & numeric_values.isna()

        for index in data[invalid_numeric].index:
            validation_errors.append({
                "Row": index + 2,
                "Column": column,
                "Error": "Value must be numeric"
            })

        minimum, maximum = limits

        invalid_range = numeric_values.notna() & (
            (numeric_values < minimum)
        )

        if maximum is not None:
            invalid_range |= (
                numeric_values.notna()
                & (numeric_values > maximum)
            )

        for index in data[invalid_range].index:
            validation_errors.append({
                "Row": index + 2,
                "Column": column,
                "Error": f"Value must be between {minimum} and {maximum}"
                if maximum is not None
                else f"Value must be at least {minimum}"
            })

    # Display validation results
    st.markdown("### Student Data Validation")
        # Student Name is also required for database import.
    if "name" in data.columns:
        name_column = "name"
    elif "student_name" in data.columns:
        name_column = "student_name"
    else:
        name_column = None

    if name_column is None:
        validation_errors.append({
            "Row": "All",
            "Column": "name",
            "Error": "Required Student Name column is missing"
        })
    else:
        missing_names = (
            data[name_column].isna()
            | (data[name_column].astype(str).str.strip() == "")
        )

        for index in data[missing_names].index:
            validation_errors.append({
                "Row": index + 2,
                "Column": name_column,
                "Error": "Student Name is missing"
            })

    valid_error_count = len(validation_errors)

    has_global_error = any(
        error["Row"] == "All" for error in validation_errors
    )

    valid_records = (
        0 if has_global_error
        else len(data) - len({
            error["Row"]
            for error in validation_errors
            if isinstance(error["Row"], int)
        })
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))
    col2.metric("Valid Records", valid_records)
    col3.metric("Validation Errors", valid_error_count)

    if validation_errors:

        st.warning("Some records need correction.")

        st.dataframe(
            pd.DataFrame(validation_errors),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success("All student records passed the initial validation checks.")
        # ========================================================
    # DOWNLOAD VALIDATION ERROR REPORT
    # ========================================================

    if validation_errors:

        st.markdown("### Download Validation Error Report")

        error_report = pd.DataFrame(validation_errors)

        csv_report = error_report.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="Download Validation Error Report",
            data=csv_report,
            file_name="Trekverse_Validation_Errors.csv",
            mime="text/csv",
            key="download_validation_errors"
        )

        st.info(
            "Download this report, correct the errors in your "
            "original dataset, and upload the corrected file."
        )    
        # ========================================================
    # IMPORT VALID RECORDS INTO DATABASE
    # ========================================================

    st.markdown("### Import Student Records into Database")

    invalid_row_numbers = {
        error["Row"]
        for error in validation_errors
        if isinstance(error["Row"], int)
    }

    valid_data = data.loc[
        [
            index for index in data.index
            if index + 2 not in invalid_row_numbers
        ]
    ].copy()

    if has_global_error:
        valid_data = data.iloc[0:0].copy()

    st.write(f"Records ready for import: **{len(valid_data)}**")

    if len(valid_data) == 0:
        st.warning("No valid records available for database import.")

    else:

        consent_confirmed = st.checkbox(
            "I confirm that the college is authorized to process "
            "these student records for academic purposes.",
            key="college_data_consent"
        )

        if st.button(
            "Import Valid Records to Database",
            type="primary",
            key="import_college_data",
            disabled=not consent_confirmed
        ):
            try:
                from database import import_students_from_dataframe

                result = import_students_from_dataframe(valid_data)

                st.success("Student data imported successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric("New Records", result["inserted"])
                col2.metric("Updated Records", result["updated"])
                col3.metric("Skipped Records", result["skipped"])

            except Exception as e:
                st.error("Database import failed.")
                st.exception(e)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_students = len(data)

    total_columns = len(data.columns)

    missing_values = int(
        data.isnull().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    # --------------------------------------------------------
    # DATA OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-header">
            <h2>📊 Dataset Overview</h2>
            <p>
                Basic information about the uploaded college dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">👨‍🎓</div>
                <div class="metric-title">Students</div>
                <div class="metric-value">{total_students}</div>
                <div class="metric-subtitle">Student records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📋</div>
                <div class="metric-title">Features</div>
                <div class="metric-value">{total_columns}</div>
                <div class="metric-subtitle">Dataset columns</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">❓</div>
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{missing_values}</div>
                <div class="metric-subtitle">Total missing cells</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🔁</div>
                <div class="metric-title">Duplicates</div>
                <div class="metric-value">{duplicate_rows}</div>
                <div class="metric-subtitle">Duplicate records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            <h2>🔎 Data Quality Check</h2>
            <p>
                Automatically check the uploaded dataset before analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if missing_values == 0 and duplicate_rows == 0:

        st.markdown(
            """
            <div class="success-card">
                <h3>✅ Dataset Looks Good</h3>
                <p>
                    No missing values or duplicate records were detected.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        if missing_values > 0:

            st.warning(
                f"⚠️ The dataset contains {missing_values} missing values."
            )

        if duplicate_rows > 0:

            st.warning(
                f"⚠️ The dataset contains {duplicate_rows} duplicate records."
            )

    # ========================================================
    # DATA PREVIEW
    # ========================================================

    if show_preview:

        st.markdown(
            """
            <div class="section-header">
                <h2>👀 Data Preview</h2>
                <p>
                    Preview the uploaded student records
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            data.head(20),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # NUMERICAL STATISTICS
    # ========================================================

    numerical_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if show_statistics and len(numerical_columns) > 0:

        st.markdown(
            """
            <div class="section-header">
                <h2>📈 Numerical Statistics</h2>
                <p>
                    Summary statistics for numerical student attributes
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        statistics = (
            data[numerical_columns]
            .describe()
            .T
        )

        st.dataframe(
            statistics,
            use_container_width=True
        )

    # ========================================================
    # ACADEMIC PERFORMANCE ANALYSIS
    # ========================================================

    score_column = (
        "final_marks" if "final_marks" in data.columns
        else "g3" if "g3" in data.columns
        else None
    )

    if score_column is not None:
        score_values = pd.to_numeric(data[score_column], errors="coerce").dropna()

        if not score_values.empty:
            st.markdown(
                """
                <div class="section-header">
                    <h2>🎯 Academic Performance</h2>
                    <p>Summary of student final academic scores</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            p1, p2, p3 = st.columns(3)
            p1.metric("Average Score", f"{score_values.mean():.2f}")
            p2.metric("Highest Score", f"{score_values.max():.2f}")
            p3.metric("Lowest Score", f"{score_values.min():.2f}")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(score_values, bins=10)
            ax.set_xlabel("Final Marks")
            ax.set_ylabel("Number of Students")
            ax.set_title("Student Final Marks Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ========================================================
    # ATTENDANCE ANALYSIS
    # ========================================================

    attendance_column = (
        "attendance" if "attendance" in data.columns
        else "absences" if "absences" in data.columns
        else None
    )

    if attendance_column is not None and score_column is not None:
        plot_data = data[[attendance_column, score_column]].copy()
        plot_data[attendance_column] = pd.to_numeric(
            plot_data[attendance_column], errors="coerce"
        )
        plot_data[score_column] = pd.to_numeric(
            plot_data[score_column], errors="coerce"
        )
        plot_data = plot_data.dropna()

        if not plot_data.empty:
            st.markdown(
                """
                <div class="section-header">
                    <h2>🏫 Attendance Analysis</h2>
                    <p>Relationship between attendance and academic scores</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(
                plot_data[attendance_column],
                plot_data[score_column],
                alpha=0.7
            )
            ax.set_xlabel(
                "Attendance (%)" if attendance_column == "attendance"
                else "Absences"
            )
            ax.set_ylabel("Final Marks")
            ax.set_title("Attendance vs Final Marks")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ========================================================
    # STUDENT RISK OVERVIEW
    # ========================================================

    if "risk_level" in data.columns:
        st.markdown(
            """
            <div class="section-header">
                <h2>⚠️ Student Risk Overview</h2>
                <p>Risk categories found in the uploaded student records</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        risk_counts = (
            data["risk_level"].fillna("Unknown").astype(str).value_counts()
        )
        st.bar_chart(risk_counts)

        risk_display_columns = [
            column for column in [
                "student_id", "name", "student_name",
                "department", "semester", "current_cgpa", "risk_level"
            ]
            if column in data.columns
        ]

        high_risk = data[
            data["risk_level"].astype(str).str.strip().str.lower().isin(
                ["high", "high risk"]
            )
        ]

        if not high_risk.empty and risk_display_columns:
            st.warning(
                f"{len(high_risk)} student(s) are marked as high risk."
            )
            st.dataframe(
                high_risk[risk_display_columns],
                use_container_width=True,
                hide_index=True
            )

    elif score_column is not None:
        # Legacy fallback when an uploaded file has no risk_level column.
        risk_students = data[
            pd.to_numeric(data[score_column], errors="coerce") < 10
        ].copy()

        if not risk_students.empty:
            st.markdown("### ⚠️ Students Requiring Academic Attention")
            st.dataframe(
                risk_students.head(50),
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            <h2>💾 Export Data</h2>
            <p>
                Download the uploaded dataset after processing
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    csv_data = data.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Download Processed CSV",
        data=csv_data,
        file_name="processed_student_data.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div class="upload-card">
            <h2>📂 No Dataset Uploaded</h2>
            <p>
                Upload a CSV or Excel file above to start analysing college
                student performance.
            </p>
            <br>
            <p>
                <strong>Recommended columns:</strong>
                student information, study time, absences,
                failures, and academic scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>⚙️ How It Works</h2>
        <p>
            College data analysis workflow
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <h3>1️⃣ Upload</h3>
            <p>
                Faculty uploads the college student CSV dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="info-card">
            <h3>2️⃣ Validate</h3>
            <p>
                The system checks missing values, duplicates,
                and available columns.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class="info-card">
            <h3>3️⃣ Analyse</h3>
            <p>
                The system generates statistics, charts,
                performance groups, and risk indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
        <div class="info-card">
            <h3>4️⃣ Export</h3>
            <p>
                Faculty can download the processed student data.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        📁 College Data Management & Analytics
        <br>
        Machine Learning • Data Analytics • Streamlit
        <br><br>
        Trekverse Internship Project
    </div>
    """,
    unsafe_allow_html=True
)