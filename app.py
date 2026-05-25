from groq import Groq
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page title

st.set_page_config(
    page_title="AI Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI data Analyzer")

st.markdown("""
Upload a CSV file and let AI analyze automatically.
""")

client = Groq(
    api_key=st.secrets["API_KEY"]
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
if uploaded_file:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # Analyze Button
    if st.button("Analyze Equipments"):

        SAFE_LIMIT = 0.605
        TOLERANCE = 0.01

        progress_bar = st.progress(0)

        with st.spinner("Analyzing Values..."):

            Fault = []

            fault_started = False
            fault_start_index = None
            fault_end_index = None
            stabilize_index = None

            values = df["SM2.Pe [HYP1]"]

            total_rows = len(values)

            for index, val in enumerate(values):

                # Fault condition
                if abs(val - SAFE_LIMIT) > TOLERANCE:

                    Fault.append("Yes")

                    if not fault_started:
                        fault_started = True
                        fault_start_index = index

                else:

                    Fault.append("No")

                    # fault ends
                    if fault_started and fault_end_index is None:
                        fault_end_index = index

                # stabilization detection
                if (
                    fault_end_index is not None
                    and stabilize_index is None
                    and abs(val - SAFE_LIMIT) <= TOLERANCE
                ):
                    stabilize_index = index

                progress = (index + 1) / total_rows
                progress_bar.progress(progress)

            df["Fault?"] = Fault

        st.success("✅ Analysis Complete!")

        # ----------------------------
        # TIME ANALYSIS
        # ----------------------------

        if fault_start_index is not None:
            fault_start = fault_start_index
        else:
            fault_start = "No Fault"

        if fault_end_index is not None:
            fault_end = fault_end_index
        else:
            fault_end = "Not Found"

        if (
            isinstance(fault_start, int)
            and isinstance(fault_end, int)
        ):
            stabilization_time = fault_end - fault_start
        else:
            stabilization_time = "Unknown"

        # ----------------------------
        # METRICS
        # ----------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Fault Start",
            fault_start
        )

        col2.metric(
            "Fault End",
            fault_end
        )

        col3.metric(
            "Stabilization Time",
            stabilization_time
        )

        # ----------------------------
        # GRAPH
        # ----------------------------

        st.subheader("📈 SM2.Pe [HYP1] Analysis")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(df["SM2.Pe [HYP1]"])

        ax.axhline(
            y=SAFE_LIMIT,
            linestyle='--'
        )

        ax.set_xlabel("Samples")
        ax.set_ylabel("SM2.Pe [HYP1]")
        ax.set_title("Fault Detection Analysis")

        st.pyplot(fig)

        # ----------------------------
        # AI REPORT
        # ----------------------------

        prompt = f"""
        Generate a professional electrical system analysis report.

        Safe Limit:
        {SAFE_LIMIT}

        Tolerance:
        ±{TOLERANCE}

        Analysis Results:

        Fault Start Index:
        {fault_start}

        Fault End Index:
        {fault_end}

        Stabilization Time:
        {stabilization_time}

        Total Faults:
        {Fault.count("Yes")}

        Maximum Value:
        {df["SM2.Pe [HYP1]"].max()}

        Minimum Value:
        {df["SM2.Pe [HYP1]"].min()}

        Average Value:
        {df["SM2.Pe [HYP1]"].mean()}

        Explain:
        - when fault occurred
        - severity
        - stabilization behavior
        - system condition
        - recommendations
        """

        with st.spinner("Generating AI Report..."):

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            ai_report = (
                response
                .choices[0]
                .message
                .content
            )

        st.subheader("🤖 AI System Report")

        st.write(ai_report)

        # ----------------------------
        # FINAL DATAFRAME
        # ----------------------------

        st.subheader("📋 Analyzed Data")

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name="analyzed_equipment_data.csv",
            mime="text/csv"
        )