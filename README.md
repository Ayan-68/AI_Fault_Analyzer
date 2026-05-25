# 🤖 AI Equipment Data Analyzer

An AI-powered electrical equipment fault analysis system built with Python, Streamlit, Pandas, Matplotlib, and Groq LLMs.

This application allows users to upload CSV datasets containing electrical measurements and automatically:

- Detect faults
- Analyze stabilization behavior
- Generate AI-powered system reports
- Visualize equipment performance
- Export analyzed results

---

# 🚀 Features

## ✅ CSV Upload
Upload electrical equipment datasets directly from the browser.

## ✅ Automatic Fault Detection
Detects abnormal values in:

- `SM2.Pe [HYP1]`

using a configurable safe limit and tolerance range.

## ✅ AI-Generated Reports
Uses Groq LLMs to generate professional analysis reports including:

- Fault occurrence
- Fault end
- Stabilization behavior
- Severity analysis
- Recommendations

## ✅ Interactive Visualization
Displays graphs for fault analysis and system behavior.

## ✅ Download Results
Export analyzed datasets with fault labels as CSV.

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Groq API
- Llama 3.1

---

# 📂 Project Structure

```bash
project/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_data.csv
```

---

# ⚙ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-equipment-analyzer.git
```

## 2. Navigate to Project

```bash
cd ai-equipment-analyzer
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Setup Groq API Key

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY="your_api_key_here"
```

OR use environment variables:

```bash
export GROQ_API_KEY="your_api_key_here"
```

---

# ▶ Run Application

```bash
streamlit run app.py
```

---

# 📊 Fault Detection Logic

The system uses:

```python
SAFE_LIMIT = 0.605
TOLERANCE = 0.01
```

Fault condition:

```python
abs(value - SAFE_LIMIT) > TOLERANCE
```

---

# 📈 AI Report Includes

- Fault start point
- Fault end point
- Stabilization analysis
- Maximum deviation
- Minimum deviation
- Average system behavior
- Recommendations

---

# 📥 Input CSV Requirements

Your CSV file should contain:

| Column Name |
|---|
| SM2.Pe [HYP1] |

Example:

```csv
SM2.Pe [HYP1]
0.605
0.604
0.590
0.620
```

---

# 📤 Output

The application generates:

- Fault classification column
- Visual graphs
- AI-generated report
- Downloadable analyzed CSV

---

# 🔒 Security Note

Never hardcode API keys directly into source code.

Use:

- Streamlit Secrets
- Environment Variables

instead.

---

# 🧠 Future Improvements

- Multi-parameter fault analysis
- Real-time monitoring
- Advanced AI diagnostics
- Predictive maintenance
- Dashboard analytics
- PDF report generation

---

# 👨‍💻 Author

Developed by Ayan Shaikj

---

# 📜 License

This project is licensed under the MIT License.