import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Premium Page Setup ---
st.set_page_config(
    page_title="Student Performance Prediction Based on Grades",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded" # Expanded by default to show sidebar metadata
)

# --- Premium Custom CSS Styling (Dark Slate, Glassmorphism, tailored HSL gradients) ---
st.markdown("""
<style>
    /* Overall Background */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                          radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 45%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0e1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-bottom: 2px solid #818cf8 !important;
    }

    /* Title & Headers */
    .dashboard-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 60%, #fbbf24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-align: center;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.2rem;
        text-align: center;
    }
    
    /* Glassmorphic Cards & Form Panel */
    .input-panel {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: rgba(22, 28, 45, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .kpi-title {
        color: #94a3b8;
        text-transform: uppercase;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
    }
    
    .kpi-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        line-height: 1.1;
    }
    
    .kpi-grade {
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 0 25px currentColor;
    }
    
    .kpi-risk {
        font-size: 1.3rem;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        display: inline-block;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Dynamic & Cached Data Loader ---
@st.cache_data
def load_and_preprocess_data():
    csv_path = os.path.join(BASE_DIR, "student_habits_performance.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Preprocess Part-Time Job
    if 'part_time_job' in df.columns:
        df['part_time_job'] = df['part_time_job'].fillna('No').astype(str).str.strip().str.capitalize()
    else:
        df['part_time_job'] = 'No'
        
    return df

# --- Load Pre-trained Model ---
@st.cache_resource
def load_fyp_model():
    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading pre-trained model: {str(e)}")
        return None

# --- Load Data and Models ---
df = load_and_preprocess_data()
model = load_fyp_model()

# ========================================================
# MINIMALIST SIDEBAR PROJECT INFO
# ========================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
    <h2 style="color: #f8fafc; margin: 0; font-size: 1.4rem; font-weight: 800;">🎓 EduPredict</h2>
    <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Student Performance Prediction Based on Grades</p>
</div>
<hr style="border-color: rgba(255,255,255,0.08); margin: 0.5rem 0 1.5rem 0;">
<div style="padding: 0 1rem; color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
    <strong>Model Architecture:</strong><br>
    Linear Regression Pipeline<br><br>
    <strong>Key Features Analyzed:</strong><br>
    • Daily Study Target<br>
    • Cohort Attendance<br>
    • Sleep Schedule<br>
    • Stress Indices<br>
    • Employment Balances
</div>
""", unsafe_allow_html=True)

# ========================================================
# MAIN CONTENT HEADER
# ========================================================
st.markdown('<div style="text-align: center; color: #818cf8; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem; margin-top: 0.5rem;">🔮 Welcome to EduPredict Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-title">🎓 Student Performance Prediction Based on Grades</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Final Year Project: Predictive Student Performance System</div>', unsafe_allow_html=True)

if df is not None and model is not None:
    # ========================================================
    # LANDING PAGE GLASSMORPHIC INPUT CARD
    # ========================================================
    with st.container(border=True):
        st.markdown("### 📝 Student Attributes Input Console")
        st.write("Enter the student profile parameters to execute the predictive engine:")
        
        # 3-column input layout directly on the main screen
        col_in1, col_in2, col_in3 = st.columns(3, gap="medium")
        
        with col_in1:
            study_hours = st.number_input("📚 Daily Study Hours (0 - 12)", min_value=0.0, max_value=12.0, value=0.0, step=0.5)
            attendance = st.number_input("🏫 Attendance Percentage (0 - 100)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            
        with col_in2:
            sleep_hours = st.number_input("😴 Sleep Hours / Night (0 - 12)", min_value=0.0, max_value=12.0, value=0.0, step=0.5)
            mental_health = st.number_input("🧠 Mental Health Rating (0 - 10)", min_value=0, max_value=10, value=0, step=1, help="0: Low Stress/Poor, 10: Excellent")
            
        with col_in3:
            part_time_job_str = st.selectbox("💼 Has Part-Time Job?", options=["No", "Yes"])
            st.markdown("<br>", unsafe_allow_html=True)
            # Prediction execution button
            submit_button = st.button("🔮 Run Diagnostic Analysis", use_container_width=True, type="primary")

    if submit_button:
        st.session_state['show_results'] = True

    if st.session_state.get('show_results', False):
        
        part_time_job_val = 1 if part_time_job_str == "Yes" else 0
        input_vector = pd.DataFrame([{
            'study_hours_per_day': study_hours,
            'attendance_percentage': attendance,
            'mental_health_rating': mental_health,
            'sleep_hours': sleep_hours,
            'part_time_job': part_time_job_val
        }])
        
        # Calculate Prediction Score
        pred_score = float(model.predict(input_vector)[0])
        pred_score = min(max(pred_score, 0.0), 100.0) # bound
        
        # Determine Grades and Colors
        if pred_score >= 85:
            grade, color, risk_tier, risk_color, risk_bg = "A", "#10b981", "HONORS CLASS", "#10b981", "rgba(16, 185, 129, 0.15)"
        elif pred_score >= 70:
            grade, color, risk_tier, risk_color, risk_bg = "B", "#3b82f6", "STABLE / LOW RISK", "#3b82f6", "rgba(59, 130, 246, 0.15)"
        elif pred_score >= 55:
            grade, color, risk_tier, risk_color, risk_bg = "C", "#f59e0b", "MODERATE RISK", "#f59e0b", "rgba(245, 158, 11, 0.15)"
        elif pred_score >= 40:
            grade, color, risk_tier, risk_color, risk_bg = "D", "#f97316", "ACADEMIC WARNING", "#f97316", "rgba(249, 115, 22, 0.15)"
        else:
            grade, color, risk_tier, risk_color, risk_bg = "F", "#ef4444", "INTERVENTION REQUIRED", "#ef4444", "rgba(239, 68, 68, 0.15)"

        # --- KPI Grid Top Row ---
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3, gap="large")
        
        with col_kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Projected Exam Score</div>
                <div class="kpi-value" style="background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{pred_score:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Calculated Letter Grade</div>
                <div class="kpi-grade" style="color: {color};">{grade}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Diagnostic Risk Status</div>
                <div class="kpi-risk" style="color: {risk_color}; background-color: {risk_bg}; border: 1px solid {risk_color};">
                    {risk_tier}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Dashboard Navigation Tabs ---
        tab_diagnostics, tab_recommendations = st.tabs([
            "📊 Dynamic Diagnostics & Visual Cohorts",
            "💡 Smart Intervention Advisories"
        ])
        
        # ----------------------------------------------------
        # TAB 1: Visual Diagnostics
        # ----------------------------------------------------
        with tab_diagnostics:
            st.markdown("### 📊 Live Diagnostic Charting (Relative to Result)")
            
            # --- Dynamic Chart 1: Student vs. Cohort Metrics (Relative to prediction) ---
            col_bar1, col_bar2 = st.columns(2, gap="large")
            
            with col_bar1:
                st.markdown("##### 🏆 Predicted Score vs Cohort Benchmarks")
                cohort_avg_score = df['exam_score'].mean()
                
                fig_compare = go.Figure(go.Bar(
                    x=[pred_score, cohort_avg_score, 50.0],
                    y=['Predicted Student', 'Cohort Average', 'Passing Mark'],
                    orientation='h',
                    marker=dict(
                        color=[color, '#475569', '#ef4444'],
                        line=dict(color='rgba(255,255,255,0.1)', width=1)
                    )
                ))
                fig_compare.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0, 100], title="Score (%)"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=250
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            with col_bar2:
                st.markdown("##### 🧬 Student Gap Analysis (vs. High A-Grade Students)")
                # Calculate mean characteristics of A-grade students (Score >= 85)
                a_students = df[df['exam_score'] >= 85.0]
                a_study = a_students['study_hours_per_day'].mean()
                a_attend = a_students['attendance_percentage'].mean()
                
                # Deficit calculations
                study_gap = study_hours - a_study
                attend_gap = attendance - a_attend
                
                fig_gap = go.Figure(go.Bar(
                    x=[study_gap, attend_gap],
                    y=['Study Hours Gap', 'Attendance Gap %'],
                    orientation='h',
                    marker=dict(
                        color=['#ef4444' if study_gap < 0 else '#10b981', '#ef4444' if attend_gap < 0 else '#10b981'],
                    )
                ))
                fig_gap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Deficit / Surplus Value"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=250
                )
                st.plotly_chart(fig_gap, use_container_width=True)

            # --- Distribution Analysis Chart ---
            col_plot1, col_plot2 = st.columns([1.1, 0.9], gap="large")
            
            with col_plot1:
                st.markdown("##### 📈 Cohort Distribution & Student Position")
                fig_hist = px.histogram(
                    df, 
                    x="exam_score", 
                    nbins=40,
                    labels={"exam_score": "Examination Score (%)"},
                    template="plotly_dark",
                    color_discrete_sequence=["rgba(99, 102, 241, 0.6)"]
                )
                fig_hist.add_vline(
                    x=pred_score, 
                    line_width=3, 
                    line_dash="dash", 
                    line_color="#ef4444",
                    annotation_text=f"Current Prediction ({pred_score:.1f}%)",
                    annotation_position="top right",
                    annotation_font=dict(color="#f8fafc", size=11, family="Inter")
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Count of Students",
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_plot2:
                st.markdown("##### 🌌 3D Student Space Explorer")
                sample_df = df.sample(n=400, random_state=42)
                fig_3d = px.scatter_3d(
                    sample_df,
                    x="study_hours_per_day",
                    y="attendance_percentage",
                    z="exam_score",
                    color="exam_score",
                    color_continuous_scale="Viridis",
                    labels={
                        "study_hours_per_day": "Study Hours",
                        "attendance_percentage": "Attendance %",
                        "exam_score": "Score"
                    },
                    template="plotly_dark"
                )
                fig_3d.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    scene=dict(
                        xaxis_backgroundcolor="rgba(0,0,0,0)",
                        yaxis_backgroundcolor="rgba(0,0,0,0)",
                        zaxis_backgroundcolor="rgba(0,0,0,0)"
                    )
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        # ----------------------------------------------------
        # TAB 2: Smart Interventions
        # ----------------------------------------------------
        with tab_recommendations:
            st.markdown("### 💡 AI Clinical Advisories & Recommendations")
            
            if grade in ["C", "D", "F"]:
                st.markdown(f"""
                <div style="background-color: rgba(245, 158, 11, 0.08); border-left: 5px solid #f59e0b; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
                    <h4 style="color: #f59e0b; margin: 0 0 0.5rem 0; font-weight: 700;">⚠️ Academic Support Warning Required</h4>
                    <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
                        This student profile has been categorised as <strong>{risk_tier}</strong>. Implementing immediate interventions in the highlighted lifestyle categories can prevent academic deterioration and target grade improvement.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(16, 185, 129, 0.08); border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
                    <h4 style="color: #10b981; margin: 0 0 0.5rem 0; font-weight: 700;">🎉 Academic Excellence Maintained</h4>
                    <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
                        The student has demonstrated highly disciplined lifestyle indices resulting in a projected <strong>Grade {grade}</strong>. Focus on reinforcement, peer leadership development, and higher honors tracks.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            col_rec1, col_rec2 = st.columns(2, gap="large")
            
            with col_rec1:
                st.markdown("#### 📚 Strategic Study Interventions")
                if study_hours < 5.0:
                    st.markdown(f"✅ **Increase Daily Study Goal:** Current study hours (`{study_hours}h/day`) are below the high-performing class average. Target study hours of **6.0+ hours** to achieve an estimated **+7.5%** exam score improvement.")
                else:
                    st.markdown("✅ **Study discipline maintained:** Current daily study allocation is optimal. Continue reinforcing this structure with qualitative mock assessments.")
                    
                if attendance < 85.0:
                    st.markdown(f"✅ **Raise Attendance Minimums:** Current attendance is at `{attendance}%`. Boosting attendance to **90%+** is highly vital to regain connection to underlying core course materials.")
                else:
                    st.markdown("✅ **Excellent Attendance Record:** The student shows outstanding class involvement. This ensures perfect coverage of syllabus milestones.")

            with col_rec2:
                st.markdown("#### 🧠 Cognitive & Stress Interventions")
                if sleep_hours < 7.0:
                    st.markdown(f"✅ **Maximize Rest Durations:** The student gets only `{sleep_hours} hours` of sleep per night. Target **7.5 to 8.5 hours** of nightly rest. Insufficient rest acts as a severe force-multiplier for low exam scores due to impaired recollection cycles.")
                else:
                    st.markdown("✅ **Optimal Sleep Pattern:** Sleep schedule is healthy and aligns with high cognitive function. Excellent job.")
                    
                if mental_health <= 4:
                    st.markdown("✅ **Introduce Stress Counseling:** Mental health indicator is critical. Suggest routine stress management seminars or counseling to help cope with final year evaluation workloads.")
                else:
                    st.markdown("✅ **Healthy Mental Baseline:** Mental fitness scores are excellent. Keep encouraging balanced student lifestyles.")
    
        # --- Page Footer ---
        st.markdown("""
        <hr style="border-color: rgba(255,255,255,0.08); margin-top: 3rem;">
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 2rem;">
            🎓 Student Performance Prediction Based on Grades • Final Year Project Suite • © 2026 All Rights Reserved
        </div>
        """, unsafe_allow_html=True)
    else:
        # Prompt user to run the dashboard
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; padding: 3rem; background: rgba(22, 28, 45, 0.4); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 16px;">
            <h2 style="color: #cbd5e1; margin-bottom: 0.5rem; font-weight: 700;">🔮 Welcome to EduPredict Analytics</h2>
            <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto 0 auto;">
                Input a student profile in the console above and click <strong>"Run Diagnostic Analysis"</strong> to generate the predictive report and result-relative visual charts.
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("💡 Ensure student_habits_performance.csv and best_model.pkl exist in the directory to activate predictions.")
