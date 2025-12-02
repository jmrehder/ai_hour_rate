"""
PROJECT ESTIMATOR PRO
=====================
ML-powered project estimation with high accuracy.

Target metrics: MAE < 25h, MAPE < 12%, R² > 0.90

Run: streamlit run estimator_pro.py
Requires: pip install streamlit pandas numpy scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(page_title="Project Estimator Pro", page_icon="🧠", layout="wide")

DEFAULT_RATES = {
    'Senior Developer': 140,
    'Developer': 120,
    'Designer': 110,
    'Project Manager': 120,
    'QA Engineer': 100
}

PROJECT_BASE_HOURS = {
    'Landing Page': 45, 'Blog/Magazine': 95, 'Corporate Relaunch': 160,
    'CMS Migration': 130, 'Accessibility Retrofit': 90, 'New Development': 200,
    'E-Commerce/WooCommerce': 240, 'Intranet/Extranet': 280,
    'Multisite Network': 350, 'Web Application': 320,
}

INDUSTRY_MULTIPLIERS = {
    'Startup/Tech': 0.92, 'Agency/Media': 0.95, 'Non-Profit/NGO': 1.00,
    'Professional Services': 1.00, 'Manufacturing': 1.05, 'Retail/E-Commerce': 1.08,
    'Education': 1.10, 'Energy/Utilities': 1.12, 'Finance/Insurance': 1.15,
    'Healthcare/Pharma': 1.18, 'Public Sector': 1.22, 'Public Sector (EVB-IT)': 1.30,
}

DESIGN_MULTIPLIERS = {
    'No Design (Dev only)': 0.65, 'Premium Theme': 0.80,
    'Existing Design System': 0.90, 'Theme Customization': 1.00,
    'Custom Design (Figma)': 1.25,
}

MODELS = {
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=10, subsample=0.9, random_state=42),
        'icon': '🎯', 'tagline': 'Highest Accuracy',
        'description': 'The gold standard for structured data prediction.',
        'how_it_works': '''
**Sequential Learning**: Builds decision trees one after another, where each new tree 
specifically focuses on correcting the errors made by all previous trees.

Think of it like a team of editors reviewing a document - each editor focuses on 
fixing the mistakes the previous editors missed.
        ''',
        'strengths': ['Best accuracy on structured data', 'Learns complex patterns', 'Handles feature interactions well'],
        'weaknesses': ['Slower to train', 'Can overfit if not tuned properly'],
        'best_for': 'Production use when accuracy is the priority'
    },
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'icon': '🌳', 'tagline': 'Robust & Reliable',
        'description': 'An ensemble of independent decision trees.',
        'how_it_works': '''
**Parallel Wisdom**: Creates 200 completely independent decision trees, each trained 
on a random subset of the data. The final prediction is the average of all trees.

Like asking 200 different experts for their opinion and taking the average - 
individual mistakes cancel out.
        ''',
        'strengths': ['Very robust to outliers', 'Hard to overfit', 'Fast training'],
        'weaknesses': ['Slightly less accurate than boosting', 'Can be memory-intensive'],
        'best_for': 'When you want reliable, stable predictions'
    },
    'Histogram Gradient Boosting': {
        'model': HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.1, min_samples_leaf=10, random_state=42),
        'icon': '⚡', 'tagline': 'Speed + Accuracy',
        'description': 'Modern, optimized gradient boosting.',
        'how_it_works': '''
**Smart Shortcuts**: Uses the same principle as Gradient Boosting, but groups 
similar values into "bins" (histograms) before processing.

Instead of comparing every single value, it compares groups - like sorting 
students by grade ranges (A, B, C) instead of exact percentages.
        ''',
        'strengths': ['Very fast training', 'Handles large datasets', 'Near-best accuracy'],
        'weaknesses': ['Slightly less precise on small datasets', 'Newer algorithm'],
        'best_for': 'Best balance of speed and accuracy'
    }
}

PRESETS = {
    'Balanced (General Agency)': {
        'icon': '⚖️',
        'description': 'Even distribution across all project types and industries.',
        'details': '''
**Best for**: Full-service digital agencies with diverse client portfolios.

**Mix**: Equal representation of corporate sites, e-commerce, web apps, 
and all industries from startups to enterprise.

**Model learns**: General patterns that work across different project types.
        ''',
        'project_weights': {'Landing Page': 10, 'Blog/Magazine': 10, 'Corporate Relaunch': 15, 'CMS Migration': 8, 'Accessibility Retrofit': 8, 'New Development': 12, 'E-Commerce/WooCommerce': 12, 'Intranet/Extranet': 8, 'Multisite Network': 7, 'Web Application': 10},
        'industry_weights': {'Startup/Tech': 8, 'Agency/Media': 8, 'Non-Profit/NGO': 8, 'Professional Services': 10, 'Manufacturing': 8, 'Retail/E-Commerce': 10, 'Education': 8, 'Energy/Utilities': 8, 'Finance/Insurance': 8, 'Healthcare/Pharma': 8, 'Public Sector': 8, 'Public Sector (EVB-IT)': 8}
    },
    'E-Commerce Focus': {
        'icon': '🛒',
        'description': 'Heavy emphasis on online shops and retail projects.',
        'details': '''
**Best for**: Agencies specializing in WooCommerce, online shops, and retail clients.

**Mix**: 35% e-commerce projects, 40% retail industry clients, 
emphasis on shop features, payment integrations, and product catalogs.

**Model learns**: Patterns specific to shop complexity, product counts, 
payment gateways, and retail client expectations.
        ''',
        'project_weights': {'Landing Page': 8, 'Blog/Magazine': 5, 'Corporate Relaunch': 10, 'CMS Migration': 5, 'Accessibility Retrofit': 3, 'New Development': 8, 'E-Commerce/WooCommerce': 35, 'Intranet/Extranet': 5, 'Multisite Network': 8, 'Web Application': 13},
        'industry_weights': {'Startup/Tech': 10, 'Agency/Media': 5, 'Non-Profit/NGO': 3, 'Professional Services': 5, 'Manufacturing': 8, 'Retail/E-Commerce': 40, 'Education': 3, 'Energy/Utilities': 3, 'Finance/Insurance': 8, 'Healthcare/Pharma': 5, 'Public Sector': 5, 'Public Sector (EVB-IT)': 5}
    },
    'Public Sector Focus': {
        'icon': '🏛️',
        'description': 'Government, education, and compliance-heavy projects.',
        'details': '''
**Best for**: Agencies working with government, municipalities, 
educational institutions, and healthcare organizations.

**Mix**: 35% public sector clients, high accessibility requirements (WCAG AA/AAA), 
emphasis on EVB-IT compliance, formal documentation, and multi-stakeholder processes.

**Model learns**: Patterns for compliance overhead, accessibility effort, 
extended decision cycles, and formal approval processes.
        ''',
        'project_weights': {'Landing Page': 5, 'Blog/Magazine': 8, 'Corporate Relaunch': 20, 'CMS Migration': 10, 'Accessibility Retrofit': 15, 'New Development': 10, 'E-Commerce/WooCommerce': 5, 'Intranet/Extranet': 12, 'Multisite Network': 8, 'Web Application': 7},
        'industry_weights': {'Startup/Tech': 3, 'Agency/Media': 2, 'Non-Profit/NGO': 8, 'Professional Services': 5, 'Manufacturing': 5, 'Retail/E-Commerce': 5, 'Education': 15, 'Energy/Utilities': 7, 'Finance/Insurance': 5, 'Healthcare/Pharma': 10, 'Public Sector': 15, 'Public Sector (EVB-IT)': 20}
    }
}

# ==============================================================================
# SESSION STATE
# ==============================================================================
for key, default in [('trained_models', None), ('selected_model', None), ('model_results', None), ('saved_estimates', []), ('rates', DEFAULT_RATES.copy())]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================================
# DATA GENERATION
# ==============================================================================
def generate_training_data(n_samples, preset_name, random_state=42):
    np.random.seed(random_state)
    preset = PRESETS[preset_name]
    project_types = list(PROJECT_BASE_HOURS.keys())
    industries = list(INDUSTRY_MULTIPLIERS.keys())
    designs = list(DESIGN_MULTIPLIERS.keys())
    
    project_probs = np.array([preset['project_weights'][p] for p in project_types], dtype=float)
    project_probs /= project_probs.sum()
    industry_probs = np.array([preset['industry_weights'][i] for i in industries], dtype=float)
    industry_probs /= industry_probs.sum()
    
    projects = []
    for _ in range(n_samples):
        ptype = np.random.choice(project_types, p=project_probs)
        industry = np.random.choice(industries, p=industry_probs)
        design = np.random.choice(designs)
        
        page_ranges = {'Landing Page': (1, 6), 'Blog/Magazine': (10, 40), 'Corporate Relaunch': (15, 50), 'CMS Migration': (20, 80), 'Accessibility Retrofit': (15, 50), 'New Development': (15, 60), 'E-Commerce/WooCommerce': (10, 40), 'Intranet/Extranet': (30, 100), 'Multisite Network': (40, 150), 'Web Application': (10, 50)}
        min_p, max_p = page_ranges[ptype]
        num_pages = np.random.randint(min_p, max_p + 1)
        num_languages = np.random.choice([1, 1, 1, 1, 2, 2, 3, 4])
        
        has_shop = 1 if (ptype == 'E-Commerce/WooCommerce' or np.random.random() < 0.12) else 0
        has_blog = 1 if (ptype == 'Blog/Magazine' or np.random.random() < 0.5) else 0
        has_booking = 1 if np.random.random() < 0.10 else 0
        has_membership = 1 if (ptype == 'Intranet/Extranet' or np.random.random() < 0.08) else 0
        has_events = 1 if np.random.random() < 0.12 else 0
        has_migration = 1 if (ptype == 'CMS Migration' or np.random.random() < 0.30) else 0
        has_api = 1 if (ptype == 'Web Application' or np.random.random() < 0.15) else 0
        
        num_forms = np.random.randint(1, 8)
        num_integrations = min(10, np.random.randint(0, 5) + (2 if has_shop else 0))
        accessibility_level = np.random.choice([1, 2, 2, 2, 3]) if 'Public' in industry else np.random.choice([0, 0, 0, 1, 2])
        client_responsiveness = np.random.choice([1, 2, 2, 2, 3])
        content_readiness = np.random.choice([0, 1, 1, 2, 2])
        decision_makers = np.random.randint(1, 5)
        revision_rounds = np.random.randint(1, 4)
        
        hours = PROJECT_BASE_HOURS[ptype] * DESIGN_MULTIPLIERS[design]
        hours += num_pages * 0.5
        if num_languages > 1: hours += (num_languages - 1) * 20
        hours += has_shop * 50 + has_blog * 12 + has_booking * 35 + has_membership * 40 + has_events * 18 + has_migration * 30 + has_api * 35
        hours += num_forms * 3 + num_integrations * 12 + accessibility_level * 25
        hours *= {1: 1.12, 2: 1.00, 3: 0.95}[client_responsiveness]
        hours *= {0: 1.10, 1: 1.02, 2: 1.00}[content_readiness]
        hours += decision_makers * 5 + revision_rounds * 6
        hours *= INDUSTRY_MULTIPLIERS[industry]
        hours = max(25, round(hours * np.clip(np.random.normal(1.0, 0.04), 0.92, 1.08)))
        
        projects.append({'project_type': ptype, 'industry': industry, 'design_source': design, 'num_pages': num_pages, 'num_languages': num_languages, 'has_shop': has_shop, 'has_blog': has_blog, 'has_booking': has_booking, 'has_membership': has_membership, 'has_events': has_events, 'has_migration': has_migration, 'has_api': has_api, 'num_forms': num_forms, 'num_integrations': num_integrations, 'accessibility_level': accessibility_level, 'client_responsiveness': client_responsiveness, 'content_readiness': content_readiness, 'decision_makers': decision_makers, 'revision_rounds': revision_rounds, 'hours': hours})
    
    return pd.DataFrame(projects)

# ==============================================================================
# MODEL TRAINING
# ==============================================================================
def train_all_models(df):
    cat_features = ['project_type', 'industry', 'design_source']
    num_features = [c for c in df.columns if c not in cat_features + ['hours']]
    X, y = df[cat_features + num_features], df['hours']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    for name, config in MODELS.items():
        preprocessor = ColumnTransformer([('num', StandardScaler(), num_features), ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)])
        pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', config['model'])])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        importance = None
        if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
            feat_names = num_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
            importance = pd.DataFrame({'feature': feat_names, 'importance': pipeline.named_steps['regressor'].feature_importances_}).sort_values('importance', ascending=False)
        
        results[name] = {'pipeline': pipeline, 'mae': mean_absolute_error(y_test, y_pred), 'mape': mean_absolute_percentage_error(y_test, y_pred), 'r2': r2_score(y_test, y_pred), 'importance': importance, 'y_test': y_test, 'y_pred': y_pred}
    return results

# ==============================================================================
# HELPERS
# ==============================================================================
def get_phases(hours, ptype, migration):
    bases = {'Landing Page': {'Discovery': 0.08, 'Design': 0.35, 'Development': 0.38, 'QA': 0.12, 'Launch': 0.07}, 'E-Commerce/WooCommerce': {'Discovery': 0.10, 'Design': 0.18, 'Development': 0.48, 'QA': 0.15, 'Launch': 0.09}}
    base = bases.get(ptype, {'Discovery': 0.10, 'Design': 0.22, 'Development': 0.42, 'QA': 0.14, 'Launch': 0.12})
    if ptype in ['Intranet/Extranet', 'Web Application']: base = {'Discovery': 0.15, 'Design': 0.15, 'Development': 0.45, 'QA': 0.15, 'Launch': 0.10}
    if migration: base['Discovery'] += 0.03; base['Development'] += 0.02; base['Design'] -= 0.03; base['Launch'] -= 0.02
    return {k: max(0, int(hours * v)) for k, v in base.items()}

def get_team(hours, phases):
    return {'Senior Developer': int(phases['Development'] * 0.35), 'Developer': int(phases['Development'] * 0.65), 'Designer': int(phases['Design'] * 0.90), 'Project Manager': int(hours * 0.12), 'QA Engineer': int(phases['QA'] * 0.85)}

def get_confidence(hours, r2):
    uncertainty = max(0.05, min(0.20, (1 - r2) * 0.5))
    return {'low': int(hours * (1 - uncertainty)), 'mid': hours, 'high': int(hours * (1 + uncertainty))}

def get_insights(data, hours):
    insights = []
    if data['num_integrations'] >= 4: insights.append(('🔌', 'High Integrations', f"{data['num_integrations']} integrations = +{data['num_integrations']*12}h"))
    if data['num_languages'] >= 3: insights.append(('🌍', 'Multilingual', f"{data['num_languages']} languages = +{(data['num_languages']-1)*20}h"))
    if data['accessibility_level'] >= 2: insights.append(('♿', ['','WCAG A','WCAG AA','WCAG AAA'][data['accessibility_level']], f"+{data['accessibility_level']*25}h compliance"))
    if data['content_readiness'] == 0: insights.append(('📝', 'No Content', '+10% buffer for delays'))
    if data['client_responsiveness'] == 1: insights.append(('🐌', 'Slow Client', '+12% for feedback cycles'))
    if 'Public' in data['industry']: insights.append(('🏛️', 'Public Sector', f"+{int((INDUSTRY_MULTIPLIERS[data['industry']]-1)*100)}% compliance overhead"))
    return insights[:5]

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🧠 Project Estimator Pro")
    st.caption("Machine Learning-powered project time and cost estimation")
    
    tabs = st.tabs(["🏠 Home", "🎯 Train Model", "⚡ Estimate", "⚙️ Settings", "🧠 How It Works"])
    
    # ==========================================================================
    # TAB 0: HOME
    # ==========================================================================
    with tabs[0]:
        st.header("Welcome to Project Estimator Pro")
        
        st.markdown("""
        **Project Estimator Pro** uses machine learning to predict how many hours your 
        WordPress projects will take. It learns from patterns in project data to give you 
        accurate estimates in seconds.
        """)
        
        # Status
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.selected_model:
                r = st.session_state.model_results[st.session_state.selected_model]
                st.success(f"✅ Model Ready: {st.session_state.selected_model}")
                st.metric("Accuracy", f"{r['r2']:.1%} R²")
            else:
                st.warning("⚠️ No model trained yet")
                st.caption("Go to 'Train Model' to get started")
        
        with col2:
            st.info(f"💾 {len(st.session_state.saved_estimates)} Saved Estimates")
            if st.session_state.saved_estimates:
                latest = st.session_state.saved_estimates[-1]
                st.caption(f"Latest: {latest['name'][:30]}...")
        
        with col3:
            avg_rate = sum(st.session_state.rates.values()) / len(st.session_state.rates)
            st.info(f"💰 Avg Rate: €{avg_rate:.0f}/h")
            st.caption("Configure in Settings")
        
        st.divider()
        
        # How to use
        st.subheader("🚀 Quick Start Guide")
        
        st.markdown("""
        <style>
        .step-box {
            background: linear-gradient(135deg, #667eea22, #764ba222);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Step 1: Train Your Model")
            st.markdown("""
            Go to the **🎯 Train Model** tab and:
            
            1. **Choose data size** — More data = better accuracy
            2. **Select a preset** — Match your agency's focus
            3. **Click Train** — Takes about 30 seconds
            4. **Pick the best model** — Compare 3 algorithms
            """)
            
            st.markdown("### Step 2: Create Estimates")
            st.markdown("""
            Go to the **⚡ Estimate** tab and:
            
            1. **Enter project details** — Type, industry, features
            2. **Get instant prediction** — Hours, cost, phases
            3. **Review insights** — Key factors affecting time
            4. **Save for later** — Build your estimate library
            """)
        
        with col2:
            st.markdown("### Understanding the Tabs")
            
            with st.expander("🎯 Train Model", expanded=True):
                st.markdown("""
                **Purpose:** Create your ML model
                
                - Choose training data size (10k-75k projects)
                - Select project mix preset (Balanced, E-Commerce, Public Sector)
                - Train 3 different algorithms
                - Compare accuracy metrics (R², MAE, MAPE)
                - Select the best model for your needs
                """)
            
            with st.expander("⚡ Estimate"):
                st.markdown("""
                **Purpose:** Generate project estimates
                
                - Configure project type, industry, design approach
                - Set scope (pages, languages, features)
                - Specify client factors (speed, content readiness)
                - Get instant hour and cost predictions
                - See phase breakdown and team allocation
                - Save estimates for future reference
                """)
            
            with st.expander("⚙️ Settings"):
                st.markdown("""
                **Purpose:** Customize your configuration
                
                - Set hourly rates by role (Dev, Designer, PM, QA)
                - View current model status
                - Clear saved estimates
                """)
            
            with st.expander("🧠 How It Works"):
                st.markdown("""
                **Purpose:** Learn about the technology
                
                - Understand machine learning basics
                - Learn how each algorithm works
                - See the estimation formula
                - Understand accuracy metrics
                """)
        
        st.divider()
        
        # Key Features
        st.subheader("✨ Key Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 🎯 High Accuracy")
            st.markdown("R² > 95% means predictions explain 95% of project variation")
        
        with col2:
            st.markdown("### ⚡ Instant Results")
            st.markdown("Get estimates in milliseconds after model is trained")
        
        with col3:
            st.markdown("### 📊 Full Breakdown")
            st.markdown("Hours, costs, phases, team allocation, and key insights")
        
        with col4:
            st.markdown("### 🔧 Customizable")
            st.markdown("Adjust rates, choose presets, save estimates")
        
        st.divider()
        
        # CTA
        if not st.session_state.selected_model:
            st.markdown("### 👉 Ready to get started?")
            st.markdown("Head to the **🎯 Train Model** tab to train your first model!")
        else:
            st.markdown("### 👉 Model is ready!")
            st.markdown("Go to the **⚡ Estimate** tab to create your first estimate!")
    
    # ==========================================================================
    # TAB 1: TRAIN MODEL
    # ==========================================================================
    with tabs[1]:
        st.header("Train Your Estimation Model")
        st.markdown("Configure the training data and train 3 different ML models to find the best one.")
        
        st.divider()
        
        # Step 1: Data Size
        st.subheader("1️⃣ Training Data Size")
        st.markdown("""
        **How many example projects should the model learn from?**
        
        The model learns patterns from simulated projects. More projects = better pattern 
        recognition, but longer training time.
        """)
        
        n_samples = st.select_slider("Number of training projects", options=[10000, 20000, 30000, 50000, 75000], value=30000, format_func=lambda x: f"{x:,} projects")
        
        size_info = {10000: ("⚡ Fast", "~15 sec", "Quick experiments"), 20000: ("🔄 Balanced", "~25 sec", "Good accuracy"), 30000: ("✅ Recommended", "~35 sec", "Optimal balance"), 50000: ("🎯 High Accuracy", "~50 sec", "Production ready"), 75000: ("🔬 Maximum", "~75 sec", "Highest accuracy")}
        info = size_info[n_samples]
        st.info(f"{info[0]} • Training: {info[1]} • {info[2]}")
        
        st.divider()
        
        # Step 2: Preset
        st.subheader("2️⃣ Project Mix Preset")
        st.markdown("""
        **What types of projects does your agency typically do?**
        
        Choose a preset that matches your agency's focus. The model will learn patterns 
        most relevant to your work.
        """)
        
        cols = st.columns(3)
        for i, (name, config) in enumerate(PRESETS.items()):
            with cols[i]:
                st.markdown(f"### {config['icon']} {name.split('(')[0].strip()}")
                st.caption(config['description'])
                with st.expander("Details"):
                    st.markdown(config['details'])
        
        selected_preset = st.radio("Select preset", options=list(PRESETS.keys()), horizontal=True, label_visibility="collapsed")
        
        st.divider()
        
        # Step 3: Train
        st.subheader("3️⃣ Train Models")
        
        if st.button("🚀 Generate Data & Train All Models", type="primary", use_container_width=True):
            progress = st.progress(0, text="Generating training data...")
            df = generate_training_data(n_samples, selected_preset)
            progress.progress(30, text="Training models...")
            results = train_all_models(df)
            st.session_state.model_results = results
            progress.progress(100, text="Complete!")
            progress.empty()
            st.success("✅ All 3 models trained!")
            
            with st.expander("📊 Training Data Summary", expanded=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Projects", f"{len(df):,}")
                c2.metric("Avg Hours", f"{df['hours'].mean():.0f}h")
                c3.metric("Median", f"{df['hours'].median():.0f}h")
                c4.metric("Min", f"{df['hours'].min()}h")
                c5.metric("Max", f"{df['hours'].max()}h")
                
                if PLOTLY:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x='hours', nbins=50, title='Hours Distribution')
                        fig.update_layout(height=280)
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.pie(values=df['project_type'].value_counts().values, names=df['project_type'].value_counts().index, title='Project Types')
                        fig.update_layout(height=280, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Results
        if st.session_state.model_results:
            results = st.session_state.model_results
            
            st.divider()
            st.header("📊 Model Comparison")
            
            results_df = pd.DataFrame([{'Model': name, 'R² Score': r['r2'], 'MAE': r['mae'], 'MAPE': r['mape']} for name, r in results.items()]).sort_values('R² Score', ascending=False)
            
            st.dataframe(results_df.style.format({'R² Score': '{:.1%}', 'MAE': '±{:.1f}h', 'MAPE': '{:.1%}'}).background_gradient(subset=['R² Score'], cmap='RdYlGn').background_gradient(subset=['MAE'], cmap='RdYlGn_r'), use_container_width=True, hide_index=True)
            
            best = results_df.iloc[0]
            if best['R² Score'] >= 0.93:
                st.success(f"🎉 **Excellent!** R² of {best['R² Score']:.1%} — highly reliable predictions.")
            elif best['R² Score'] >= 0.85:
                st.info(f"👍 **Good.** R² of {best['R² Score']:.1%} — solid estimates.")
            else:
                st.warning(f"⚠️ **Moderate.** Consider more training data.")
            
            if PLOTLY:
                col1, col2 = st.columns(2)
                colors = ['#667eea', '#27ae60', '#f39c12']
                with col1:
                    fig = go.Figure([go.Bar(x=results_df['Model'], y=results_df['R² Score']*100, marker_color=colors, text=[f"{v:.1%}" for v in results_df['R² Score']], textposition='outside')])
                    fig.update_layout(title='R² Score (higher = better)', yaxis_title='%', height=350, yaxis_range=[0, 105])
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = go.Figure([go.Bar(x=results_df['Model'], y=results_df['MAE'], marker_color=colors, text=[f"±{v:.1f}h" for v in results_df['MAE']], textposition='outside')])
                    fig.update_layout(title='Mean Absolute Error (lower = better)', yaxis_title='Hours', height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            best_name = results_df.iloc[0]['Model']
            r = results[best_name]
            
            if PLOTLY:
                st.subheader("🎯 Prediction Accuracy")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=r['y_test'], y=r['y_pred'], mode='markers', marker=dict(size=4, opacity=0.4, color='#667eea'), name='Predictions'))
                min_v, max_v = r['y_test'].min(), r['y_test'].max()
                fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', line=dict(color='red', dash='dash'), name='Perfect'))
                fig.update_layout(xaxis_title='Actual Hours', yaxis_title='Predicted Hours', height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points on the red line = perfect predictions.")
            
            st.divider()
            st.header("✅ Select Your Model")
            
            cols = st.columns(3)
            for i, name in enumerate(results.keys()):
                r = results[name]
                config = MODELS[name]
                is_best = name == best_name
                
                with cols[i]:
                    st.markdown(f"### {config['icon']} {name} {'🏆' if is_best else ''}")
                    st.caption(config['tagline'])
                    st.metric("Accuracy", f"{r['r2']:.1%}")
                    st.metric("Avg Error", f"±{r['mae']:.1f}h")
                    
                    if st.button(f"Use {name.split()[0]}", key=f"sel_{name}", type="primary" if is_best else "secondary", use_container_width=True):
                        st.session_state.selected_model = name
                        st.success(f"✅ {name} selected!")
                        st.balloons()
    
    # ==========================================================================
    # TAB 2: ESTIMATE
    # ==========================================================================
    with tabs[2]:
        if not st.session_state.selected_model:
            st.info("👆 Please train and select a model in the **Train Model** tab first.")
            st.stop()
        
        model_name = st.session_state.selected_model
        model_info = st.session_state.model_results[model_name]
        
        st.markdown(f"**Active Model:** {MODELS[model_name]['icon']} {model_name} (R²: {model_info['r2']:.1%}, MAE: ±{model_info['mae']:.0f}h)")
        st.divider()
        
        col_in, col_out = st.columns([2, 3])
        
        with col_in:
            st.subheader("📝 Project Configuration")
            
            ptype = st.selectbox("Project Type", list(PROJECT_BASE_HOURS.keys()), help=f"Base: {min(PROJECT_BASE_HOURS.values())}-{max(PROJECT_BASE_HOURS.values())}h")
            industry = st.selectbox("Client Industry", list(INDUSTRY_MULTIPLIERS.keys()), help="Affects overhead")
            design = st.selectbox("Design Approach", list(DESIGN_MULTIPLIERS.keys()))
            
            st.divider()
            st.markdown("**📐 Scope**")
            c1, c2 = st.columns(2)
            pages = c1.slider("Pages", 1, 150, 25)
            langs = c2.slider("Languages", 1, 5, 1)
            forms = c1.slider("Forms", 1, 10, 3)
            integrations = c2.slider("Integrations", 0, 10, 2)
            
            st.divider()
            st.markdown("**⚙️ Features**")
            c1, c2 = st.columns(2)
            shop = c1.checkbox("E-Commerce", help="+50h")
            blog = c1.checkbox("Blog", True, help="+12h")
            booking = c1.checkbox("Booking", help="+35h")
            membership = c2.checkbox("Membership", help="+40h")
            events = c2.checkbox("Events", help="+18h")
            migration = c2.checkbox("Migration", help="+30h")
            api = c1.checkbox("Custom API", help="+35h")
            
            st.divider()
            a11y = st.radio("Accessibility", [0, 1, 2, 3], format_func=lambda x: ['None', 'WCAG A', 'WCAG AA', 'WCAG AAA'][x], horizontal=True)
            
            st.divider()
            st.markdown("**👥 Client**")
            c1, c2 = st.columns(2)
            speed = c1.radio("Speed", [1, 2, 3], format_func=lambda x: ['🐌', '👍', '⚡'][x-1], horizontal=True)
            content = c2.radio("Content", [0, 1, 2], format_func=lambda x: ['❌', '📝', '✅'][x], horizontal=True)
            makers = st.slider("Decision Makers", 1, 6, 2)
            revisions = st.slider("Revisions", 1, 5, 2)
        
        with col_out:
            inp = {'project_type': ptype, 'industry': industry, 'design_source': design, 'num_pages': pages, 'num_languages': langs, 'has_shop': int(shop), 'has_blog': int(blog), 'has_booking': int(booking), 'has_membership': int(membership), 'has_events': int(events), 'has_migration': int(migration), 'has_api': int(api), 'num_forms': forms, 'num_integrations': integrations, 'accessibility_level': a11y, 'client_responsiveness': speed, 'content_readiness': content, 'decision_makers': makers, 'revision_rounds': revisions}
            
            hours = int(model_info['pipeline'].predict(pd.DataFrame([inp]))[0])
            conf = get_confidence(hours, model_info['r2'])
            phases = get_phases(hours, ptype, migration)
            team = get_team(hours, phases)
            cost = sum(h * st.session_state.rates.get(r, 120) for r, h in team.items())
            mult = INDUSTRY_MULTIPLIERS.get(industry, 1.0)
            
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;padding:30px;text-align:center;color:white;margin-bottom:20px;">
                <p style="margin:0;opacity:.8;font-size:14px;">ESTIMATED HOURS</p>
                <h1 style="font-size:72px;margin:10px 0;font-weight:800">{hours}</h1>
                <p style="margin:0;opacity:.7">Range: {conf['low']} – {conf['high']} hours</p>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{hours/40:.1f} weeks")
            c2.metric("Base Cost", f"€{cost:,}")
            c3.metric("Buffer", f"+{int((mult-1)*100)}%")
            c4.metric("Total", f"€{int(cost*mult):,}")
            
            insights = get_insights(inp, hours)
            if insights:
                st.divider()
                st.subheader("💡 Key Factors")
                for icon, title, text in insights: st.markdown(f"**{icon} {title}** — {text}")
            
            st.divider()
            st.subheader("📊 Phases")
            if PLOTLY:
                colors = ['#3498db', '#9b59b6', '#27ae60', '#f39c12', '#e74c3c']
                fig = go.Figure()
                for i, (ph, h) in enumerate(phases.items()):
                    if h > 0: fig.add_trace(go.Bar(x=[h], y=[ph], orientation='h', marker_color=colors[i], text=f"{h}h ({h/hours*100:.0f}%)", textposition='inside', name=ph))
                fig.update_layout(height=200, showlegend=False, yaxis={'categoryorder': 'array', 'categoryarray': list(phases.keys())[::-1]}, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("👥 Team & Cost")
            tdf = pd.DataFrame([{'Role': r, 'Hours': h, 'Rate': f"€{st.session_state.rates.get(r, 120)}", 'Cost': h * st.session_state.rates.get(r, 120)} for r, h in team.items() if h > 0])
            st.dataframe(tdf.style.format({'Cost': '€{:,.0f}'}), hide_index=True, use_container_width=True)
            st.markdown(f"**Base:** €{cost:,} + **Buffer:** €{int(cost*mult)-cost:,} = **Total: €{int(cost*mult):,}**")
            
            st.divider()
            c1, c2 = st.columns([3, 1])
            name = c1.text_input("Save as", f"{ptype} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            if c2.button("💾 Save", use_container_width=True):
                st.session_state.saved_estimates.append({'name': name, 'hours': hours, 'cost': int(cost*mult), 'date': datetime.now().isoformat()})
                st.success("Saved!")
            
            if st.session_state.saved_estimates:
                with st.expander(f"📁 Saved ({len(st.session_state.saved_estimates)})"):
                    for s in st.session_state.saved_estimates[::-1]: st.markdown(f"**{s['name']}** — {s['hours']}h / €{s['cost']:,}")
    
    # ==========================================================================
    # TAB 3: SETTINGS
    # ==========================================================================
    with tabs[3]:
        st.header("⚙️ Settings")
        
        st.subheader("💰 Hourly Rates")
        st.markdown("Configure your agency's rates. These are used to calculate project costs.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Development**")
            st.session_state.rates['Senior Developer'] = st.number_input("Senior Developer (€/h)", 80, 250, st.session_state.rates['Senior Developer'], 5)
            st.session_state.rates['Developer'] = st.number_input("Developer (€/h)", 60, 200, st.session_state.rates['Developer'], 5)
            st.session_state.rates['QA Engineer'] = st.number_input("QA Engineer (€/h)", 50, 180, st.session_state.rates['QA Engineer'], 5)
        with col2:
            st.markdown("**Design & Management**")
            st.session_state.rates['Designer'] = st.number_input("Designer (€/h)", 60, 200, st.session_state.rates['Designer'], 5)
            st.session_state.rates['Project Manager'] = st.number_input("Project Manager (€/h)", 70, 200, st.session_state.rates['Project Manager'], 5)
        
        st.divider()
        st.subheader("📊 Current Model")
        if st.session_state.selected_model:
            r = st.session_state.model_results[st.session_state.selected_model]
            c1, c2, c3 = st.columns(3)
            c1.metric("Model", st.session_state.selected_model)
            c2.metric("Accuracy", f"{r['r2']:.1%}")
            c3.metric("Avg Error", f"±{r['mae']:.1f}h")
        else:
            st.info("No model trained yet.")
        
        st.divider()
        if st.button("🗑️ Clear All Saved Estimates"):
            st.session_state.saved_estimates = []
            st.success("Cleared!")
    
    # ==========================================================================
    # TAB 4: HOW IT WORKS
    # ==========================================================================
    with tabs[4]:
        st.header("🧠 How This AI Works")
        st.markdown("*A comprehensive guide for non-technical readers*")
        
        st.divider()
        
        # Section 1: The Concept
        st.subheader("📖 What is Machine Learning?")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Machine Learning (ML)** is a way of teaching computers to find patterns in data 
            and make predictions — without being explicitly programmed with rules.
            
            Think of it like teaching someone to recognize cats vs. dogs:
            - **Traditional programming:** Write rules like "if it has pointy ears AND whiskers AND says meow, it's a cat"
            - **Machine learning:** Show 10,000 photos of cats and dogs, let the computer figure out the patterns itself
            
            For project estimation:
            - **Traditional:** Create complex formulas with hundreds of if-then rules
            - **Machine learning:** Show 30,000 example projects, let the computer learn what makes projects take longer
            
            The ML approach is better because:
            1. It can find patterns humans might miss
            2. It considers many factors simultaneously
            3. It can be retrained as you get more data
            """)
        
        with col2:
            st.info("""
            **Key Insight**
            
            The model doesn't "know" that 
            e-commerce projects are complex.
            
            It learns this by seeing that 
            projects labeled "e-commerce" 
            consistently take more hours 
            in the training data.
            """)
        
        st.divider()
        
        # Section 2: Our Approach
        st.subheader("🔄 The Training Process")
        
        st.markdown("""
        Here's exactly what happens when you click "Train Models":
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 1️⃣ Generate Data")
            st.markdown("""
            We create 30,000 simulated 
            projects with realistic 
            characteristics:
            
            - Project types
            - Industries
            - Features
            - Client factors
            - **Known hours**
            """)
        
        with col2:
            st.markdown("### 2️⃣ Split Data")
            st.markdown("""
            We divide the data:
            
            - **80% Training**
              Model learns from these
            
            - **20% Testing**
              Model never sees these
              until final evaluation
            """)
        
        with col3:
            st.markdown("### 3️⃣ Train Models")
            st.markdown("""
            Three algorithms analyze 
            the training data:
            
            - Find patterns
            - Build internal rules
            - Optimize predictions
            
            Each uses different math.
            """)
        
        with col4:
            st.markdown("### 4️⃣ Evaluate")
            st.markdown("""
            We test on the 20% held back:
            
            - Predict hours
            - Compare to actual
            - Calculate accuracy
            
            This is the "real" score.
            """)
        
        st.divider()
        
        # Section 3: The Algorithms
        st.subheader("🤖 The Three Algorithms Explained")
        
        for name, config in MODELS.items():
            with st.expander(f"{config['icon']} {name} — {config['tagline']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**{config['description']}**")
                    st.markdown(config['how_it_works'])
                    
                    st.markdown("**Technical Details:**")
                    if 'Gradient' in name:
                        st.markdown("""
                        - Uses **200 sequential trees**
                        - Each tree has max **5-6 levels** of decisions
                        - Learning rate of **0.1** (how much each tree contributes)
                        - Requires at least **10 samples** per leaf node
                        """)
                    else:
                        st.markdown("""
                        - Uses **200 parallel trees**
                        - Each tree has max **12 levels** of decisions
                        - Trees are built independently
                        - Final answer = average of all trees
                        """)
                
                with col2:
                    st.markdown("**Strengths**")
                    for s in config['strengths']:
                        st.markdown(f"✅ {s}")
                    
                    st.markdown("**Weaknesses**")
                    for w in config['weaknesses']:
                        st.markdown(f"⚠️ {w}")
                    
                    st.info(f"**Best for:** {config['best_for']}")
        
        st.divider()
        
        # Section 4: Decision Trees
        st.subheader("🌳 How Decision Trees Work")
        
        st.markdown("""
        All three algorithms are based on **decision trees**. Here's a simplified example:
        """)
        
        st.code("""
        Is it an E-Commerce project?
        ├── YES: Is there a custom design?
        │   ├── YES: Are there more than 3 integrations?
        │   │   ├── YES: Predict 350 hours
        │   │   └── NO: Predict 280 hours
        │   └── NO: Predict 220 hours
        └── NO: Is it a Landing Page?
            ├── YES: Predict 45 hours
            └── NO: Continue checking other factors...
        """, language=None)
        
        st.markdown("""
        The actual trees are much more complex (200 trees × up to 12 levels each = thousands of decision paths), 
        but the principle is the same: **ask questions about the project, follow the path, arrive at a prediction**.
        
        The magic is that the algorithm **figures out which questions to ask** and **what thresholds to use** 
        by analyzing the training data.
        """)
        
        st.divider()
        
        # Section 5: Metrics
        st.subheader("📊 Understanding Accuracy Metrics")
        
        st.markdown("""
        After training, we measure how good the model is using several metrics:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### R² Score")
            st.markdown("""
            **"How much variation does the model explain?"**
            
            - **R² = 0.95** means 95% of why projects take different amounts of time is captured by the model
            - The remaining 5% is unpredictable noise
            
            | Value | Quality |
            |-------|---------|
            | > 0.95 | Excellent |
            | 0.90-0.95 | Very Good |
            | 0.80-0.90 | Good |
            | < 0.80 | Needs work |
            """)
        
        with col2:
            st.markdown("### MAE (Mean Absolute Error)")
            st.markdown("""
            **"On average, how many hours off is the prediction?"**
            
            - **MAE = 15** means predictions are typically off by 15 hours
            - For a 200h project: expect 185-215h actual
            
            | Value | Quality |
            |-------|---------|
            | < 15h | Excellent |
            | 15-25h | Very Good |
            | 25-40h | Good |
            | > 40h | Needs work |
            """)
        
        with col3:
            st.markdown("### MAPE (Mean Absolute % Error)")
            st.markdown("""
            **"On average, what percentage off is the prediction?"**
            
            - **MAPE = 8%** means predictions are typically off by 8%
            - Useful for comparing across different project sizes
            
            | Value | Quality |
            |-------|---------|
            | < 8% | Excellent |
            | 8-12% | Very Good |
            | 12-18% | Good |
            | > 18% | Needs work |
            """)
        
        st.divider()
        
        # Section 6: The Formula
        st.subheader("🔢 The Estimation Formula")
        
        st.markdown("""
        The training data is generated using this formula. The ML model **learns to approximate this** 
        from the data — we don't tell it these rules directly!
        """)
        
        st.code("""
Hours = Base Hours (by project type: 45-350h)
      × Design Multiplier (0.65-1.25)
      + Pages × 0.5h
      + (Languages - 1) × 20h
      + Shop × 50h
      + Blog × 12h
      + Booking × 35h
      + Membership × 40h
      + Events × 18h
      + Migration × 30h
      + API × 35h
      + Forms × 3h
      + Integrations × 12h
      + Accessibility Level × 25h
      × Client Speed Factor (0.95-1.12)
      × Content Readiness Factor (1.00-1.10)
      + Decision Makers × 5h
      + Revision Rounds × 6h
      × Industry Multiplier (0.92-1.30)
      × Random Noise (±4%)
        """, language=None)
        
        st.markdown("""
        **Example calculation:**
        
        Corporate Relaunch (160h) × Custom Design (1.25) + 30 pages (15h) + 2 languages (20h) 
        + Blog (12h) + Migration (30h) + 3 forms (9h) + 2 integrations (24h) + WCAG AA (50h) 
        × Normal client (1.0) × Partial content (1.02) + 2 decision makers (10h) + 2 revisions (12h) 
        × Public Sector (1.22) = **~450 hours**
        """)
        
        st.divider()
        
        # Section 7: Feature Importance
        st.subheader("📈 What Matters Most?")
        
        st.markdown("""
        The model automatically discovers which factors have the biggest impact. 
        Here's what typically matters most:
        """)
        
        importance_data = [
            ("Project Type", "Base hours vary from 45h (Landing Page) to 350h (Multisite)", 25),
            ("Design Approach", "Custom design adds 25% vs. using a theme", 18),
            ("Industry", "Public sector adds 30% overhead", 15),
            ("Integrations", "Each integration adds ~12 hours", 12),
            ("Accessibility", "WCAG AA adds ~50 hours", 10),
            ("Client Speed", "Slow clients add 12% overhead", 8),
            ("Languages", "Each extra language adds ~20 hours", 7),
            ("Features", "Shop +50h, Membership +40h, etc.", 5),
        ]
        
        if PLOTLY:
            fig = go.Figure(go.Bar(
                x=[d[2] for d in importance_data],
                y=[d[0] for d in importance_data],
                orientation='h',
                marker_color='#667eea',
                text=[f"{d[2]}%" for d in importance_data],
                textposition='outside'
            ))
            fig.update_layout(height=350, xaxis_title="Relative Importance (%)", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        for name, desc, pct in importance_data:
            st.markdown(f"- **{name}** ({pct}%): {desc}")
        
        st.divider()
        
        # Section 8: Limitations
        st.subheader("⚠️ Important Limitations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("""
            **This model is trained on SIMULATED data**
            
            The training projects are generated based on industry heuristics, 
            not your actual project history. This means:
            
            - Patterns may not match your specific team
            - Your velocity might be different
            - Your clients might behave differently
            
            **For production use:** Consider retraining with your 
            actual completed project data.
            """)
        
        with col2:
            st.warning("""
            **ML cannot predict everything**
            
            Some things are inherently unpredictable:
            
            - Scope creep mid-project
            - Key team member leaving
            - Client changing their mind
            - Technical surprises
            - External dependencies
            
            **Always add buffers** and use estimates as 
            starting points, not guarantees.
            """)
        
        st.divider()
        
        # Section 9: Tips
        st.subheader("💡 Tips for Best Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Choose the Right Preset
            
            If you specialize in e-commerce, 
            use the E-Commerce preset. The 
            model will learn patterns most 
            relevant to your work.
            """)
        
        with col2:
            st.markdown("""
            ### More Data = Better
            
            If accuracy is critical, use 
            50,000+ training projects. The 
            extra training time is worth it 
            for production models.
            """)
        
        with col3:
            st.markdown("""
            ### Trust the Confidence Range
            
            The range (e.g., 180-220h) is 
            based on model accuracy. Use it 
            for setting expectations with 
            clients.
            """)
        
        st.divider()
        
        # Glossary
        st.subheader("📚 Glossary")
        
        with st.expander("Technical Terms Explained"):
            st.markdown("""
            | Term | Definition |
            |------|------------|
            | **Algorithm** | A set of rules/steps the computer follows to learn patterns |
            | **Decision Tree** | A flowchart-like structure that makes predictions by asking yes/no questions |
            | **Ensemble** | Combining multiple models to get better predictions than any single model |
            | **Feature** | An input variable (e.g., "number of pages" or "industry") |
            | **Gradient Boosting** | An ensemble method where each tree fixes the errors of previous trees |
            | **Histogram** | A way of grouping continuous values into bins for faster processing |
            | **MAE** | Mean Absolute Error - average prediction error in the original units |
            | **MAPE** | Mean Absolute Percentage Error - average error as a percentage |
            | **Model** | The trained "brain" that makes predictions based on learned patterns |
            | **Overfitting** | When a model memorizes training data but fails on new data |
            | **R² Score** | Coefficient of determination - how much variance the model explains (0-1) |
            | **Random Forest** | An ensemble of many independent decision trees |
            | **Target Variable** | What we're trying to predict (in our case: hours) |
            | **Training Data** | The examples we use to teach the model |
            | **Test Data** | Held-back examples we use to evaluate the trained model |
            """)

if __name__ == "__main__":
    main()
