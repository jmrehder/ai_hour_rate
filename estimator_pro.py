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
    'Landing Page': 45,
    'Blog/Magazine': 95,
    'Corporate Relaunch': 160,
    'CMS Migration': 130,
    'Accessibility Retrofit': 90,
    'New Development': 200,
    'E-Commerce/WooCommerce': 240,
    'Intranet/Extranet': 280,
    'Multisite Network': 350,
    'Web Application': 320,
}

INDUSTRY_MULTIPLIERS = {
    'Startup/Tech': 0.92,
    'Agency/Media': 0.95,
    'Non-Profit/NGO': 1.00,
    'Professional Services': 1.00,
    'Manufacturing': 1.05,
    'Retail/E-Commerce': 1.08,
    'Education': 1.10,
    'Energy/Utilities': 1.12,
    'Finance/Insurance': 1.15,
    'Healthcare/Pharma': 1.18,
    'Public Sector': 1.22,
    'Public Sector (EVB-IT)': 1.30,
}

DESIGN_MULTIPLIERS = {
    'No Design (Dev only)': 0.65,
    'Premium Theme': 0.80,
    'Existing Design System': 0.90,
    'Theme Customization': 1.00,
    'Custom Design (Figma)': 1.25,
}

MODELS = {
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=10, subsample=0.9, random_state=42),
        'icon': '🎯',
        'tagline': 'Highest Accuracy',
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
        'icon': '🌳',
        'tagline': 'Robust & Reliable',
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
        'icon': '⚡',
        'tagline': 'Speed + Accuracy',
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
        'project_weights': {
            'Landing Page': 10, 'Blog/Magazine': 10, 'Corporate Relaunch': 15,
            'CMS Migration': 8, 'Accessibility Retrofit': 8, 'New Development': 12,
            'E-Commerce/WooCommerce': 12, 'Intranet/Extranet': 8,
            'Multisite Network': 7, 'Web Application': 10
        },
        'industry_weights': {
            'Startup/Tech': 8, 'Agency/Media': 8, 'Non-Profit/NGO': 8,
            'Professional Services': 10, 'Manufacturing': 8, 'Retail/E-Commerce': 10,
            'Education': 8, 'Energy/Utilities': 8, 'Finance/Insurance': 8,
            'Healthcare/Pharma': 8, 'Public Sector': 8, 'Public Sector (EVB-IT)': 8
        }
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
        'project_weights': {
            'Landing Page': 8, 'Blog/Magazine': 5, 'Corporate Relaunch': 10,
            'CMS Migration': 5, 'Accessibility Retrofit': 3, 'New Development': 8,
            'E-Commerce/WooCommerce': 35, 'Intranet/Extranet': 5,
            'Multisite Network': 8, 'Web Application': 13
        },
        'industry_weights': {
            'Startup/Tech': 10, 'Agency/Media': 5, 'Non-Profit/NGO': 3,
            'Professional Services': 5, 'Manufacturing': 8, 'Retail/E-Commerce': 40,
            'Education': 3, 'Energy/Utilities': 3, 'Finance/Insurance': 8,
            'Healthcare/Pharma': 5, 'Public Sector': 5, 'Public Sector (EVB-IT)': 5
        }
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
        'project_weights': {
            'Landing Page': 5, 'Blog/Magazine': 8, 'Corporate Relaunch': 20,
            'CMS Migration': 10, 'Accessibility Retrofit': 15, 'New Development': 10,
            'E-Commerce/WooCommerce': 5, 'Intranet/Extranet': 12,
            'Multisite Network': 8, 'Web Application': 7
        },
        'industry_weights': {
            'Startup/Tech': 3, 'Agency/Media': 2, 'Non-Profit/NGO': 8,
            'Professional Services': 5, 'Manufacturing': 5, 'Retail/E-Commerce': 5,
            'Education': 15, 'Energy/Utilities': 7, 'Finance/Insurance': 5,
            'Healthcare/Pharma': 10, 'Public Sector': 15, 'Public Sector (EVB-IT)': 20
        }
    }
}

# ==============================================================================
# SESSION STATE
# ==============================================================================
for key, default in [
    ('trained_models', None), ('selected_model', None), ('model_results', None),
    ('saved_estimates', []), ('rates', DEFAULT_RATES.copy())
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================================
# DATA GENERATION
# ==============================================================================
def generate_training_data(n_samples, preset_name, random_state=42):
    """Generate training data with clear, learnable patterns."""
    np.random.seed(random_state)
    
    preset = PRESETS[preset_name]
    project_types = list(PROJECT_BASE_HOURS.keys())
    industries = list(INDUSTRY_MULTIPLIERS.keys())
    designs = list(DESIGN_MULTIPLIERS.keys())
    
    # Normalize weights
    project_probs = np.array([preset['project_weights'][p] for p in project_types], dtype=float)
    project_probs /= project_probs.sum()
    industry_probs = np.array([preset['industry_weights'][i] for i in industries], dtype=float)
    industry_probs /= industry_probs.sum()
    
    projects = []
    
    for _ in range(n_samples):
        ptype = np.random.choice(project_types, p=project_probs)
        industry = np.random.choice(industries, p=industry_probs)
        design = np.random.choice(designs)
        
        # Pages based on project type
        page_ranges = {
            'Landing Page': (1, 6), 'Blog/Magazine': (10, 40), 'Corporate Relaunch': (15, 50),
            'CMS Migration': (20, 80), 'Accessibility Retrofit': (15, 50), 'New Development': (15, 60),
            'E-Commerce/WooCommerce': (10, 40), 'Intranet/Extranet': (30, 100),
            'Multisite Network': (40, 150), 'Web Application': (10, 50)
        }
        min_p, max_p = page_ranges[ptype]
        num_pages = np.random.randint(min_p, max_p + 1)
        
        num_languages = np.random.choice([1, 1, 1, 1, 2, 2, 3, 4])
        
        # Features with correlations
        has_shop = 1 if (ptype == 'E-Commerce/WooCommerce' or np.random.random() < 0.12) else 0
        has_blog = 1 if (ptype == 'Blog/Magazine' or np.random.random() < 0.5) else 0
        has_booking = 1 if np.random.random() < 0.10 else 0
        has_membership = 1 if (ptype == 'Intranet/Extranet' or np.random.random() < 0.08) else 0
        has_events = 1 if np.random.random() < 0.12 else 0
        has_migration = 1 if (ptype == 'CMS Migration' or np.random.random() < 0.30) else 0
        has_api = 1 if (ptype == 'Web Application' or np.random.random() < 0.15) else 0
        
        num_forms = np.random.randint(1, 8)
        num_integrations = min(10, np.random.randint(0, 5) + (2 if has_shop else 0))
        
        if 'Public' in industry:
            accessibility_level = np.random.choice([1, 2, 2, 2, 3])
        else:
            accessibility_level = np.random.choice([0, 0, 0, 1, 2])
        
        client_responsiveness = np.random.choice([1, 2, 2, 2, 3])
        content_readiness = np.random.choice([0, 1, 1, 2, 2])
        decision_makers = np.random.randint(1, 5)
        revision_rounds = np.random.randint(1, 4)
        
        # Calculate hours
        hours = PROJECT_BASE_HOURS[ptype]
        hours *= DESIGN_MULTIPLIERS[design]
        hours += num_pages * 0.5
        if num_languages > 1:
            hours += (num_languages - 1) * 20
        hours += has_shop * 50 + has_blog * 12 + has_booking * 35
        hours += has_membership * 40 + has_events * 18 + has_migration * 30 + has_api * 35
        hours += num_forms * 3 + num_integrations * 12 + accessibility_level * 25
        hours *= {1: 1.12, 2: 1.00, 3: 0.95}[client_responsiveness]
        hours *= {0: 1.10, 1: 1.02, 2: 1.00}[content_readiness]
        hours += decision_makers * 5 + revision_rounds * 6
        hours *= INDUSTRY_MULTIPLIERS[industry]
        
        noise = np.clip(np.random.normal(1.0, 0.04), 0.92, 1.08)
        hours = max(25, round(hours * noise))
        
        projects.append({
            'project_type': ptype, 'industry': industry, 'design_source': design,
            'num_pages': num_pages, 'num_languages': num_languages,
            'has_shop': has_shop, 'has_blog': has_blog, 'has_booking': has_booking,
            'has_membership': has_membership, 'has_events': has_events,
            'has_migration': has_migration, 'has_api': has_api,
            'num_forms': num_forms, 'num_integrations': num_integrations,
            'accessibility_level': accessibility_level,
            'client_responsiveness': client_responsiveness,
            'content_readiness': content_readiness,
            'decision_makers': decision_makers, 'revision_rounds': revision_rounds,
            'hours': hours
        })
    
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
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ])
        pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', config['model'])])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        regressor = pipeline.named_steps['regressor']
        importance = None
        if hasattr(regressor, 'feature_importances_'):
            feat_names = num_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
            importance = pd.DataFrame({'feature': feat_names, 'importance': regressor.feature_importances_}).sort_values('importance', ascending=False)
        
        results[name] = {
            'pipeline': pipeline, 'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred), 'importance': importance,
            'y_test': y_test, 'y_pred': y_pred
        }
    return results

# ==============================================================================
# HELPERS
# ==============================================================================
def get_phases(hours, ptype, migration):
    bases = {
        'Landing Page': {'Discovery': 0.08, 'Design': 0.35, 'Development': 0.38, 'QA': 0.12, 'Launch': 0.07},
        'E-Commerce/WooCommerce': {'Discovery': 0.10, 'Design': 0.18, 'Development': 0.48, 'QA': 0.15, 'Launch': 0.09},
    }
    base = bases.get(ptype, {'Discovery': 0.10, 'Design': 0.22, 'Development': 0.42, 'QA': 0.14, 'Launch': 0.12})
    if ptype in ['Intranet/Extranet', 'Web Application']:
        base = {'Discovery': 0.15, 'Design': 0.15, 'Development': 0.45, 'QA': 0.15, 'Launch': 0.10}
    if migration:
        base['Discovery'] += 0.03; base['Development'] += 0.02
        base['Design'] -= 0.03; base['Launch'] -= 0.02
    return {k: max(0, int(hours * v)) for k, v in base.items()}

def get_team(hours, phases):
    return {
        'Senior Developer': int(phases['Development'] * 0.35),
        'Developer': int(phases['Development'] * 0.65),
        'Designer': int(phases['Design'] * 0.90),
        'Project Manager': int(hours * 0.12),
        'QA Engineer': int(phases['QA'] * 0.85)
    }

def get_confidence(hours, r2):
    uncertainty = max(0.05, min(0.20, (1 - r2) * 0.5))
    return {'low': int(hours * (1 - uncertainty)), 'mid': hours, 'high': int(hours * (1 + uncertainty))}

def get_insights(data, hours):
    insights = []
    if data['num_integrations'] >= 4:
        insights.append(('🔌', 'High Integrations', f"{data['num_integrations']} integrations = +{data['num_integrations']*12}h"))
    if data['num_languages'] >= 3:
        insights.append(('🌍', 'Multilingual', f"{data['num_languages']} languages = +{(data['num_languages']-1)*20}h"))
    if data['accessibility_level'] >= 2:
        insights.append(('♿', ['','WCAG A','WCAG AA','WCAG AAA'][data['accessibility_level']], f"+{data['accessibility_level']*25}h compliance"))
    if data['content_readiness'] == 0:
        insights.append(('📝', 'No Content', '+10% buffer for delays'))
    if data['client_responsiveness'] == 1:
        insights.append(('🐌', 'Slow Client', '+12% for feedback cycles'))
    if 'Public' in data['industry']:
        insights.append(('🏛️', 'Public Sector', f"+{int((INDUSTRY_MULTIPLIERS[data['industry']]-1)*100)}% compliance overhead"))
    return insights[:5]

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    # Header
    st.title("🧠 Project Estimator Pro")
    st.caption("Machine Learning-powered project estimation for WordPress agencies")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Train Model", "⚡ Estimate", "⚙️ Settings", "🧠 How It Works"])
    
    # ==========================================================================
    # TAB 1: TRAIN MODEL
    # ==========================================================================
    with tab1:
        st.header("Train Your Estimation Model")
        st.markdown("""
        The model learns patterns from simulated project data. Configure the training 
        to match your agency's typical project mix for best results.
        """)
        
        st.divider()
        
        # --- STEP 1: DATA SIZE ---
        st.subheader("1️⃣ Training Data Size")
        
        st.markdown("""
        **How many example projects should the model learn from?**
        
        More projects = better pattern recognition, but longer training time.
        For most agencies, 30,000 projects provides an excellent balance.
        """)
        
        n_samples = st.select_slider(
            "Number of training projects",
            options=[10000, 20000, 30000, 50000, 75000],
            value=30000,
            format_func=lambda x: f"{x:,} projects"
        )
        
        size_info = {
            10000: ("⚡ Fast", "~15 seconds", "Good for quick experiments"),
            20000: ("🔄 Balanced", "~25 seconds", "Good accuracy with fast training"),
            30000: ("✅ Recommended", "~35 seconds", "Optimal balance for most agencies"),
            50000: ("🎯 High Accuracy", "~50 seconds", "Best for production models"),
            75000: ("🔬 Maximum", "~75 seconds", "Highest accuracy, slowest training")
        }
        info = size_info[n_samples]
        st.info(f"{info[0]} • Training time: {info[1]} • {info[2]}")
        
        st.divider()
        
        # --- STEP 2: DATA MIX ---
        st.subheader("2️⃣ Project Mix (Preset)")
        
        st.markdown("""
        **What types of projects does your agency typically do?**
        
        The model learns better when training data matches your real project distribution.
        Choose a preset that reflects your agency's focus.
        """)
        
        # Preset selection with cards
        cols = st.columns(3)
        preset_choice = None
        
        for i, (name, config) in enumerate(PRESETS.items()):
            with cols[i]:
                with st.container():
                    st.markdown(f"### {config['icon']} {name.split('(')[0].strip()}")
                    st.caption(config['description'])
                    
                    with st.expander("Details"):
                        st.markdown(config['details'])
        
        selected_preset = st.radio(
            "Select preset",
            options=list(PRESETS.keys()),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # --- STEP 3: TRAIN ---
        st.subheader("3️⃣ Train Models")
        
        st.markdown("""
        **Ready to train!** We'll train 3 different algorithms and compare their accuracy.
        You can then choose the best one for your estimates.
        """)
        
        if st.button("🚀 Generate Data & Train All Models", type="primary", use_container_width=True):
            
            progress = st.progress(0, text="Generating training data...")
            df = generate_training_data(n_samples, selected_preset)
            progress.progress(30, text="Training Gradient Boosting...")
            
            results = train_all_models(df)
            st.session_state.model_results = results
            progress.progress(100, text="Complete!")
            progress.empty()
            
            st.success("✅ All 3 models trained successfully!")
            
            # Data summary
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
                        fig = px.pie(values=df['project_type'].value_counts().values, 
                                    names=df['project_type'].value_counts().index, title='Project Types')
                        fig.update_layout(height=280, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        # --- RESULTS ---
        if st.session_state.model_results:
            results = st.session_state.model_results
            
            st.divider()
            st.header("📊 Model Comparison")
            
            # Results table
            results_df = pd.DataFrame([
                {'Model': name, 'R² Score': r['r2'], 'MAE': r['mae'], 'MAPE': r['mape']}
                for name, r in results.items()
            ]).sort_values('R² Score', ascending=False)
            
            st.dataframe(
                results_df.style.format({'R² Score': '{:.1%}', 'MAE': '±{:.1f}h', 'MAPE': '{:.1%}'})
                .background_gradient(subset=['R² Score'], cmap='RdYlGn')
                .background_gradient(subset=['MAE'], cmap='RdYlGn_r'),
                use_container_width=True, hide_index=True
            )
            
            # Interpretation
            best = results_df.iloc[0]
            if best['R² Score'] >= 0.93:
                st.success(f"🎉 **Excellent accuracy!** R² of {best['R² Score']:.1%} means predictions are highly reliable.")
            elif best['R² Score'] >= 0.85:
                st.info(f"👍 **Good accuracy.** R² of {best['R² Score']:.1%} provides solid estimates.")
            else:
                st.warning(f"⚠️ **Moderate accuracy.** Consider more training data.")
            
            # Visual comparison
            if PLOTLY:
                col1, col2 = st.columns(2)
                with col1:
                    colors = ['#667eea', '#27ae60', '#f39c12']
                    fig = go.Figure([go.Bar(x=results_df['Model'], y=results_df['R² Score']*100, 
                                           marker_color=colors, text=[f"{v:.1%}" for v in results_df['R² Score']], 
                                           textposition='outside')])
                    fig.update_layout(title='R² Score (higher = better)', yaxis_title='%', height=350, yaxis_range=[0, 105])
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = go.Figure([go.Bar(x=results_df['Model'], y=results_df['MAE'], 
                                           marker_color=colors, text=[f"±{v:.1f}h" for v in results_df['MAE']], 
                                           textposition='outside')])
                    fig.update_layout(title='Mean Absolute Error (lower = better)', yaxis_title='Hours', height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Prediction accuracy plot
            best_name = results_df.iloc[0]['Model']
            r = results[best_name]
            
            if PLOTLY:
                st.subheader("🎯 Prediction Accuracy")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=r['y_test'], y=r['y_pred'], mode='markers',
                                        marker=dict(size=4, opacity=0.4, color='#667eea'), name='Predictions'))
                min_v, max_v = min(r['y_test'].min(), r['y_pred'].min()), max(r['y_test'].max(), r['y_pred'].max())
                fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines',
                                        line=dict(color='red', dash='dash'), name='Perfect'))
                fig.update_layout(xaxis_title='Actual Hours', yaxis_title='Predicted Hours', height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points on the red line = perfect predictions. Tighter clustering = higher accuracy.")
            
            # Model selection
            st.divider()
            st.header("✅ Select Your Model")
            
            cols = st.columns(3)
            for i, name in enumerate(results.keys()):
                r = results[name]
                config = MODELS[name]
                is_best = name == best_name
                
                with cols[i]:
                    if is_best:
                        st.markdown(f"### {config['icon']} {name} 🏆")
                    else:
                        st.markdown(f"### {config['icon']} {name}")
                    
                    st.caption(config['tagline'])
                    st.metric("Accuracy (R²)", f"{r['r2']:.1%}")
                    st.metric("Avg Error", f"±{r['mae']:.1f}h")
                    
                    with st.expander("How it works"):
                        st.markdown(config['how_it_works'])
                        st.markdown("**Strengths:**")
                        for s in config['strengths']: st.markdown(f"- ✅ {s}")
                        st.markdown("**Weaknesses:**")
                        for w in config['weaknesses']: st.markdown(f"- ⚠️ {w}")
                    
                    if st.button(f"Use {name.split()[0]}", key=f"sel_{name}", 
                                type="primary" if is_best else "secondary", use_container_width=True):
                        st.session_state.selected_model = name
                        st.success(f"✅ {name} is now active!")
                        st.balloons()
    
    # ==========================================================================
    # TAB 2: ESTIMATE
    # ==========================================================================
    with tab2:
        if not st.session_state.selected_model:
            st.info("👆 Please train and select a model in the **Train Model** tab first.")
            st.stop()
        
        model_name = st.session_state.selected_model
        model_info = st.session_state.model_results[model_name]
        
        st.markdown(f"**Active Model:** {MODELS[model_name]['icon']} {model_name} (R²: {model_info['r2']:.1%})")
        
        st.divider()
        
        col_in, col_out = st.columns([2, 3])
        
        with col_in:
            st.subheader("📝 Project Configuration")
            
            ptype = st.selectbox("Project Type", list(PROJECT_BASE_HOURS.keys()), 
                                help=f"Base hours range from {min(PROJECT_BASE_HOURS.values())}h to {max(PROJECT_BASE_HOURS.values())}h")
            industry = st.selectbox("Client Industry", list(INDUSTRY_MULTIPLIERS.keys()),
                                   help="Industry affects overhead due to compliance, communication style, and processes")
            design = st.selectbox("Design Approach", list(DESIGN_MULTIPLIERS.keys()),
                                 help="Custom design adds significant hours vs. using existing themes")
            
            st.divider()
            st.markdown("**📐 Scope**")
            c1, c2 = st.columns(2)
            pages = c1.slider("Content Pages", 1, 150, 25)
            langs = c2.slider("Languages", 1, 5, 1)
            forms = c1.slider("Complex Forms", 1, 10, 3)
            integrations = c2.slider("Integrations", 0, 10, 2)
            
            st.divider()
            st.markdown("**⚙️ Features**")
            c1, c2 = st.columns(2)
            shop = c1.checkbox("E-Commerce/Shop", help="+50h base")
            blog = c1.checkbox("Blog", True, help="+12h base")
            booking = c1.checkbox("Booking System", help="+35h base")
            membership = c2.checkbox("Membership", help="+40h base")
            events = c2.checkbox("Events", help="+18h base")
            migration = c2.checkbox("Content Migration", help="+30h base")
            api = c1.checkbox("Custom API", help="+35h base")
            
            st.divider()
            st.markdown("**♿ Accessibility**")
            a11y = st.radio("Level", [0, 1, 2, 3], format_func=lambda x: ['None', 'WCAG A (+25h)', 'WCAG AA (+50h)', 'WCAG AAA (+75h)'][x], horizontal=True)
            
            st.divider()
            st.markdown("**👥 Client Factors**")
            c1, c2 = st.columns(2)
            speed = c1.radio("Responsiveness", [1, 2, 3], format_func=lambda x: ['🐌 Slow (+12%)', '👍 Normal', '⚡ Fast (-5%)'][x-1], horizontal=True)
            content = c2.radio("Content Ready", [0, 1, 2], format_func=lambda x: ['❌ No (+10%)', '📝 Partial (+2%)', '✅ Yes'][x], horizontal=True)
            makers = st.slider("Decision Makers", 1, 6, 2, help="+5h per person")
            revisions = st.slider("Revision Rounds", 1, 5, 2, help="+6h per round")
        
        with col_out:
            inp = {
                'project_type': ptype, 'industry': industry, 'design_source': design,
                'num_pages': pages, 'num_languages': langs,
                'has_shop': int(shop), 'has_blog': int(blog), 'has_booking': int(booking),
                'has_membership': int(membership), 'has_events': int(events),
                'has_migration': int(migration), 'has_api': int(api),
                'num_forms': forms, 'num_integrations': integrations,
                'accessibility_level': a11y, 'client_responsiveness': speed,
                'content_readiness': content, 'decision_makers': makers, 'revision_rounds': revisions
            }
            
            hours = int(model_info['pipeline'].predict(pd.DataFrame([inp]))[0])
            conf = get_confidence(hours, model_info['r2'])
            phases = get_phases(hours, ptype, migration)
            team = get_team(hours, phases)
            cost = sum(h * st.session_state.rates.get(r, 120) for r, h in team.items())
            mult = INDUSTRY_MULTIPLIERS.get(industry, 1.0)
            
            # Main result
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;padding:30px;text-align:center;color:white;margin-bottom:20px;">
                <p style="margin:0;opacity:.8;font-size:14px;">ESTIMATED HOURS</p>
                <h1 style="font-size:72px;margin:10px 0;font-weight:800">{hours}</h1>
                <p style="margin:0;opacity:.7">Confidence Range: {conf['low']} – {conf['high']} hours</p>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{hours/40:.1f} weeks")
            c2.metric("Base Cost", f"€{cost:,}")
            c3.metric("Industry Buffer", f"+{int((mult-1)*100)}%")
            c4.metric("Total", f"€{int(cost*mult):,}")
            
            insights = get_insights(inp, hours)
            if insights:
                st.divider()
                st.subheader("💡 Key Factors")
                for icon, title, text in insights:
                    st.markdown(f"**{icon} {title}** — {text}")
            
            st.divider()
            st.subheader("📊 Phase Breakdown")
            if PLOTLY:
                colors = ['#3498db', '#9b59b6', '#27ae60', '#f39c12', '#e74c3c']
                fig = go.Figure()
                for i, (ph, h) in enumerate(phases.items()):
                    if h > 0:
                        fig.add_trace(go.Bar(x=[h], y=[ph], orientation='h', marker_color=colors[i],
                                            text=f"{h}h ({h/hours*100:.0f}%)", textposition='inside', name=ph))
                fig.update_layout(height=200, showlegend=False, yaxis={'categoryorder': 'array', 'categoryarray': list(phases.keys())[::-1]}, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("👥 Team & Cost")
            tdf = pd.DataFrame([{'Role': r, 'Hours': h, 'Rate': f"€{st.session_state.rates.get(r, 120)}", 'Cost': h * st.session_state.rates.get(r, 120)} for r, h in team.items() if h > 0])
            st.dataframe(tdf.style.format({'Cost': '€{:,.0f}'}), hide_index=True, use_container_width=True)
            
            st.markdown(f"**Base:** €{cost:,} + **Buffer ({int((mult-1)*100)}%):** €{int(cost*mult)-cost:,} = **Total: €{int(cost*mult):,}**")
            
            st.divider()
            c1, c2 = st.columns([3, 1])
            name = c1.text_input("Save estimate as", f"{ptype} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            if c2.button("💾 Save", use_container_width=True):
                st.session_state.saved_estimates.append({'name': name, 'hours': hours, 'cost': int(cost*mult), 'date': datetime.now().isoformat()})
                st.success("Saved!")
            
            if st.session_state.saved_estimates:
                with st.expander(f"📁 Saved Estimates ({len(st.session_state.saved_estimates)})"):
                    for s in st.session_state.saved_estimates[::-1]:
                        st.markdown(f"**{s['name']}** — {s['hours']}h / €{s['cost']:,}")
    
    # ==========================================================================
    # TAB 3: SETTINGS
    # ==========================================================================
    with tab3:
        st.header("⚙️ Settings")
        
        st.subheader("💰 Hourly Rates")
        st.markdown("""
        Configure your agency's hourly rates by role. These are used to calculate 
        project costs based on the estimated hours per role.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Development Roles**")
            st.session_state.rates['Senior Developer'] = st.number_input(
                "Senior Developer (€/h)", 80, 250, st.session_state.rates['Senior Developer'], 5,
                help="Typically handles architecture, complex features, code review"
            )
            st.session_state.rates['Developer'] = st.number_input(
                "Developer (€/h)", 60, 200, st.session_state.rates['Developer'], 5,
                help="Implements features, templates, standard functionality"
            )
            st.session_state.rates['QA Engineer'] = st.number_input(
                "QA Engineer (€/h)", 50, 180, st.session_state.rates['QA Engineer'], 5,
                help="Testing, bug verification, quality assurance"
            )
        
        with col2:
            st.markdown("**Design & Management**")
            st.session_state.rates['Designer'] = st.number_input(
                "Designer (€/h)", 60, 200, st.session_state.rates['Designer'], 5,
                help="UI/UX design, visual design, prototyping"
            )
            st.session_state.rates['Project Manager'] = st.number_input(
                "Project Manager (€/h)", 70, 200, st.session_state.rates['Project Manager'], 5,
                help="Planning, client communication, coordination"
            )
        
        st.divider()
        
        st.subheader("📊 Current Model")
        if st.session_state.selected_model:
            r = st.session_state.model_results[st.session_state.selected_model]
            col1, col2, col3 = st.columns(3)
            col1.metric("Active Model", st.session_state.selected_model)
            col2.metric("Accuracy (R²)", f"{r['r2']:.1%}")
            col3.metric("Avg Error", f"±{r['mae']:.1f}h")
        else:
            st.info("No model trained yet. Go to **Train Model** tab.")
        
        st.divider()
        
        st.subheader("🗑️ Reset")
        if st.button("Clear All Saved Estimates"):
            st.session_state.saved_estimates = []
            st.success("All estimates cleared!")
    
    # ==========================================================================
    # TAB 4: HOW IT WORKS
    # ==========================================================================
    with tab4:
        st.header("🧠 How This AI Works")
        st.markdown("*A guide for non-technical readers*")
        
        st.divider()
        
        st.subheader("📖 The Core Concept")
        st.markdown("""
        Imagine you're a senior project manager with **20 years of experience**. After seeing 
        thousands of projects, you develop intuition:
        
        > *"E-commerce projects with custom design usually take around 300 hours..."*
        > 
        > *"Public sector clients need 30% more time due to compliance..."*
        > 
        > *"Every additional language adds about 20 hours of work..."*
        
        **Machine Learning does exactly this — but with mathematics instead of intuition.**
        
        We show the computer 30,000 example projects. It analyzes patterns and learns rules 
        like "integrations add 12 hours each" without being told explicitly.
        """)
        
        st.divider()
        
        st.subheader("🎯 The Three Algorithms")
        
        for name, config in MODELS.items():
            with st.expander(f"{config['icon']} {name} — {config['tagline']}"):
                st.markdown(config['how_it_works'])
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strengths:**")
                    for s in config['strengths']:
                        st.markdown(f"- ✅ {s}")
                with col2:
                    st.markdown("**Weaknesses:**")
                    for w in config['weaknesses']:
                        st.markdown(f"- ⚠️ {w}")
                st.info(f"**Best for:** {config['best_for']}")
        
        st.divider()
        
        st.subheader("📊 Understanding the Metrics")
        
        st.markdown("""
        | Metric | What it measures | Good value | Example |
        |--------|------------------|------------|---------|
        | **R² Score** | How much variation the model explains | > 90% | R²=0.95 means 95% of differences between projects are understood |
        | **MAE** | Average prediction error in hours | < 20h | MAE=15h means predictions are typically off by 15 hours |
        | **MAPE** | Average error as percentage | < 12% | MAPE=8% means a 200h project might be 184-216h |
        """)
        
        st.divider()
        
        st.subheader("🔢 The Hour Calculation Formula")
        
        st.markdown("""
        The training data uses this formula (which the model learns to approximate):
        
        ```
        Hours = Base Hours (45-350h by project type)
              × Design Multiplier (0.65-1.25)
              + Pages × 0.5h
              + Extra Languages × 20h
              + Features (Shop: +50h, Blog: +12h, API: +35h, etc.)
              + Forms × 3h
              + Integrations × 12h
              + Accessibility Level × 25h
              × Client Speed Factor (0.95-1.12)
              × Content Readiness Factor (1.00-1.10)
              + Decision Makers × 5h
              + Revision Rounds × 6h
              × Industry Multiplier (0.92-1.30)
        ```
        
        The model **discovers these patterns** from the data without being told the formula.
        That's the power of machine learning!
        """)
        
        st.divider()
        
        st.subheader("⚠️ Important Limitations")
        
        st.warning("""
        **This model is trained on simulated data**, not real projects.
        
        For production use:
        - Treat estimates as **starting points**, not guarantees
        - Apply **human judgment** for unusual projects
        - Consider **retraining with your actual project history** for best accuracy
        - The confidence range shows expected uncertainty
        """)

if __name__ == "__main__":
    main()
