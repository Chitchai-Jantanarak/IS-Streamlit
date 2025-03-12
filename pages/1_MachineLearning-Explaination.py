import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import tempfile

@st.cache_resource
def get_ydata_profiling(df):
    profile = ProfileReport(
        df,
        title="Profiling Report",
        html={"style": {"full_width": True}},
        sort=None,
        explorative=True
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        profile.to_file(tmp_file.name)

        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_code = f.read()

    return html_code

def get_dataframe() -> pd:
    data = [
            ["studyName", "object"],
            ["Sample Number", "int64"],
            ["Species", "object"],
            ["Region", "object"],
            ["Island", "object"],
            ["Stage", "object"],
            ["Individual ID", "object"],
            ["Clutch Completion", "object"],
            ["Date Egg", "object"],
            ["Culmen Length (mm)", "float64"],
            ["Culmen Depth (mm)", "float64"],
            ["Flipper Length (mm)", "float64"],
            ["Body Mass (g)", "float64"],
            ["Sex", "object"],
            ["Delta 15 N (o/oo)", "float64"],
            ["Delta 13 C (o/oo)", "float64"]
        ]

    df = pd.DataFrame(data, columns=["Column Name", "Data Type"])
    
    return df

def preparation(df) -> None :
    st.markdown("<a id='dataset-preparation'></a>", unsafe_allow_html=True)
    st.markdown("""
    ## Dataset Preparation

    การเตรียมข้อมูล เป็นข้อมูลที่หยิบมาใช้จาก   "[kaggle](https://www.kaggle.com/)" <br />
    [<span style="color:blue">ข้อมูลของเพนกวินในหมู่เกาะ Palmer Archipelago](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)
    """, unsafe_allow_html=1)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ### Data Set
    เป็นองค์ข้อมูลโดยรวมของประชากรเพนกวินที่พบเจอในหมู่เกาะ [Palmer Archipelago](https://en.wikipedia.org/wiki/Palmer_Archipelago) ที่พบเจอ
    และ ที่สามารถเก็บข้อมูลได้ตามเกาะต่างๆในหมู่เกาะนี้ โดยข้อมูลที่รวบรวมนั้นถูกรวบรวมโดยสมาชิกของ [เครือข่ายการวิจัยระบบนิเวศในระยะยาว Long Term Ecological Research Network](https://lternet.edu/)
                
    ข้อมูลต่าง ๆ จะประกอบไปด้วยตัวแปร ดังนี้
                
        1.   studyName            KeyCode โครงสร้างการวิจัย    
        2.   Sample Number        ลำดับของกลุ่มตัวอย่าง
        3.   Species              Species ของเพนกวิน (Chinstrap, Adélie, or Gentoo)
        4.   Region               สัญชาติ : ถิ่นเกิด
        5.   Island               เกาะที่อาศัย (ขณะที่เก็บข้อมูล)
        6.   Stage                ช่วงอายุขัย
        7.   Individual ID        ID ระบุเฉพาะตัว (marking)
        8.   Clutch Completion    สถานะการวางไข่ (เพื่อวิจัยความสำเร็จในการสืบพันธุ์)
        9.   Date Egg             วันที่ออกไข่   
        10.  Culmen Length (mm)   ความยาวของจะงอย   
        11.  Culmen Depth (mm)    ความกว้างของจะงอย
        12.  Flipper Length (mm)  ความยาวของปีก
        13.  Body Mass (g)        น้ำหนักตัว : grams
        14.  Sex                  เพศ
        15.  Delta 15 N (o/oo)    อัตราเสถียรของ Isotope N (อัตราการปรับตัว, ระดับโภชนาการ)
        16.  Delta 13 C (o/oo)    อัตราเสถียรของ Isotope C (อัตราการปรับตัว, ระดับโภชนาการ)
        17.  Comments             หมายเหตุเพิ่มเติม
                
    """)

    st.dataframe(df)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    จากการนำข้อมูลมาวิเคราะห์โดยรวม ตัวช้อมูลสามารถนำมาทำได้ทั้ง
    - classification = จากข้อมูลที่สามารถแยกแยะเป็นส่วน ๆ ได้ เช่น **Species**, **Island**
    - regression = ข้อมูลที่เป็นค่าตัวเลขต่าง ๆ เช่น **Body Mass**, **Filpper Length** หรือข้อมูลทางกายภาพของเพนกวินแต่ละตัว
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ### ซึ่งจะทำทั้งโมเดลทั้ง 2 ชนิด
    #### - classification : **SVM** (support vector machine)
    #### - regression : **RF** (random forest)
    """)

    st.markdown("---")

def workflow(df) -> None :

    def svm_explaination() -> None :

        def highlight_rows(row):
            styles = ['color: grey'] * len(row)

            if row["Column Name"] in [
                "Clutch Completion", "Date Egg", "Culmen Length (mm)",
                "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)",
                "Sex", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"
            ]:
                return ['color: blue'] * len(row)
            
            if row["Column Name"] in [
                "Species"
            ]:
                return ['color: red'] * len(row)

            return styles

        def workflow():
            st.markdown("<a id='svm_workflow'></a>", unsafe_allow_html=True)
            st.markdown("""
            ## SVM - Classification
                        
            ### WORKFLOW

            ###### เหตุผลที่ใช้ SVM model
                    
            Support vector machines (SVM) เป็นโมเดลเพื่อสำหรับการ Classification 
            ใช้เพื่อการแบ่งกลุ่มตามอัตราส่วนต่างๆ โดยมี kernel เป็นตัวกำหนดเส้นแล้วยังมี margin
            สำหรับการแยกช่วงกับกลุ่มที่ใกล้กันมากๆ

            ###### การหยิบใช้ dataFrame
            จะใช้ตัว SVM เพื่อ classifacate ตามข้อมูลเพื่อแยกแยะ Species ต่างๆของเพนกวิน
            """)
            df = get_dataframe()
            styled_df = df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col3:
                st.markdown("""
                :grey[== Variant ที่ไม่ได้ใช้]\n
                :red[== ค่า y (ค่าที่ต้องการ classification)]\n  
                :blue[== ค่า x (parameter)]    
                """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### 1. การทำโมเดล
                        
            ระหว่างการทำโมเดลจะใช้ ```StandardScaler``` เป็นตัว preprocessing เพื่อ standardizes 
            ให้ข้อมูลแต่ละ feature เพราะตัว SVM ใช้เรื่องของ d ของแต่ละข้อมูลจึงจับทำเป็น group 
            จึงจำเป็นต้องมีการปรับขนาดและนำมาใช้ระหว่าง evaluation ( boundaries balancing )

            ###### KERNEL : rbf (Gaussian)
            """)
            st.image('public/machine_learning/model1_SVM/workflow_rbf.png')
            st.markdown("""
            ```python
            # __builder_model.py

            def build_svm_model():
                        
                # SVM Builder by get cleaned data and classify prepare data
                #
                # Args:
                #   model (sklearn.svm.SVC)
                #   scaler (sklearn.preprocessing.StandardScaler)
                #   X_test (Array)
                #   y_test (Array)
                #
                # Returns:
                #     Model, Scaler, (sklearn.model_selection.train_test_split)

                df = data_cleansing()
                
                data_prep = df[['Body Mass (g)', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 
                            'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 
                            'Sex', 'Clutch Completion', 'Species']]
                data_prep = pd.get_dummies(data_prep, columns=['Species'])  # One-hot encoding
                

                X = data_prep.drop(columns=['Species_Adelie Penguin (Pygoscelis adeliae)', 
                                        'Species_Chinstrap penguin (Pygoscelis antarctica)', 
                                        'Species_Gentoo penguin (Pygoscelis papua)'])
                y = df['Species']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                svm_model = SVC(kernel='rbf', random_state=42)
                svm_model.fit(X_train_scaled, y_train)
                
                # Save
                joblib.dump(svm_model, 'svm_model.pkl')
                joblib.dump(scaler, 'scaler.pkl')
                
                return svm_model, scaler, X_train, X_test, y_train, y_test, X, y
            ```        
            """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### 2. ประเมิน และ ทดสอบโมเดล
                        
            มีไว้เพื่อทดสอบการดึงข้อมูล และนำมาทดสอบ ผ่านการทำ Scaler ควบคู่ไปด้วย
                        
            ```python
            # __evaluation_model.py
                        
            # SVM Evaluation by making confusion matrix
            #
            # Returns:
            #     Accuracy, confusion matix, classification report

            def evaluate_model(model=None, scaler=None, X_test=None, y_test=None):

                if model is None or X_test is None or y_test is None:
                    model = joblib.load('svm_model.pkl')
                    scaler = joblib.load('scaler.pkl')
                
                # Scale test data
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test) # Take Standardizes on X_test
                else:
                    X_test_scaled = X_test  # Take an examine (IF NOT SCALER)
                
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Adelie', 'Chinstrap', 'Gentoo'], 
                            yticklabels=['Adelie', 'Chinstrap', 'Gentoo'])
                plt.title("Confusion Matrix")
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig('confusion_matrix.png', dpi=300)
                plt.close()
                
                return accuracy, conf_matrix, class_report
            ```
            """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### 3. PCA - Principal component analysis
            
            เพื่อให้ง่าย และ เห็นภาพช่วงของกลุ่มได้ง่ายยิ่งขึ้น และสามารถอธิบายว่า feature ต่าง ๆ
            ที่อยู่ใน parameter ต่างๆ (x) มีความสำคัญมากแค่ไหนในการจัดกลุ่ม

            ```python
            # Caller function
            def visualize_pca_and_decision_boundary(X=None, y=None, model=None, scaler=None):
                
                if X is None or y is None or model is None:
                    model = joblib.load('svm_model.pkl')
                    scaler = joblib.load('scaler.pkl')
                    
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y)
                
                X_scaled = scaler.transform(X)
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                plt.figure(figsize=(10, 8))
                
                # Set species choice in dataset
                species_names = ['Adelie', 'Chinstrap', 'Gentoo']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                markers = ['o', 's', '^']
                
                # Transform encoded to plot
                for i, species in enumerate(np.unique(y_encoded)):
                    idx = y_encoded == species
                    plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                            c=colors[i],
                            s=80,
                            marker=markers[i],
                            edgecolor='k',
                            linewidth=1,
                            alpha=0.8,
                            label=species_names[i])
                
                plt.title('Penguin Species in PCA Space', fontsize=14)
                plt.xlabel('Principal Component I')
                plt.ylabel('Principal Component II')
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                
                var_ratio = pca.explained_variance_ratio_
                var_text = f'PC1: {var_ratio[0]:.2%} var PC2: {var_ratio[1]:.2%} var'
                plt.annotate(var_text, xy=(0.02, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                plt.savefig('pca_visualization.png', dpi=300)
                plt.close()

                # Boundary plot
                plot_decision_boundary(X_scaled, y_encoded, model, pca)
                
                # Create PCA loadings
                plot_pca_loadings(pca, X)

            def plot_decision_boundary(X_scaled, y_encoded, model, pca):

                X_pca = pca.transform(X_scaled)
                
                svm = SVC(kernel='rbf', random_state=42)
                svm.fit(X_pca, y_encoded)
                
                # Create a mesh grid
                x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                    np.arange(y_min, y_max, 0.02))
                
                # meshgrid prediction
                Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
                
                species_names = ['Adelie', 'Chinstrap', 'Gentoo']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                markers = ['o', 's', '^']
                
                # Get unique numeric classes (0, 1, 2)
                for i in range(3):  # 3 Species
                    idx = y_encoded == i
                    ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                            c=colors[i],
                            s=80,
                            marker=markers[i],
                            edgecolor='k',
                            linewidth=1,
                            alpha=0.8,
                            label=species_names[i])
                
                ax.set_title("SVM Decision Boundary (PCA-reduced Data)", fontsize=16)
                ax.set_xlabel(f'Principal Component 1', fontsize=14)
                ax.set_ylabel(f'Principal Component 2', fontsize=14)

                fig.colorbar(contour, ax=ax, label='Class')
                
                ax.legend(fontsize=12, loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.6)
                
                kernel_name = svm.kernel.upper() if hasattr(svm, 'kernel') else 'Unknown'
                c_value = svm.C if hasattr(svm, 'C') else 'Unknown'
                
                textstr = f'Model: SVM ({kernel_name} kernel) C parameter: {c_value}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
                
                if hasattr(pca, 'explained_variance_ratio_'):
                    var_ratio = pca.explained_variance_ratio_
                    var_text = f'PC1: {var_ratio[0]:.2%} variance PC2: {var_ratio[1]:.2%} variance'
                    ax.text(0.05, 0.85, var_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                
                plt.tight_layout()
                plt.savefig('svm_decision_boundary.png', dpi=300)
                plt.close()

            def plot_pca_loadings(pca, X):

                # Plot PCA feature as a heatmap

                # Get feature names from X
                feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
                
                # PCA loadings
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=['PC1', 'PC2'],
                    index=feature_names
                )
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title('Feature Contributions to Principal Components', fontsize=14)
                plt.tight_layout()
                plt.savefig('pca_feature_contributions.png', dpi=300)
                plt.close()
            ```
            """)

            st.markdown(""" --- """)

        def conclusion():
            st.markdown("<a id='svm_conclusion'></a>", unsafe_allow_html=True)
            st.markdown("""
            ## Conclusion
                        
            จากการเทรนมีผลสรุปการทดสอบโมเดล มีผลสรุปเป็น
            - confusion matrix
            - PCA feature contribution
            - PCA visualization
            - decision boundaries
                        
            ##### ผลสรุปที่ได้
            ตัวโมเดลสามารถคาดการณ์ และแบ่งกลุ่มได้ตามขนาดต่างๆ โดยมีอัตราส่วนที่มีใน **PCA component 2 ตัว**
            > **Principle Component I**  : ให้ความสำคัญกับ **['Body Mass (g)', 'Flipper Length (mm)']** น้ำหนักตัวที่เป็นตัวกลาง ที่กำหนดค่าต่างๆ
            
            > **Principle Component II** : ให้ความสำคํญกับ **['Sex']** หรือ เพศของเพนกวินเป็นหลัก ซึ่งเพศจะส่งผลกับข้อมูลทางกายภาพเช่นกัน
                        
            ###### Species 
            - Gentoo สามารถแบ่งแยกได้ชัด ตามข้อมูลทางกายภาพ (PC 1)
            - Adelie ,Chinstrap มีความใกล้เคียงกัน ถูกอิงความต่างกับ (PC 2)
                        
            ###### ส่วนผลสรุปนี้วิธีทำต่าง ๆ สามารถดูได้ [ที่นี่](https://github.com/Chitchai-Jantanarak/IS-Streamlit/tree/main/train/machine_learning)
            """,)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### confusion matrix
            """)
            st.image("public/machine_learning/model1_SVM/confusion_matrix.png")

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### feature contribution
            """)
            st.image("public/machine_learning/model1_SVM/pca_feature_contributions.png")

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### visualization
            """)
            st.image("public/machine_learning/model1_SVM/pca_visualization.png")

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### visualization + decision boundaries
            """)
            st.image("public/machine_learning/model1_SVM/svm_decision_boundary.png")

        workflow()
        conclusion()

    def rf_explaination() -> None :
        def highlight_rows(row):
            styles = ['color: grey'] * len(row)

            if row["Column Name"] in [
                "Species", "Sex", "Island", "Clutch Completion",
                "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Culmen Length (mm)", 
                "Culmen Depth (mm)", "Flipper Length (mm)"
            ]:
                return ['color: blue'] * len(row)
            
            if row["Column Name"] in [
                "Body Mass (g)"
            ]:
                return ['color: red'] * len(row)

            return styles

        def workflow():
            st.markdown("<a id='svm_workflow'></a>", unsafe_allow_html=True)
            st.markdown("""
            ## RF - Regression
                        
            ### WORKFLOW

            ###### เหตุผลที่ใช้ Random forest model
                    
            Random forest (RF) เป็นโมเดลแบบ Unsupervised 
            เพื่อสำหรับการ Classification และ Regression ตรวจจับ Pattern
            โดยใช้หน้าที่ในการทำ Decision Tree หลายๆ อันและนับมารวมกันแล้วนำ sub-sample
            ที่แยกย่อยแบบสุ่มนำมาประมวลผล และ ป้องกันการ overfit

            ###### การหยิบใช้ dataFrame
            จะใช้ตัว RF **เพื่อคาดเดา** น้ำหนักตัวของเพนกวิน ['body mass (g)'] 
            """)
            df = get_dataframe()
            styled_df = df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col3:
                st.markdown("""
                :grey[== Variant ที่ไม่ได้ใช้]\n
                :red[== ค่า y (ค่าที่ต้องการทำนาย)]\n  
                :blue[== ค่า x (parameter)]    
                """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### 1. การทำโมเดล

            ก่อนการทำโมเดลจะแยก ข้อมูลที่เป็น **Categorical** และ **Numerical** เพื่อการเตรียมข้อมูล หรือ Preprocessor
            จะแปลงข้อมูลให้อยู่ในรูปที่เหมาะสมต่อการทำ Decision โดยข้อมูล
            - Categorical : One hot encoder แยกเป็น feature แทน
            - Numerical   : Standardize
                        
            e.g. ข้อมูลตัวอย่าง [-0.97427508  0.34113484 -0.59372706 ...  0. 1. 0.]
                        
            จากข้อมูลทั้ง Categorical 0:1, Numerical ค่า Standard (z-score)
                                    
            จะใช้ ```Pipeline``` จาก sklearn.pipeline นำรวบรวมตัวข้อมูลไว้เนื่องจากใช้เพื่อการหา Hyperparameter, Tuning
            ในการทำ และจำเป็นต้องมีตัว preprocessor หรือข้อมูลที่จัดเรียงประเภทข้อมูล
            """)
            st.markdown("""
            ```python
            # __builder_model.py

            def build_svm_model():
                        
                # SVM Builder by get cleaned data and classify prepare data
                #
                # Args:
                #   model (sklearn.svm.SVC)
                #   X_test (Array)
                #   y_test (Array)
                #   X (Array)
                #   y (Array)
                #
                # Returns:
                #     Model, (sklearn.model_selection.train_test_split), X, y

                df = data_cleansing()

                categorical_feat = ['Species', 'Sex', 'Island', 'Clutch Completion']
                numerical_feat   = ['Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 'Culmen Length (mm)', 
                                    'Culmen Depth (mm)', 'Flipper Length (mm)']
                
                X = df[categorical_feat + numerical_feat]
                y = df['Body Mass (g)']

                preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_feat),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feat)
                ])

                rf_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(
                        n_estimators=275, 
                        max_features='log2', 
                        max_depth=12, 
                        min_samples_split=7,
                        min_samples_leaf=2, 
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                rf_pipeline.fit(X_train, y_train)

                # Save feature for sending to another files
                feat_datas = {
                    'numerical_feat': numerical_feat,
                    'categorical_feat': categorical_feat
                }

                # Save
                joblib.dump(rf_pipeline, 'rf_model.pkl')
                joblib.dump(feat_datas, 'feature_data.pkl')

                return rf_pipeline, X_train, X_test, y_train, y_test, X, y
            ```        
            """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### 2. ประเมิน และ ทดสอบโมเดล
                        
            เนื่องจากเป็น Unsupervised โมเดล จึงจำเป็นต้องวิธีในการเทรน และเทสข้อมูลด้วยวิธีต่าง
            ในนี้จึงใช้ Cross-Validation หรือ [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
                        
            ```python
            # __evaluation_model.py
                        
            # Evaluate by using sklearn.model_selection.cross_val_score
            # Args:
            #   model (sklearn.pipeline)
            #   X_test (Array)
            #   X_train (Array)
            #   y_test (Array)
            #   y_train (Array)
            # Returns:
            #     y_prediction, mean_squared_err, mean_absolute_err, r2, cross_val_score

            def evaluate_model(model=None, X_test=None, X_train=None, y_test=None,  y_train=None):

                if (    
                    model is None or 
                    X_test is None or 
                    X_train is None or 
                    y_test is None or 
                    y_train is None
                ):
                    try:
                        model = joblib.load('svm_model.pkl')
                        _, X_train, X_test, y_train, y_test, _, _ = build_rf_model()
                    except FileNotFoundError:
                        model, X_train, X_test, y_train, y_test, _, _ = build_rf_model()


                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, scoring='neg_root_mean_squared_error'
                )

                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print("-----------------------------")
                print("Model performance on test set:")
                print(f"Root Mean Squared Error: {rmse:.2f} grams")
                print(f"Mean Absolute Error: {mae:.2f} grams")
                print(f"R² Score: {r2:.4f}")
                print("-----------------------------")

                return y_pred, rmse, mae, r2, cv_scores
            ```
            """)
            st.image("public/machine_learning/model2_RF/workflow_evaluate.jpg", use_container_width=1)
            st.markdown(""" --- """)

        def conclusion():
            st.markdown("<a id='svm_conclusion'></a>", unsafe_allow_html=True)
            st.markdown("""
            ## Conclusion
                        
            จากการเทรนมีผลสรุปการทดสอบโมเดล มีผลสรุปเป็น
            - Regression ค่าจริง : ค่าทำนาย
            - Regression Residual
            - Feature importance
            - Morphology features
            - Prediction Distribution (ตัวอย่าง feature)
                        
            ##### ผลสรุปที่ได้
            - ตัวโมเดลอาจจะได้ค่าประมาณที่ดี แต่ยังมี Residual ที่ยังกระจุกเป็นกลุ่มอยู่ทำให้ค่าที่วัดอาจเพี้ยนเล็กน้อย จาก Residual และ Morphology features
            ที่บอกถึงค่า Accuracy ที่ test มีจุดที่ :orange[Prediction Error] อยู่บางจุด
            - หากเทียบการกระจายของข้อมูล body mass : Species จะทำให้เห็น (Adelie, Chinstrap) Overlapped กัน
            และ Gentoo ถูกแยกออกมา ตัวโมเดลจะต้องแยกข้อมูลของ (Adelie, Chinstrap) ได้ชัดเจนจาก feature อื่นๆ
                        
            ###### ส่วนผลสรุปนี้วิธีทำต่าง ๆ สามารถดูได้ [ที่นี่](https://github.com/Chitchai-Jantanarak/IS-Streamlit/tree/main/train/machine_learning)
            """,)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### Regression ค่าจริง : ค่าทำนาย
            """)
            st.image("public/machine_learning/model2_RF/actual_predicted.png")

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### Regression Residual
            """)
            st.image("public/machine_learning/model2_RF/residuals.png")

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### features importance
            """)
            st.image("public/machine_learning/model2_RF/feature_importance.png")
            st.markdown("""
            การจัดสรรความสำคัญของ features
            1. ลักษณะทางกายภาพ
            2. สายพันธุ์ Gentoo (มีลักษณะทางกายภาพต่างจากสายพันธุ์อื่น)
            3. เกาะ
            4. เพศ
            """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### morphology features
            """)
            st.image("public/machine_learning/model2_RF/morphology_accuracy.png")
            st.markdown("""
            บอกถึง การทำนายที่คลาดเคลื่อนต่อการแยกฟีเจอร์หลายๆรูปแบบ 
                        
            ยังพบเจอค่าที่ทำนายผิดพลาดในช่วง [0.8, 1] (สีเหลือง) อยู่
            """)

            st.markdown(""" <br /> """, unsafe_allow_html=1)
            st.markdown("""
            ##### prediction distribution < Species : Y >
            """)
            st.image("public/machine_learning/model2_RF/prediction_distribution_species.png")

        workflow()
        conclusion()

    st.markdown('<a id="workflow_a"></a>', unsafe_allow_html=True)
    st.markdown("""
    ## WORKFLOW
    ใน workflow นี้จะทำการ EDA และ data cleansing ก่อนนำมาทำ Model ต่าง ๆ
    ลดจำนวน outlier และ foreigness ออกจาก records
    """,)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ### EDA : exploratory data analysis
    ในขั้นตอนการวิเคราะห์ข้อมูลผ่าน eda จะใช้เป็น ```ProfileReport``` ของ [ydata_profiling](https://docs.profiling.ydata.ai/latest/)
    """)

    html_code = get_ydata_profiling(df)
    st.components.v1.html(html_code, height=800, scrolling=True)

    st.markdown("""
    #### ผลสรุปจากการ EDA
    - ส่วนที่ต้องตัดทอน และ ลบข้อมูล
        - Region        : มีค่า unique ค่าเดียว _// ไม่หยิบใช้_
        - Stage         : มีค่า unique ค่าเดียว _// ไม่หยิบใช้_
        - Sex           : มีข้อมูลที่ไม่พบ (none, **' . '**) _// ตัดทิ้ง_
        - δ13C, δ15N    : ข้อมูลหาย หรือ ไม่มีอยู่เยอะมาก _// ควรเติม_
        - Comments      : หมายเหตุแต่ละข้อมูลไม่จำเป็นในการหยิบใช้ _// ตัดทิ้ง_                   
    - ข้อสังเกต
        - ข้อมูลที่ความเกี่ยวเนื่องกันมาก (High correlation) ตัว Model จะต้องมีการเชื่อมหลายๆ feature
        - อัตราส่วนของทางกายภาพสามารถจัดแบ่งได้เป็นกลุ่มอย่างชัดเจน
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ### Data cleansing
    จะนำเอาข้อมูลที่วิเคราะห์จากผลสรุป EDA ที่ได้ใช้นำมาเติมข้อมูล และ ลบข้อมูล
                
    #### Strategy
    1. ลบ ฟีเจอร์ 'Comment' ออก
    2. ลบข้อมูลแปลกปลอม และ ไม่มีค่าใน ฟีเจอร์ 'Sex'
    3. เติม Mean เข้าไปใน ฟีเจอร์ ['δ13C', 'δ15N'] โดยขึ้นอยู่กับ Species จากข้อมูลสายพันธุ์ต่าง ๆ มีอัตราการปรับตัว และ ระดับโภชนาที่ไม่เท่ากัน
    จึงหยิบใช้ Species ในการ Zip ข้อมูล
    4. เข้ารหัสข้อมูลจาก Enumerate (String) => Boolean
                
    ###### ปล. ข้อมูลที่มีค่าโดด หรือ feature ต่างๆ จะถูกดัดแปลงอีกในแต่ละ Model
                
    ```python
    # __EDA.py

    def data_cleansing() -> pd:

        def processing(df) -> pd:
            df = df.drop( columns = ["Comments"] ) # Drop Unnecessary Attr.
            df = df.dropna(subset = ["Sex"])       # Drop Na

            # Found unknown value on 'Sex'
            df = df[df['Sex'] != '.']
                
            # Fill Na based on Species <Mapping>
            species_mean15 = df.groupby('Species')['Delta 15 N (o/oo)'].mean()
            species_mean_dict15 = species_mean15.to_dict()

            species_mean13 = df.groupby('Species')['Delta 13 C (o/oo)'].mean()
            species_mean_dict13 = species_mean13.to_dict()

            # Loop dict & set
            for s, val in species_mean_dict15.items():
                df.loc[(df['Species'] == s) & (df['Delta 15 N (o/oo)'].isna()), 'Delta 15 N (o/oo)'] = val

            for s, val in species_mean_dict13.items():
                df.loc[(df['Species'] == s) & (df['Delta 13 C (o/oo)'].isna()), 'Delta 13 C (o/oo)'] = val


            # ENCODING
            df.loc[:, 'Clutch Completion'] = df['Clutch Completion'].map({'Yes': 1, 'No': 0})
            df.loc[:, 'Sex'] = df['Sex'].map({'MALE': 1, 'FEMALE': 0})

            return df
        
        # Configurate the file path from here kub :)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_dir, 'penguins_lter.csv')
        df = pd.read_csv(file_path)

        return processing(df)
    ```
                
    จากไฟล์ ```__EDA.py``` จะถูกนำหยิบใช้ต่อจากโมเดลอื่นๆ อีกจึงได้ทำเป็นฟังก์ขั่นไว้
                
    ##### จำนวน Data & Data type ของตัวแปรข้อมูล หลัง Data cleansing

        0   studyName            333 non-null    object 
        1   Sample Number        333 non-null    int64  
        2   Species              333 non-null    object 
        3   Region               333 non-null    object 
        4   Island               333 non-null    object 
        5   Stage                333 non-null    object 
        6   Individual ID        333 non-null    object 
        7   Clutch Completion    333 non-null    object 
        8   Date Egg             333 non-null    object
        9   Culmen Length (mm)   333 non-null    float64
        10  Culmen Depth (mm)    333 non-null    float64
        11  Flipper Length (mm)  333 non-null    float64
        12  Body Mass (g)        333 non-null    float64
        13  Sex                  333 non-null    object 
        14  Delta 15 N (o/oo)    333 non-null    float64
        15  Delta 13 C (o/oo)    333 non-null    float64
    """)

    st.markdown("""
    ---
    ### Model's Workflow
    """)
    choice = st.radio("Select the option below to show Model's Workflow", ["SVM Model", "RF Model"])

    if choice == "SVM Model":
        svm_explaination()
    elif choice == "RF Model":
        rf_explaination()

def main():
    df = pd.read_csv("data/machine_learning/penguins_lter.csv")

    st.set_page_config(page_title="Machine learning Explanation", page_icon=":penguin:")

    st.title(":red[Machine Learning]")
    st.markdown("---")

    preparation(df)
    workflow(df)

    col1, col2, col3 = st.columns(3)
    col1.markdown('<a href="#model-s-workflow">Go to Model workflow</a>', unsafe_allow_html=1)

    if col3.button('Go to Model Page', use_container_width=1, type='primary'):
        st.switch_page("pages/2_MachineLearning-Model.py")

if __name__ == "__main__" :
    main()