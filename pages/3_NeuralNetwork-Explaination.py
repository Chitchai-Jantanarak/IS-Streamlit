import streamlit as st

def preparation() -> None :
    st.markdown("<a id='dataset-preparation'></a>", unsafe_allow_html=True)
    st.markdown("""
    ## Dataset Preparation

    การเตรียมข้อมูล เป็นข้อมูลที่หยิบมาใช้จาก   "[kaggle](https://www.kaggle.com/)" <br />
    [<span style="color:blue">การวินิจฉัยโรคปอดอักเสบจากภาพ X-ray (Chest X-ray Images diagnose PNEUMONIA)</span>](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    """, unsafe_allow_html=1)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    เพื่อมาทำโมเดลให้มาวิเคราะห์รูปภาพจากรูปภาพมีความผิดปกติภาพ X-ray จะสามารถวิเคราะห์ได้จาก :
    - ความทึบแสง, ความฟุ้งของแสง
    - ความหนาแน่นที่บอกถึงการติดเชื้อ
    - ลักษณะของปอด ที่มีการบวม
    - ปริมาณ และจำนวนของหลอมลม
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ซึ่งตัวข้อมูลที่ได้นั้น ใช้เป็น Images ทั้งหมดในการเทรน ประกอบไปด้วยไฟล์ 
    โดยทั้ง 3 ที่จัดเตรียมไว้ได้แยกส่วนเป็นแบบ "ปกติ-ปอดอักเสบ(ไม่แยกชนิดของปอดอักเสบ)" ไว้แล้ว

        - test
            test/NORMAL
            test/PNEUMONIA
        - train 
            train/NORMAL
            train/PNEUMONIA
        - val
            val/NORMAL
            val/PNEUMONIA
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.image('public/neural_network/dataprep_images.jpg')

    st.markdown("""
    เนื่องจากตัวข้อมูลมีความสมบูรณ์ และ แยกชนิดให้ไว้แล้วเพื่อไม่ให้ง่ายเกินไป 
    จึงนำไฟล์ต่างๆ ในแต่ละที่ใน directory มา random เป็นอีกข้อมูลเพื่อป้องกันการเทรน samples เก่าๆ มากเกินไป

    ```python
    # EXAMPLE !
    # นำช้อมูลจากส่วนอื่นมาเพิ่มไว้ e.g. ∀(train) + 0.05(test)
    # remark: อาจจะเป็นวิธีที่ไม่ดีสำหรับการ testing แต่ทำไว้เพื่อให้เทรนได้ไม่ซ้ำกันเกินไป :)          

    train_x = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32,
    )

    # 0.05 of test
    test_x_sample = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.05, 
        subset='validation', 
    )
                
    for image, label in test_x_sample:
        train_x = train_x.concatenate(
            tf.data.Dataset.from_tensor_slices((image, label))
        )
                
    # dataset = [[image, label], [...], ...]
    ```
    """)

    st.markdown("---")

def workflow() -> None :
    st.markdown("<a id='workflow'></a>", unsafe_allow_html=True)
    st.markdown("""
    ## WORKFLOW
                
    ##### Model ที่ใช้ : [Convolutional neural network (CNN)](https://www.tensorflow.org/tutorials/images/cnn?hl=en)
    
    ###### เหตุผลที่ใช้ CNN model
                
    Convolutional Neural Network (CNN) เป็นโมเดล Deep Learning 
    ที่เน้นใช้ในการจดจำรูปภาพ ทำให้เหมาะกับการแยกภาพในโมเดลนี้ โดยทั้งหมดนั้นเป็นข้อมูลดิบทั้งหมด และ
    จัดเรียงไว้ให้แล้ว เพื่อให้ตัวโมเดลดึงจุดสำคํญต่าง ๆ ได้ ผ่าน Activation Function ในแต่ละ layer
    """,)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### ขั้นตอนการ training ในนี้จะใช้เป็น CNN + [VGG16 model](https://keras.io/api/applications/vgg/) (CNN classify images)
    ##### Activation Function
    - (ReLU : Convolutional)
    - (Sigmoid : Fully connected)
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### 1. ขั้นตอนการอ่าน, เรียกใช้ข้อมูล และ การทำ samples
                
    ```python
    # __builder_model.py

    base_path = '.....' # Data PATH   
                
    def load_image(base_path) -> tuple:
          
        # Load images from the specified directory
        #
        # Args:
        #     base_path (str): Base path to the image directory 
        #     [INNER : {PreData : test, train, val}]
        #
        # Returns:
        #     Tuple of (train_dataset, validation_dataset)
          
        
        data_dir = pathlib.Path(base_path)

        if not data_dir.exists():
            raise ValueError(f"Director does not exist: {data_dir}")
        
        train_x = tf.keras.utils.image_dataset_from_directory(
            str(data_dir / 'train'),
            color_mode='rgb',  
            seed=123,
            image_size=(224, 224),  
            batch_size=32
        )

        test_x = tf.keras.utils.image_dataset_from_directory(
            str(data_dir / 'test'),
            color_mode='rgb',  
            seed=123,
            image_size=(224, 224),  
            batch_size=32
        )
        
        val_x = tf.keras.utils.image_dataset_from_directory(
            str(data_dir / 'val'),
            color_mode='rgb',  
            seed=123,
            image_size=(224, 224),  
            batch_size=32
        )
        return (train_x, test_x, val_x)
    ```
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### 2. Preprocess ก่อนเทรนโมเดล Augmentation Filtering, Normalization
                
    ทำเพื่อให้อยู่ในสถานะพร้อมเทรน
    - Augment : ทำให้รูปภาพอยู่ในหลายรูปแบบ (บิดภาพ, ขยับ, หมุน, ซูม) แบบสุ่ม
    - Normalization : ให้ภาพ RGB Code ใน scale [0, 1]
                
    ```python
    # __builder_model.py

    def preprocess_image(train_x, test_x, val_x) -> tuple:
       
        # Preprocess rescaling an image bit in range [0, 1] 
        #
        # Args:
        #     train_x (Array)
        #     test_x (Array)
        #     val_x (Array)
        #
        # Returns:
        #     Tuple of (train_dataset, validation_dataset)

        # Augment Filtering
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        # Normalize datasets
        train_x_norm = train_x.map(normalize)
        test_x_norm = test_x.map(normalize)
        val_x_norm = val_x.map(normalize)

        return (train_x_norm, test_x_norm, val_x_norm)
    ```
    """)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### 3. Building Model & Training
                
    - ทำโมเดล หรือ Layers ในการเทรน โดยอิงตามตัว 
    VGG16 model จึงมีการทำ Flatten ระหว่างช่วงแรกหลังจากผ่าน VGG16 
    และนำมาเข้า Activation function อีกรอบ โดย FREEZING VGG16 ไม่ให้เทรนตามโมเดลไว้ด้วย
    เพื่อให้ตัว base ของโมเดล (ตัวอ้างอิง) ไม่ขยับระหว่างเทรน    
    - หลังจาก Building จะใช้ตัว Complie เป็น Adam ที่ init learning rate = 1e-3 และตั้งเป็น binary 
    เพื่อจำแนกแค่ เป็นปกติ หรือ ปอดอักเสบ
    - ระหว่างการเทรน จะใช้ early_stopping + checkpoint เพื่อให้สามารถกำหนด learning rate ให้ลดลง
    และหยุดการเทรนไว้หากโมเดลเทรนแล้ว overfit จนเกินไป
                
    ```python
    # __builder_model.py

    def modelsConstruct() -> tf.keras.models.Sequential:
                
        # Model builder using VGG16 as base model

        base_model = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3)  # Match image size and RGB channels
        )

        model = Sequential([
            base_model,
            Flatten(),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        base_model.trainable = False

        return model

    def modelsBuilt(model):
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', Recall(), Precision()]
        )

        model.summary()
        return model

    def training(train_data, val_data, model):
        # Add early stopping and model checkpoint
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'pneumonia_model.h5', 
            monitor='val_accuracy', 
            save_best_only=True
        )

        model.fit(
            train_data,
            epochs=10,
            validation_data=val_data,
            callbacks=[early_stopping, model_checkpoint]
        )
    ```
    """)
    st.markdown("""
        ##### Layer ต่างๆในโมเดลโดยรวม
    """)

    col1, col2 = st.columns([1, 2])
    with col1: 
        st.image("public/neural_network/workflow_model.png", use_container_width=1)
    with col2: 
        st.image("public/neural_network/workflow_model2.jpg", use_container_width=1)


    st.markdown("""
    ###### สามารถดูลำดับขั้นตอนการทำทั้งหมดได้ [ที่นี่](https://github.com/Chitchai-Jantanarak/IS-Streamlit/blob/main/train/neural_network/__builder_model.py)
    ---
    """)

def conclusion() -> None :
    st.markdown("<a id='conclusion'></a>", unsafe_allow_html=True)
    st.markdown("""
    ## Conclusion
                
    จากการเทรนมีผลสรุปการเทรน ใช้เวลาเทรนโมเดลที่ดีที่สุด **5 hr 48 mins** มีผลสรุปเป็น
    - training flow
    - sample prediction images
    - confusion matrix
    - ROC curve
                
    ##### ผลสรุปที่ได้
    ค่อนข้างมีความใกล้กับการวัดผลเป็น Pneumonia มากจนเกินไป จากการ test ที่มีการแยกภาพที่
    เป็นปอดอักเสบค่อนข้างมากจึงทำให้มีสิทธิ์ทำให้รูปภาพที่ได้รับจะถูกวัดผลเป็นปอดอักเสบมาก **TYPE I ERROR**
                
    ###### ส่วนผลสรุปนี้วิธีทำต่าง ๆ สามารถดูได้ [ที่นี่](https://github.com/Chitchai-Jantanarak/IS-Streamlit/tree/main/train/neural_network)
                
    ปล. ทำไฟล์โมเดลหายบ่อย + ปรับจูนเยอะเลยใช้เวลาครับ ;-;
    """,)

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### training flow
    """)
    st.image("public/neural_network/conclusion_training_flows.png")

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### sample prediction images (รูปภาพจาก Test file)
    """)
    st.image("public/neural_network/conclusion_sample_predictions.png")

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### confusion matrix
    """)
    st.image("public/neural_network/conclusion_confusion_matrix.png")

    st.markdown(""" <br /> """, unsafe_allow_html=1)
    st.markdown("""
    ##### ROC
    """)
    st.image("public/neural_network/conclusion_roc_curve.png")

def main():
    st.set_page_config(page_title="Neural Networks Explanation", page_icon="🫁")

    st.title(":red[Neural Networks]")
    st.markdown("---")

    preparation()
    workflow()
    conclusion()

    # Footer or additional content
    col1, col2, col3 = st.columns(3)
    if col3.button('Go to Model Page', use_container_width=1, type='primary'):
        st.switch_page("pages/4_NeuralNetwork-Model.py")

if __name__ == "__main__" :
    main()