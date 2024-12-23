import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('/Users/najlaalhomaid/Desktop/stre/best (2).pt')  # Replace with your model path

# Landmark information with updated details
landmark_info = {
    "King Abdullah Financial District -KAFD-": {
        "description": (
            "KAFD was inspired by King Abdullah’s vision to create a new financial district that "
            "will take the economy of Riyadh to new heights. KAFD provides the community with a vibrant "
            "experience in the heart of Riyadh, redefining Riyadh’s skyline with outstanding buildings inspired "
            "by the native landscape. **KAFD: where you Envision, Accelerate, Live.**"
        ),
        "resources": [{"text": "KAFD Official Site", "url": "https://www.kafd.sa/en/about-kafd"}]
    },
    "Al Faisaliah Tower": {
        "description": (
            "At the top of Al Faisaliah Tower, you can catch the most beautiful views of Riyadh from the restaurants "
            "that overlook it. Al Faisaliah Tower, the first skyscraper in Saudi Arabia, boasts a hotel and facilities "
            "equipped with the latest technologies, along with a children’s amusement park and international shopping stores."
        ),
        "resources": [{"text": "Visit Al Faisaliah Tower", "url": "https://www.visitsaudi.com/en/riyadh/attractions/al-faisaliah-tower-in-riyadh"}]
    },
    "Al Masmak Palace": {
        "description": (
            "The magnificent Al Masmak Palace, constructed in the hijri 14th century, served as a residence and a strong fort "
            "against enemies with its solid high walls. Today, it stands as a symbol of history and was transformed into a "
            "museum showcasing the legacy of the era."
        ),
        "resources": [{"text": "Learn About Al Masmak Palace", "url": "https://www.visitsaudi.com/en/riyadh/attractions/al-masmak-palace-in-riyadh"}]
    },
    "King Abdulaziz Center for World Culture (Ithra)": {
        "description": (
            "Ithra, meaning 'enrichment' in Arabic, is a cultural and creative destination for talent development and "
            "cross-cultural experiences. Since its opening in 2018, Ithra celebrates human potential and empowers creativity "
            "through inspiring workshops, performances, and events."
        ),
        "resources": [{"text": "Learn About Ithra", "url": "https://www.ithra.com/en/about-ithra"}]
    },
    "Kingdom Tower": {
        "description": (
            "The Kingdom Center’s mixed-use tower exhibits a unique design with a 200-foot-long observatory bridge spanning "
            "an inverted catenary arch. The striking triangular opening is visible from nearly all parts of Riyadh."
        ),
        "resources": [{"text": "Learn About Kingdom Center", "url": "https://omrania.com/project/kingdom-center/"}]
    },
    "Jabal AlFil -Elephant Rock-": {
        "description": (
            "A true icon of AlUla, the elephant-shaped rock formation stretches up 52 metres into the sky, shaped by wind "
            "and water erosion over millions of years. It is particularly magical at dusk when the fading sun casts a deep "
            "crimson light over its trunk and body."
        ),
        "resources": [{"text": "Learn About Elephant Rock", "url": "https://www.experiencealula.com/en/places-to-go/elephant-rock"}]
    },
    "Maraya": {
        "description": (
            "‘Maraya’, or mirror in Arabic, is a purpose-built event venue in AlUla covered entirely in mirrors that reflect "
            "the impressive natural landscape. Beyond its visual appeal, Maraya represents a modern wonder in a place of ancient history."
        ),
        "resources": [{"text": "Learn About Maraya", "url": "https://www.marayaalula.com/aboutmaraya"}]
    },
    "Hegra": {
        "description": (
            "Hegra, an ancient Nabataean city, boasts captivating tombs, wells, and stone-lined water channels showcasing "
            "ancient engineering and craftsmanship. Defensive walls, gates, and towers reveal Roman influence on this historic site."
        ),
        "resources": [{"text": "Learn About Hegra", "url": "https://www.experiencealula.com/en/places-to-go/hegra"}]
    },
    "Al Murabba Historical Palace": {
    "description": (
        "Al Murabba Historical Palace, established in 1939 by King Abdulaziz Al Saud, served as his official residence, the headquarters of state affairs, "
        "and a Diwan for meetings with global leaders. Located 2 km from old Riyadh, it played a pivotal role in shaping Saudi Arabia's political and economic landscape. "
        "This historic site witnessed the drafting of the Kingdom's first laws and the establishment of key institutions like the Ministry of Foreign Affairs."
    ),
    "resources": [
        {"text": "Visit Al Murabba Palace", "url": "https://www.visitsaudi.com/en/riyadh/attractions/al-murabba-palace"}
    ]
    },
    "Rijal Heritage Village": {
    "description": (
        "Rijal Heritage Village, located 45 km from Abha, is a stunning historical gem with 700 years of heritage. Famous for its vibrant stone buildings made from basalt and adorned with white quartz, the village features around sixty fortresses that narrate tales of military strength and social hospitality. Visitors can explore the village museum, showcasing artifacts across 20 sections that bring the history and traditions of Rijal Almaa to life."
    ),
    "resources": [
        {"text": "Learn More About Rijal Heritage Village", "url": "https://www.visitsaudi.com/en/aseer/attractions/rijal-almaa-of-aseer"}
    ]
    },
    "Faid Historic City": {
    "description": (
        "Faid Historic City was a vital stop along the ancient Zubaida Trail, an important pilgrimage route connecting Iraq to Makkah. "
        "Ranked by historians as the third most significant ancient city after Kufa and Basra, Faid boasts a rich cultural and historical heritage. "
        "Visitors can explore fascinating remnants of the old town, such as Al Kharash Castle, ancient wells like Al-Hamra, and intricate rock engravings on the surrounding mountains. "
        "The city was home to the Asad and Tayy tribes, which played pivotal roles in both pre-Islamic and Islamic eras."
    ),
    "resources": [
        {"text": "Learn More About Faid Historic City", "url": "https://www.visitsaudi.com/en/hail/attractions/historical-fayd-city-in-hail"}
    ]
    },
    "Tabuk Archaeological Castle": {
    "description": (
        "Dating back to 1559, Tabuk Archaeological Castle is a historical gem that once served as a vital station on the pilgrimage route from Damascus to Madinah. "
        "This two-story castle features beautiful architecture, archaeological inscriptions, and heritage exhibits narrating the history of the Tabuk region. Visitors can explore its open courtyard, mosques, towers, and Ain Al-Sukkar, a historic spring that once provided water to thousands of pilgrims. Restored multiple times, it remains a significant tourist and research destination today."
    ),
    "resources": [
        {"text": "Learn More About Tabuk Archaeological Castle", "url": "https://www.visitsaudi.com/en/tabuk/attractions/islamic-archaeological-tabuk-castle"}
    ]
    },
    "Al-Subaie Palace": {
    "description": (
        "Al-Subaie Palace is a stunning example of Andalusian-Islamic architecture and one of Saudi Arabia’s most significant historical palaces. "
        "Once a resting place for King Abdulaziz during his travels, the palace offers breathtaking views of the historic souks of the Old Town. "
        "With over 80 years of history, it reflects the decorative style of central Saudi Arabia and has undergone multiple restorations, including a full restoration in 2000. "
        "Set to become a heritage museum, the palace invites visitors to explore Saudi Arabia's rich cultural and architectural heritage."
    ),
    "resources": [
        {"text": "Learn More About Al-Subaie Palace", "url": "https://www.visitsaudi.com/en/riyadh/attractions/al-subaie-palace"}
    ]
    },
    "As-Safiyyah Museum and Park": {
    "description": (
        "As-Safiyyah Museum and Park is a unique cultural destination in Madinah. " # needs modification
        "Spanning 4,400 square meters, this site features a museum showcasing the Story of Creation, a serene public garden, waterways, shops, and restaurants. "
        "The project aims to enrich the cultural and spiritual experience of Hajj and Umrah pilgrims, aligning with Saudi Vision 2030 by promoting Madinah's heritage and enhancing religious tourism."
    ),
    "resources": [
        {"text": "Learn More About As-Safiyyah Museum and Park", "url": "https://welcomesaudi.com/news/as-safiyyah-museum-and-park-inaugurated-by-madinah-governor-prince-salman-bin-sultan"}
    ]
},
    "Diriyah": {
    "description": (
        "Diriyah, the birthplace of the First Saudi State, is a historic gem that tells the story of Saudi Arabia’s foundation. "
        "Located northwest of Riyadh, Diriyah’s mudbrick architecture and historic landmarks like At-Turaif district—a UNESCO World Heritage Site—showcase the rich culture, heritage, and resilience of the Saudi people. "
        "As part of Saudi Vision 2030, Diriyah is undergoing a transformation into a global cultural and lifestyle destination, offering visitors an authentic journey through its history while enjoying world-class dining, events, and experiences."
    ),
    "resources": [
        {"text": "Discover Diriyah's Story", "url": "https://www.diriyah.sa/en/our-story"}
    ]
}   
}


# Improved UI
st.title("🌆 Modern Tour: Your Guide to Saudi Arabia's Iconic Landmarks")
st.markdown(
    """
    Welcome to **Modern Tour**, your AI-powered travel companion! 🌟  
    Discover the beauty and history of Saudi Arabia with ease.  

    Upload a photo or snap one on the go, and let our intelligent system identify the landmark, 
    provide fascinating insights, and guide you to learn more about these remarkable destinations.  
    
    🗺️ **Your adventure begins here—let's explore together!** ✨
    """
)

st.divider()  # Add a horizontal line for better separation

# Layout: Option to upload or capture
st.markdown("### Choose an option to detect landmarks:")
col1, col2 = st.columns(2)

with col1:
    use_camera = st.checkbox("📷 Capture an image using your camera")

with col2:
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])

# Handle camera input
if use_camera:
    st.markdown("#### Live Camera Feed")
    st.write("Press 'Capture' to take a photo from your webcam.")

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 if Iriun Webcam is the default camera
    FRAME_WINDOW = st.image([])

    captured_frame = None

    # Place "Capture" button outside the loop
    capture_button = st.button("📸 Capture")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam. Ensure Iriun Webcam is running.")
            break

        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, caption="Live Feed", use_container_width=True)

        # Break the loop and capture the frame if the button is pressed
        if capture_button:
            captured_frame = frame
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        # Clear the live feed to avoid duplicate display
        FRAME_WINDOW.empty()

        # Display the captured image
        captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        st.image(captured_frame_rgb, caption="Captured Image", use_container_width=True)

        # Run detection on the captured image
        results = model.predict(source=captured_frame, save=False)

        # Annotate the image
        annotated_img = results[0].plot()

        # Convert image to RGB format for display
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Display the annotated image
        # st.image(annotated_img, caption="Detected Image", use_container_width=True)

        # Extract detected classes
        detected_classes = results[0].boxes.cls
        class_names = results[0].names

        if len(detected_classes) > 0:
            detected_class = class_names[int(detected_classes[0])]

            # Normalize the detected class (strip whitespace, convert to lowercase)
            normalized_class = detected_class.strip().lower()

            # Normalize dictionary keys
            normalized_landmark_info = {key.lower(): value for key, value in landmark_info.items()}

            st.subheader(f"📍 Detected Landmark: {detected_class}")

            # Check if the normalized class exists in the dictionary
            if normalized_class in normalized_landmark_info:
                info = normalized_landmark_info[normalized_class]
                st.write(f"{info['description']}")
                st.write("**Resources**:")
                for resource in landmark_info[detected_class]["resources"]:
                    st.markdown(f"- [{resource['text']}]({resource['url']})")
            else:
                st.write("No information available for this landmark.")
        else:
            st.write("No landmarks detected in the image.")

# Handle uploaded file
if uploaded_file is not None:
    st.markdown("#### Uploaded Image")
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Run detection
    results = model.predict(source=img, save=False)

    # Annotate the image
    annotated_img = results[0].plot()

    # Convert image to RGB format for display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    # st.image(annotated_img, caption="Detected Image", use_container_width=True)

    # Extract detected classes
    detected_classes = results[0].boxes.cls
    class_names = results[0].names

    if len(detected_classes) > 0:
        detected_class = class_names[int(detected_classes[0])]

        # Normalize the detected class (strip whitespace, convert to lowercase)
        normalized_class = detected_class.strip().lower()

        # Normalize dictionary keys
        normalized_landmark_info = {key.lower(): value for key, value in landmark_info.items()}

        st.subheader(f"📍 Detected Landmark: {detected_class}")

        # Check if the normalized class exists in the dictionary
        if normalized_class in normalized_landmark_info:
            info = normalized_landmark_info[normalized_class]
            st.write(f"**Description**: {info['description']}")
            st.write("**Resources**:")
            for resource in landmark_info[detected_class]["resources"]:
                    st.markdown(f"- [{resource['text']}]({resource['url']})")
        else:
            st.write("No information available for this landmark.")
    else:
        st.write("No landmarks detected in the image.")