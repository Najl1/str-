import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('bestt.pt')  # Replace with your model path

# Landmark information with updated details
landmark_info = {

    "Tabuk Fortess": {
        "English_title": (
            "Tabuk Archaeological Castle"
        ),
        "English_description": (
      "Tabuk Archaeological Castle, built in 1559, stands as a testament to Saudi Arabia's rich heritage. Once a vital station on the pilgrimage route from Damascus to Madinah, this two-story fortress is adorned with Islamic architectural details, historical inscriptions, and a sense of timeless beauty. Visitors can explore its open courtyard, ancient mosques, and the famous Ain Al-Sukkar spring, which provided water to countless pilgrims. Today, the castle is a gateway to Tabuk's storied past, offering a glimpse into the lives of travelers who passed through centuries ago.",

        ),
         "Arabic_title": (
            "قلعة تبوك الاثرية"
        ),
        "Arabic_description": (   
               "قلعة تبوك الأثرية، التي بُنيت عام 1559، تُعد رمزًا لتراث المملكة العربية السعودية الغني. كانت القلعة محطة حيوية على طريق الحجاج من دمشق إلى المدينة المنورة، وتتميز بتفاصيلها المعمارية الإسلامية والنقوش التاريخية التي تضفي عليها جمالًا خالدًا. يمكن للزوار استكشاف فناء القلعة المفتوح، والمساجد القديمة، وعين السكر الشهيرة التي وفرت الماء لآلاف الحجاج. اليوم، تُعد القلعة بوابة إلى ماضي تبوك العريق، وتمنح الزائرين لمحة عن حياة المسافرين الذين مروا بها قبل قرون."
        ),
        "resources": [
        {"text": "Learn More About Tabuk Archaeological Castle", "url": "https://www.visitsaudi.com/en/tabuk/attractions/islamic-archaeological-tabuk-castle"}
            ]
    },
     "Al-Subaie Palace": {
        "English_title": (
            "Al-Subaie Palace"
        ),
        "English_description": (
      "Al-Subaie Palace is a majestic historical landmark in the heart of Riyadh, embodying the rich heritage and architectural brilliance of the Najd region. Constructed over 80 years ago, the palace served as a significant resting place for King Abdulaziz Al Saud during his travels. It played a pivotal role in hosting royal gatherings and state affairs, making it a vital part of Saudi history. The architectural style of the palace is a harmonious blend of traditional Saudi craftsmanship and Andalusian design elements, showcasing intricate patterns, spacious courtyards, and a reflection of the opulent lifestyle of that era. The palace has undergone several restoration projects to maintain its grandeur and historical significance, ensuring that future generations can connect with the nation’s cultural roots. Today, Al-Subaie Palace stands as a cultural treasure, welcoming visitors to explore its timeless beauty, immerse themselves in the stories of the past, and appreciate the unique blend of history and art that it represents.",

        ),
         "Arabic_title": (
            "قصر السبيعي"
        ),
        "Arabic_description": (   
      "يُعد قصر السبيعي معلمًا تاريخيًا شامخًا في قلب مدينة الرياض، ويجسد التراث الغني والبراعة المعمارية لمنطقة نجد. بُني القصر قبل أكثر من 80 عامًا، وكان بمثابة محطة استراحة مهمة للملك عبد العزيز آل سعود خلال رحلاته. لعب القصر دورًا حيويًا في استضافة التجمعات الملكية والشؤون الحكومية، مما جعله جزءًا أساسيًا من تاريخ المملكة. يتميز القصر بأسلوب معماري يجمع بين الحرفية السعودية التقليدية والعناصر التصميمية الأندلسية، مما يظهر في النقوش الدقيقة والساحات الفسيحة والانعكاس للحياة الفاخرة في ذلك العصر. خضع القصر لعدة مشاريع ترميم للحفاظ على رونقه وأهميته التاريخية، لضمان بقاء هذا الإرث للأجيال القادمة. اليوم، يقف قصر السبيعي ككنز ثقافي، مرحبًا بالزوار لاستكشاف جماله الخالد والانغماس في قصص الماضي وتقدير المزج الفريد بين التاريخ والفن الذي يمثله."
        ),
        "resources": [
        {"text": "Learn More About Al-Subaie Palace", "url": "https://www.visitsaudi.com/en/riyadh/attractions/al-subaie-palace"}
            ]
    },
     "As-Safiyyah Museum and Park": {
        "English_title": (
            "As-Safiyyah Museum and Park"
        ),
        "English_description": (
        "Nestled in the sacred city of Madinah, As-Safiyyah Museum and Park is a harmonious blend of natural beauty and cultural enrichment. Spanning an impressive 4,400 square meters, this serene destination offers a haven for pilgrims, visitors, and locals alike. At its heart lies the Story of Creation Museum, which takes guests on a mesmerizing journey through time, exploring the origins of life and the wonders of the universe. Surrounding the museum, lush gardens, tranquil waterways, and shaded pathways invite visitors to reflect and unwind in a peaceful ambiance. Designed to align with Saudi Vision 2030, the park also features shops, dining areas, and cultural exhibits that highlight Madinah's rich heritage. As-Safiyyah Museum and Park is not just a location—it is an experience that nurtures the soul, ignites curiosity, and fosters a deeper connection to the history and spirituality of this cherished city.",
        ),
         "Arabic_title": (
         "متحف وحديقة الصفيَّة"
        ),
        "Arabic_description": (   
        "في قلب المدينة المنورة المقدسة، يقع متحف وحديقة الصفيَّة كمزيج متناغم من الجمال الطبيعي والتنوير الثقافي. يمتد هذا الوجهة الساحرة على مساحة 4400 متر مربع، مما يوفر ملاذًا هادئًا للحجاج والزوار والمقيمين. يحتضن المتحف في وسطه متحف قصة الخلق، الذي يأخذ الضيوف في رحلة ساحرة عبر الزمن لاستكشاف أصول الحياة وعجائب الكون. تحيط بالمتحف حدائق غنّاء وممرات مظللة ومجاري مياه هادئة تدعو الزوار إلى التأمل والاسترخاء في أجواء هادئة. وبتصميم ينسجم مع رؤية السعودية 2030، يحتوي المكان أيضًا على متاجر وأماكن لتناول الطعام ومعارض ثقافية تُبرز التراث الغني للمدينة المنورة. متحف وحديقة الصفيَّة ليس مجرد موقع، بل هو تجربة تغذي الروح وتشعل الفضول وتعزز الاتصال العميق بتاريخ وروحانية هذه المدينة المباركة."
        ),
        "resources": [
        {"text": "Learn More About As-Safiyyah Museum and Park", "url": "https://welcomesaudi.com/news/as-safiyyah-museum-and-park-inaugurated-by-madinah-governor-prince-salman-bin-sultan"}
            ]
    },
      "Diriyah": {
        "English_title": (
            "Diriyah"
        ),
        "English_description": (
            "Diriyah, often referred to as the “Jewel of the Kingdom,” is a place where history and heritage come alive. Nestled in the heart of Saudi Arabia, it is celebrated as the birthplace of the First Saudi State and stands as a symbol of unity, resilience, and progress. Diriyah’s iconic At-Turaif District, a UNESCO World Heritage Site, showcases mudbrick architecture that tells stories of a bygone era. This historic city, once the seat of governance and culture, is now reimagined as a global hub for cultural and lifestyle experiences."

"As visitors stroll through Diriyah’s winding lanes, they are transported back in time while also experiencing the vibrancy of the present. The ongoing transformation under Saudi Vision 2030 has brought world-class dining, art, and events to this historic location, creating a seamless blend of tradition and modernity. Diriyah is not just a destination—it is a journey through the soul of Saudi Arabia, celebrating its past while envisioning its bright future."
            
            ),
         "Arabic_title": (
         "الدرعية"
        ),
        "Arabic_description": (   
            " الدرعية، التي تُعرف غالبًا باسم **جوهرة المملكة**، هي مكان ينبض بالتاريخ والإرث. تقع في قلب المملكة العربية السعودية، وتُحتفى بها كمهد الدولة السعودية الأولى ورمز للوحدة والصمود والتقدم. يعكس حي الطريف الشهير، وهو أحد مواقع التراث العالمي لليونسكو، جمال الهندسة المعمارية الطينية التي تحكي قصصًا من الماضي العريق. كانت هذه المدينة التاريخية يومًا مركزًا للحكم والثقافة، واليوم تُعاد رؤيتها كمركز عالمي للتجارب الثقافية والحياتية. بينما يتجول الزوار في أزقة الدرعية المتعرجة، يشعرون بأنهم قد عادوا إلى الزمن الماضي، ولكن مع تجربة حيوية للحاضر. تحت مظلة رؤية السعودية 2030، أُضيفت إلى هذا الموقع التاريخي تجارب طعام عالمية وفنون وفعاليات، مما خلق توازنًا رائعًا بين التراث والحداثة. الدرعية ليست مجرد وجهة؛ إنها رحلة إلى روح المملكة العربية السعودية، تحتفي بماضيها وتبشر بمستقبلها المشرق."        ),
        "resources": [
        {"text": "Discover how the Kingdom's jewel was shaped?", "url": "https://www.dgda.gov.sa/ar/the-diriyah-story"}
            ]
    },
        "King Abdullah Financial District -KAFD-": {
        "English_title": (
            "KAFD - King Abdullah Financial District"
        ),
        "English_description": (
  "KAFD was inspired by King Abdullah’s vision to create a new financial district that "
            "will take the economy of Riyadh to new heights. KAFD provides the community with a vibrant "
            "experience in the heart of Riyadh, redefining Riyadh’s skyline with outstanding buildings inspired "
            "by the native landscape. **KAFD: where you Envision, Accelerate, Live.**"        ),
         "Arabic_title": (
         "مركز الملك عبدالله المالي - كافد"
        ),
        "Arabic_description": (   
 "يستوحي مركز الملك عبدالله المالي كافد مفهومه من رؤية الملك عبدالله بن عبدالعزيز - رحمه الله - والمتمثلة بإنشاء مركز مالي جديد يرتقي باقتصاد مدينة الرياض إلى مستويات جديدة."
            " وشهد كافد، بعد الاستحواذ عليه من قبل صندوق الاستثمارات العامة في المملكة العربية السعودية، تطورًا نوعيًا ليصبح من أبرز الوجهات الرائدة للأعمال وأساليب الحياة، ويجسّد بذلك القيم الأساسية لرؤية المملكة العربية السعودية 2030."
            " ولن يقتصر دور كافد على المساهمة الفاعلة في جهود النمو والتنوع الاقتصادي، وإنما سيقدّم أيضًا مجتمعًا نابضًا بالحياة في قلب الرياض، ليمثّل بذلك مدينةً داخل مدينة تضم العديد من المباني الاستثنائية المستوحاة من مزايا الطبيعية المحلية، والتي تعيد صياغة الأفق العمراني للرياض."
            " كما يدعم كافد الشركات من خلال تزويدها ببنية تحتية مكتبية متقدمة وحلول ذكية للمدن المستدامة، فضلًا عن تقديم تجارب فريدة لأنماط الحياة العصرية ضمن مجموعة من أفضل خيارات الترفيه والتجزئة. كافد: رؤية، ازدهار، حياة "
              ),
        "resources": [{"text": "KAFD Official Site", "url": "https://www.kafd.sa/en/about-kafd"}]
    },
        
    "Al Faisaliah Tower": {
        "English_title": (
            "Al Faisaliah Tower"
        ),
        "English_description": (
   "Al-Faisaliah Tower is the first skyscraper in the Kingdom of Saudi Arabia, standing at a height of 267 m."  
                        "At the time of its construction, it was taller than any European building."  
                        "Located in al-Olaya district of Riyadh, its construction began in 1997."  
                        "The building project was inaugurated by the Custodian of the Two Holy Mosques, King Salman bin Abdulaziz Al Saud when he was the Prince of the Riyadh region."  
                        "The tower, which weighs 10,500 t, was officially opened in 2000."  
                        "Al-Faisaliyah Tower includes 30 floors of corporate offices, restaurants, service facilities, event and conference centers, a five-star hotel spanning eight floors, and four floors of shops with 100 showrooms."  
                      ),
         "Arabic_title": (
         "برج الفيصلية"
        ),
        "Arabic_description": (   
  "يعد برج الفيصلية أول ناطحة سحاب في المملكة العربية السعودية، ويبلغ ارتفاعه 267 مترًا."  
                        "في وقت بنائه، كان أطول من أي مبنى أوروبي."  
                        "يقع في حي العليا بمدينة الرياض، وبدأت أعمال بنائه في عام 1997."  
                        "تم تدشين المشروع من قبل خادم الحرمين الشريفين الملك سلمان بن عبدالعزيز آل سعود عندما كان أمير منطقة الرياض."  
                        "تم افتتاح البرج رسميًا في عام 2000، ويبلغ وزنه 10,500 طن."  

                        "يشمل برج الفيصلية 30 طابقًا من المكاتب التجارية، والمطاعم، والمرافق الخدمية، ومراكز الفعاليات والمؤتمرات، وفندقًا خمس نجوم يمتد على ثمانية طوابق، وأربعة طوابق من المحلات التجارية تحتوي على 100 معرض."  
 ),
        "resources": [{"text": "Visit Al Faisaliah Tower", "url": "https://saudipedia.com/en/article/1622/culture/al-faisaliah-tower"}]
    },

 "Al Masmak Palace": {
        "English_title": (
            "Al Masmak Palace"
        ),
        "English_description": (
   "Al-Masmak Palace, built in the 14th century AH, served as a residence and a strong fortress against enemies with its solid, high walls."  
            "Today, it stands as a symbol of history and has been transformed into a museum showcasing the legacy of that era."  
            "The historic palace includes vivid depictions of the story of unifying the Kingdom, including traces of the Battle of Riyadh represented by the mark of 'Ibn Jalawi's' spear on the gate of Al-Masmak Fortress."  
            "King Abdulaziz led his men in the battle in 1902, among them was Prince Fahd bin Jalawi."  
            "He threw his spear intending to hit Ajlan bin Rashid, but it lodged into the gate, leaving a crack that remains a testament to the heroism of one of the founding king’s most important allies."  
                ),
         "Arabic_title": (
         "قصر المصمك"
        ),
        "Arabic_description": (   
   "قصر المصمك، الذي تم بناؤه في القرن الرابع عشر الهجري، مقرًا للإقامة وحصنًا قويًا ضد الأعداء بجدرانه العالية المتينة. "
            "اليوم، يقف كرمز للتاريخ وتم تحويله إلى متحف يعرض إرث ذلك العصر."
            "يضم القصر الأثري صور حية لقصة توحيد المملكة، بما في ذلك آثار معركة استرداد الرياض المتمثلة في أثر رمح 'ابن جلوي' على باب حصن المصمك."  
            "إذ قاد الملك عبدالعزيز رجاله في المعركة عام 1902م وكان بينهم الأمير فهد بن جلوي."  
            "وقد رمى رمحه بقصد إصابة عجلان بن رشيد لكنه استقر في الباب وأحدث شرخاً بقي أثره شاهداً على قصة بطولة أحد أهم أعوان الملك المؤسس."  
 ),
        "resources": [{"text": "Learn About Al Masmak Palace", "url": "https://www.visitsaudi.com/en/riyadh/attractions/al-masmak-palace-in-riyadh"}]
    },
 
 "Ithra": {
        "English_title": (
            "King Abdulaziz Center for World Culture -Ithra-"
        ),
        "English_description": (
       "Drawing inspiration from the vision of the Kingdom’s founder, Saudi Aramco has always been committed to making valuable contributions to the growth and prosperity of the nation, its communities, and its citizens."  
                        "Our Story"  
                        "From the time of the Concession Agreement that laid the foundation for Saudi Aramco back in 1933, the late King Abdulaziz bin Abdul Rahman Al Sa‘ud sought to build an industrial and economic model that would meet the needs of the young Kingdom, and secure the nation’s prosperity, well-being, and development for future generations."  
                        "When Well No. 7 — which would later be known as the “Prosperity Well” — struck oil in 1938 and began producing commercial quantities, it fulfilled the ambition of the Concession Agreement and set in motion the potential for the Kingdom’s prosperity."  
                        "Throughout its existence, Saudi Aramco has been — besides one of the world’s pre-eminent energy companies — a provider of more than just crude oil and natural gas."  
                        "In its early days, the company provided training and jobs, helped build the Kingdom’s infrastructure, and constructed schools and hospitals, among a wide variety of citizenship activities."  
                        "On May 20, 2008, Saudi Aramco marked the company’s 75th anniversary with the laying of the cornerstone of its citizenship endeavors to accelerate human potential: The King Abdulaziz Center for World Culture."  
                        "Known as Ithra, the Arabic word for “enrichment,” the project’s vision was to build a destination that would ignite cultural curiosity, stimulate the exploration of knowledge, and inspire creativity through the power of ideas, imagination and innovation."  
                        "Seeking a unique and inspirational design, the company conducted a competition among leading architectural firms, which was won by Snøhetta, a Norwegian company known for the design of many famous buildings around the world including Bibliotheca Alexandrina, the iconic library in Alexandria, Egypt."  
                        "Snøhetta’s entry featured a simple but powerful arrangement of 'stones' that celebrated the interaction between rocks — as the source of oil that fuels the Kingdom — and the source of energy of a different kind, the power of imagination and creativity."  
                        "In a symbolic ceremony on that day in May, the late King Abdullah bin Abdulaziz Al Sa‘ud placed a symbolic cornerstone for the project not far from the Prosperity Well."  
                        "After its design was completed in early 2010, construction began later that year in August."  
                        "Determination and perseverance kept the dream alive as the tower slowly emerged above the desert landscape."
                          ),
         "Arabic_title": (
         "مركز الملك عبدالعزيز الثقافي العالمي - إثراء"
        ),
        "Arabic_description": (   
    "يحكي مبنى إثراء الفريد قصًة عميقة الجذور تبدأ من الفكرة وحتى التنفيذ."  
                        "صُمم المركز بواسطة شركة »سنوهيتا« النرويجية على هيئة مجموعة من الصخور التي تمثّل الوحدة، حيث تمثل هذه الصخور تكاملاً معماريًا يحتضن صرحًا مهمًا من صروح المعرفة والثقافة والفن والمجتمع والإبداع."  
                        "فالمبنى يبدو كمجموعة باهرة من صخورٍ هائلة الحجم، تكمن رمزيتها في العامل الزمني للتصميم الداخلي للمبنى."  
                        "فالأدوار الواقعة تحت مستوى سطح الأرض كالأرشيف والمتحف ترمز إلى أصالة الماضي."  
                        "وعند مستوى السطح نرى نهضة الحاضر عبر العروض الحية."  
                        "أما برج إثراء والمكتبة ومختبر الأفكار، يرمزون إلى المستقبل الواعد."  
                        "إن تصميم إثراء الفريد هو دليل على البراعة والريادة الهندسية."  
                        "تشكل الأنابيب الفولاذية المصنوعة بعناية والمطوية بشكلٍ فردي طبقةً فولاذية لامعة تغطي السطح الخارجي للمبنى والعديد من جدرانه الداخلية."  
                        "يتدفق الهواء حول الأنابيب؛ مما يشكل عازلًا عن المناخ الصحراوي المحيط بالمركز في الوقت الذي تبقي فيه الأنابيب ذاتها المبنى مظللًا."  
                        "كما يقع المركز على مساحة بيضاوية خضراء بين الطرق السريعة والصحراء، والتي جعلته بارزًا على بعد أميال من كافة الاتجاهات."  
                        "ويتميز المبنى باستخدامه لتقنيات البناء القديمة، ومن أهمها التربة المدكوكة."  
                        "تشكل التربة المدكوكة الجزء الداخلي من المبنى، لتساهم في عزل الصوت، باستخدام مواد طبيعية مثل الرمل والحصى والطين، والتي تم جمعها من مختلف أراضي المملكة."  
                        "ويتضمن المركز مكتبة تحتوي على 4 طوابق، وبرج إثراء الذي يضم 18 طابقًا، ومختبر الأفكار ذو 3 طوابق."  
                        "كما يضم معرضًا للطاقة، ومتحفًا يحتوي على 5 معارض، وسينما تتسع لأكثر من 300 شخص، ومسرحًا للفنون يتسع لـ900 شخص."  
                        "تتضمن المرافق أيضًا القاعة الكبرى التي تصل مساحتها إلى 1500 متر مربع، إلى جانب متحف الطفل والمسجد."  
                        "أما على الصعيد البيئي فقد بني المركز وفق مقاييس »LEED »الدولية (الريادة في التصميم البيئي والطاقة)."  
                        "حصل المركز على شهادة »LEED »الذهبية؛ لالتزام عناصر المبنى وفق المعايير الدولية والمقاييس المعتمدة بالدرجة الذهبية."    
            
       ),
        "resources": [{"text": "Learn About Ithra", "url": "https://www.ithra.com/application/files/4316/9140/9117/The_Ithra_Story_English.pdf"}]
    },
  "Kingdom Tower": {
        "English_title": (
            "Kingdom Tower"
        ),
        "English_description": (
  
                        "Kingdom Tower, is a 41-story, 302.3 m (992 ft) skyscraper in the al-Olaya district of Riyadh, Saudi Arabia."  
                        "When completed in 2002, it overtook the 267-meter (876 ft) Faisaliah Tower as the tallest tower in Saudi Arabia."  
                        "It has since been surpassed and, as of 2021, is the fifth-tallest skyscraper in the country, whose tallest two buildings are The Clock Towers and the Capital Market Authority Tower."  
                        "It is the world's third-tallest building with a hole after the Shanghai World Financial Center and the 85 Sky Tower in Taiwan."  
                        "It contains the King Abdullah Mosque, which is the world's most elevated mosque from ground level."  
                        "The tower was developed by Prince Al-Waleed bin Talal, and designed by the team of Ellerbe Becket and Omrania, who were selected through an international design competition."  
                        "It is situated on a 100,000–square-metre site and houses the 57,000-square-meter Al-Mamlaka shopping mall, offices, the Four Seasons Hotel Riyadh, and luxury apartments."  
                        "There is a 65m skybridge atop the skyscraper."  
                        "The upper third of the tower features an inverted parabolic arch topped by a public sky bridge."  
                        "The sky bridge is a 300-ton steel structure, taking the form of an enclosed corridor with windows on both sides."  
                        "After paying the admission fees, visitors take two elevators to reach that level."
            ),
         "Arabic_title": (
         "برج المملكة"
        ),
        "Arabic_description": (   
 
                        "برج المملكة، هو ناطحة سحاب مكونة من 41 طابقًا بارتفاع 302.3 متر (992 قدمًا) في حي العليا بالرياض، المملكة العربية السعودية."  
                        "عند اكتماله في عام 2002، تجاوز برج الفيصلية الذي يبلغ ارتفاعه 267 مترًا (876 قدمًا) كأطول برج في السعودية."  
                        "واعتبارًا من عام 2021، يعد خامس أطول ناطحة سحاب في البلاد، حيث يعد كل من أبراج الساعة وبرج هيئة السوق المالية الأطول."  
                        "إنه ثالث أطول مبنى في العالم يحتوي على فتحة بعد مركز شنغهاي المالي العالمي وبرج السماء 85 في تايوان."  
                        "يحتوي على مسجد الملك عبدالله، وهو أعلى مسجد في العالم من حيث الارتفاع عن سطح الأرض."  
                        "تم تطوير البرج من قبل الأمير الوليد بن طلال، وتم تصميمه من قبل فريق إليربي بيكيت وعمرانية، اللذين تم اختيارهما من خلال مسابقة تصميم دولية."  
                        "يقع البرج على مساحة تبلغ 100,000 متر مربع ويضم مركز تسوق المملكة الذي تبلغ مساحته 57,000 متر مربع، ومكاتب، وفندق فورسيزونز الرياض، وشقق فاخرة."  
                        "يحتوي البرج على جسر سماء يبلغ طوله 65 مترًا فوق قمته."  
                        "يتميز الثلث العلوي من البرج بقوس شبه مكافئ مقلوب يتوج بجسر سماء عام."  
                        "جسر السماء عبارة عن هيكل فولاذي يزن 300 طن، يتخذ شكل ممر مغلق به نوافذ على كلا الجانبين."  
                        "بعد دفع رسوم الدخول، يأخذ الزوار مصعدين للوصول إلى هذا المستوى."  
      ),
        "resources": [{"text": "Learn About Kingdom Center", "url": "https://en.wikipedia.org/wiki/Kingdom_Centre"}]
    },

"Jabal AlFil -Elephant Rock-": {
        "English_title": (
            "Jabal AlFil -Elephant Rock-"
        ),
        "English_description": (
     "The Elephant Rock is one of the world’s most popular rocks and the highlight of the region of AlUla."
        "Looking at it from afar, this rock seems like an elephant with a ground-bound trunk."
        "The Elephant Rock is also known as Jabal-AlFil in the Arabic language."
        "Standing at a height of 52 meters, the giant rock climbs three stories into the Arabian sky."
        "This mammoth stands out among the other hand-carved, ornate structures of nearby Hegra’s Nabataean tombs as it was shaped by natural forces."
        "The trunk and body of this red sandstone beast were shaped through water and wind erosion that was caused over millions of years."
        "The huge elephant stands in a landscape of golden sands, surrounded by other rocky formations which are equally impressive in size."
        "Regardless, the Elephant Rock still overshadows all that falls in its sight and acts as a reminder of the sands of time."
        "The beauty of this mighty elephant increases at nightfall where it becomes more lifelike in the warm lights that have been installed at the site."
        "Visiting the rock at night is advantageous to avoid the hot weather during the summer day."    ),
         "Arabic_title": (
         "جبل الفيل"
        ),
        "Arabic_description": (   
 
               
        "صخرة الفيل هي واحدة من أشهر الصخور في العالم وأبرز معالم منطقة العلا."
        "عند النظر إليها من بعيد، تبدو هذه الصخرة وكأنها فيل ذو خرطوم ممتد نحو الأرض."
        "تُعرف صخرة الفيل أيضًا باسم جبل الفيل باللغة العربية."
        "بارتفاع يبلغ 52 مترًا، ترتفع هذه الصخرة العملاقة ثلاثة طوابق في سماء شبه الجزيرة العربية."
        "تتميز هذه الصخرة العملاقة عن الهياكل المنحوتة يدويًا وزخارف مقابر الحِجر النبطية القريبة، حيث تشكلت بفعل القوى الطبيعية."
        "تكون خرطوم وجسم هذا الوحش الرملي الأحمر بفعل التعرية الناتجة عن الماء والرياح على مدى ملايين السنين."
        "تقف صخرة الفيل العملاقة في منظر طبيعي من الرمال الذهبية، محاطة بتشكيلات صخرية أخرى لا تقل عنها روعة من حيث الحجم."
        "ومع ذلك، فإن صخرة الفيل تظل تهيمن على كل ما يقع في مرمى نظرها وتعمل كتذكير برمال الزمن."
        "تزداد جمال هذه الصخرة الجبارة عند حلول الليل، حيث تصبح أشبه بالكائن الحي تحت الأضواء الدافئة المثبتة في الموقع."
        "زيارة الصخرة في الليل مفيدة لتجنب الطقس الحار خلال النهار الصيفي."
            ),
        "resources": [{"text": "Learn About Elephant Rock", "url": "https://www.ttnworldwide.com/ArticleMG/25322/AlUla-A-story-of-million-year-old-rocks-and-deep-history"}]
    },


"Maraya": {
        "English_title": (
            "Maraya"
        ),
        "English_description": (
         "‘Maraya’, or mirror in Arabic, is a purpose-built event venue in AlUla, Saudi Arabia."
                        "A gem of Saudi’s new north-west, Maraya is a cornerstone of Saudi’s ambitious development plans and now represents a world-class event venue."
                        "Covered entirely in mirrors, the building is a stunning visual, reflecting the impressive natural landscape of AlUla."
                        "Seemingly built with sand and stars, there are few comparisons globally that can match up to the awe-inspiring first impressions of Maraya."
                        "Beyond first glances, Maraya is a modern wonder in a place of ancient wonders."
                        "AlUla itself is home to many thousands of years of history - a crossroads of civilizations since 6 BCE."
                        "In a place driving change and vision, there is nowhere more fitting to bring your event and add to the incredible history of one of Saudi Arabia’s most precious places."
                        "Maraya offers a selection of inspiring purpose-built event spaces, including spacious rooms and grand foyers."
                        "Perfect for conferences & celebrations."
                        "Two commercial-grade kitchens provide world-class Food and Beverage service, capable of anything from lavish gala dinners to coffee breaks."   ),
         "Arabic_title": (
         "مرايا"
        ),
        "Arabic_description": (   
 
               
        "جوهرة من شمال غرب المملكة الجديد، تعد مرآة حجر الزاوية في خطط التنمية الطموحة للمملكة، وتمثل الآن مكانًا عالمي المستوى للفعاليات."
                        "مغطاة بالكامل بالمرايا، تشكل المبنى منظرًا بصريًا مذهلاً يعكس المشهد الطبيعي الرائع للعلا."
                        "ويبدو وكأنه بُني بالرمال والنجوم، مما يجعل من الصعب العثور على ما يضاهيه عالميًا في التأثير البصري المذهل عند النظر إليه لأول مرة."
                        "ما وراء الانطباعات الأولى، تُعد مرآة معجزة حديثة في مكان يزخر بالعجائب القديمة."
                        "تعتبر العلا موطنًا لآلاف السنين من التاريخ - ملتقى للحضارات منذ القرن السادس قبل الميلاد."
                        "في مكان يقود التغيير والرؤية، لا يوجد مكان أكثر ملاءمة لاستضافة حدثك وإضافة فصل جديد إلى التاريخ المذهل لأحد أغلى الأماكن في المملكة العربية السعودية." 
                        "توفر مرآة مجموعة مختارة من المساحات الملهمة المصممة خصيصًا للأحداث، بما في ذلك غرف واسعة وردهات كبيرة."
                        "مثالية للمؤتمرات والاحتفالات."
                        "تقدم مطبخان تجاريان من الطراز العالمي خدمات طعام ومشروبات استثنائية، قادرة على استضافة كل شيء من حفلات العشاء الفاخرة إلى استراحات القهوة."
        ),
        "resources": [{"text": "Learn About Maraya", "url": "https://www.marayaalula.com/aboutmaraya"}]
    },

"Hegra": {
        "English_title": (
            "Hegra"
        ),
        "English_description": (
                   "Saudi Arabia’s first UNESCO World Heritage Site, Hegra is a mesmerising and wonderfully-preserved archaeological playground for visitors to explore."
                        "Thought to have been built around the 6th century BCE, Hegra was populated by the Dadanites and Lihyanites."
                        "This settlement then flourished in the 1st century AD under the rule of the Nabataean people as it grew into a major city with dwellings, walls, more than 130 wells, irrigation channels, and reservoirs to collect rainwater."
                        "After Petra in Jordan, a couple of hundred kilometres to the north of AlUla, Hegra was the second city of the Nabataean people."
                        "Talented engineers, architects, and masters of irrigation, the Nabataeans were a nomad civilisation that became wealthy thanks to their control of the incense and spice trade routes."
                        "Their talent for masonry is clearly seen in the more than 110 tombs that form the highlight of any visit to Hegra, 94 of which are decorated."
                        "Cave drawings and more than 50 inscriptions from the Nabataean period offer an insight into how this advanced civilisation lived."
                        "Archaeologists believe the oldest tomb dates from the mid-1st century BCE, while the most recent originates from around 70 CE."
                        "Standing 21 metres in height is the Tomb of Lihyan Son of Kuza, the tallest tomb at Hegra and a landmark that takes the breath away."
                        "Partially enveloped by rock, the unfinished tomb reveals not only the skills of the Nabataean masons but also the epic scale of the task they undertook."
                           ),
         "Arabic_title": (
         "الحِجر"
        ),
        "Arabic_description": (   
 
                     "انطلق في رحلة عبر الزمن إلى الحِجر، أول موقع تراث عالمي لليونسكو في المملكة العربية السعودية، الذي يُعتبر موقع أثري فريد محفوظ كجوهرة ليستكشفها المغامرون ومحبي العراقة من كل مكان."
                        "تأسست الحِجر في القرن السادس قبل الميلاد، وشهدت حضارات متعاقبة من الدادانيين واللحيانيين، لكن ازدهارها الحقيقي جاء في القرن الأول الميلادي تحت حكم الأنباط، حيث تحولت إلى مدينة عظيمة تنبض بالحياة، ونمت لتصبح مدينة كبرى بها العديد من المساكن وأكثر من 130 بئرًا وقنوات ري وخزانات لجمع مياه الأمطار."
                        "وبعد مدينة البتراء الموجودة في الأردن على بعد بضع مئات من الكيلومترات إلى الشمال من العلا، كانت الحِجر المدينة الثانية للأنباط."
                        "كان الأنباط مهندسين موهوبين ومعماريين بارعين وأساتذة في ابتكارات الري، وكانت حضارتهم حضارة بدوية اكتسبت ثراءً واسعًا بفضل سيطرتهم على طرق تجارة البخور والتوابل."
                        "وتظهر مواهبهم في البناء بوضوح في المدافن التي يزيد عددها عن 110 مقبرة منها 94 تحمل زخارفًا رائعة وهي أبرز محطة في أي زيارة إلى منطقة الحِجر."
                        "فهناك رسومات الكهوف وأكثر من 50 نقشًا تعود جميعها إلى فترة مملكة الأنباط؛ وهي تقدم رؤية شاملة لطبيعة الحياة التي عاشها أهل تلك الحضارة."
                        "ويعتقد علماء الآثار أن أقدم المدافن يعود تاريخها إلى منتصف القرن الأول قبل الميلاد، في حين أن أحدثها تعود إلى عام 70 ميلادية على أقرب تقدير."
                        "وأبرز تلك المقابر هي مقبرة لحيان بن كوزا الذي يبلغ ارتفاعها 21 مترًا ليكون ذلك المعلم الذي يخطف الأنفاس هو أطول قبر في الحِجر."
                        "فالقبر غير المكتمل الذي تحيط به الصخور جزئيًا لا يكشف عن مهارات الأنباط السابقين لعصرهم فحسب، بل يكشف أيضاً عن حجم المهمة العظيمة التي أبدعوا في بنائه."
        ),
        "resources": [{"text": "Learn About Hegra", "url": "https://www.experiencealula.com/ar/forever-revitalising/history/a-short-history-of-hegra"}]
    },
 "murabba palace": {
        "English_title": (
            "Al Murabba Historical Palace"
        ),
        "English_description": (
      "Al Murabba Historical Palace, established in 1939 by King Abdulaziz Al Saud, served as his official residence, the headquarters of state affairs, "
        "and a Diwan for meetings with global leaders. Located 2 km from old Riyadh, it played a pivotal role in shaping Saudi Arabia's political and economic landscape. "
        "This historic site witnessed the drafting of the Kingdom's first laws and the establishment of key institutions like the Ministry of Foreign Affairs."              ),
         "Arabic_title": (
         "قصر المربع"
        ),
        "Arabic_description": (   
         "إن قصر المربع واحد من القصور القديمة التراثية الواقعة في المملكة العربية السعودية على مساحة تبلغ 1,680 متر مربع."  
        "ترجع قيمة هذا القصر إلى الأحداث التاريخية التي عاصرها، كما أنه يحتوي على الكثير من مقتنيات الملك عبد العزيز بن عبد الرحمن آل سعود، بالإضافة إلى مجالس استقبال الضيوف الملكية."  
        "تم بناء قصر المربع التاريخي بأمر من الملك عبد العزيز آل سعود، ولذلك ينتسب إليه ويطلق عليه اسم قصر الملك عبد العزيز."  

        "مرّ تاريخ بناء قصر المربع بعدة مراحل؛ فإن بناءه بدأ عام 1356 هجري الموافق 1937 ميلادي، وانتقل إليه الملك عبد العزيز بن عبد الرحمن آل سعود عام 1357 هجري الموافق 1938 ميلادي، واستمرت الأعمال في القصر والمنطقة المحيطة له مدة 10 سنوات تقريبًا، مما يعني أنها كانت مستمرة حتى بعد انتقال الملك إليه."  

        "عاصر قصر المربع التاريخي في المملكة العربية السعودية كثيرًا من الأحداث المهمة خلال فترة حكم الملك عبد العزيز، وفيما يأتي بعضًا من أبرز المعلومات عن هذا القصر:"  

        "المنفذ لبناء القصر: بعد الأمر ببنائه من قبل الملك عبد العزيز آل سعود؛ تم تنفيذ بناء قصر المربع من قبل حمد بن قباع، وتم تخطيط هذا القصر بالشكل الذي يضمن وجود فناء فسيح تطل عليه الغرف والوحدات في الطابقين من الأربع جهات."  
        "الاستخدام في عهد الملك عبد العزيز: تم استخدام قصر المربع في عهد الملك عبد العزيز لإدارة شؤون المملكة العربية السعودية بالإضافة إلى استقبال ضيوفها، ومزاولة مهمات الحكم."  
        "طوابق القصر: يحتوي قصر المربع على طابقين اثنين فحسب، واختص الأرضي منهما بأمور الخدمات والموظفين بالإضافة إلى المهمات الإدارية، بينما أفرد الطابق الأول باستقبال الوفود وممارسة المهمات السياسية."
    ),
 "resources": [
        {"text": "Visit Al Murabba Palace", "url": "https://misbar.com/qna/2024/03/28/%D9%84%D9%85%D8%A7%D8%B0%D8%A7-%D8%B3%D9%85%D9%8A-%D9%82%D8%B5%D8%B1-%D8%A7%D9%84%D9%85%D8%B1%D8%A8%D8%B9-%D8%A8%D9%87%D8%B0%D8%A7-%D8%A7%D9%84%D8%A7%D8%B3%D9%85"}
    ]    },
 "Rejal Almaa": {
        "English_title": (
            "Rijal Heritage Village"
        ),
        "English_description": (
                   "Rijal Heritage Village, a gem just 45 km from Abha."  
                    "This stunning village, with its vibrant stone buildings and unique architecture, showcases 700 years of history."  
                    "Recognized globally as one of the best tourist villages, Rijal is a treasure trove of cultural tales and rare exhibits."  
                    "Discover the village museum, which spans 20 sections filled with traditional artifacts narrating the story of its inhabitants."  
                    "Enjoy breathtaking views from Al-Ous Castle and Shokan Mountain, and indulge in the local delight of authentic Al-Ami honey at the Honey Hut."  
                    "The scenic drive via Aqabat Al-Samma, connecting Abha to Rijal Almaa, adds to the unique charm of your visit."  
                    "Every corner of Rijal Heritage Village whispers stories of the past, promising an enriching and unforgettable experience."  
                    "The ancient remains and folk tales, hidden inside the unique architectural designs of Rijal Almaa village, which have been echoing for more than 700 years, are historical evidence that attest for its early civilization."  
                    "Its tall stone-built buildings were made using Basalt rocks, which gave them their strength and durability to last through time."  
                    "The white Quartz stone is found extensively on the outer walls, while the interior is embellished with magnificently made artwork."                      ),
         "Arabic_title": (
         "قرية رجال المع التراثية"
        ),
        "Arabic_description": (   
    "قرية رجال التراثية، الجوهرة الواقعة على بعد 45 كم من مدينة أبها."  
                    "تتميز القرية بمبانيها الحجرية الملونة وطرازها المعماري الفريد الذي يعكس تاريخاً يمتد لـ 700 عام."  
                    "وقد نالت القرية اعترافاً عالمياً كواحدة من أفضل القرى السياحية، لتصبح كنزاً حافلاً بالقصص الثقافية والمعروضات النادرة."  
                    "اكتشف متحف القرية الذي يضم قطع تراثية تروي قصص سكانها."  
                    "ولا تفوق الاستمتاع بالإطلالات الخلابة من أعلى 'قصبة العوص' و'جبل شوكان'، وتذوق العسل الألمعي الأصيل في 'كوخ العسل'."  
                    "أما الرحلة البرية عبر عقبة الصماء، التي تربط أبها برجال ألمع، فهي رحلة ساحرة لزوار المنطقة."  
                    "تعد المعالم الأثرية والحكايات الشعبية المخبأة داخل الطراز المعماري الفريد لقرية رجال ألمع أحد الشواهد التاريخية الدالة على بعدها الحضاري الذي يتردد صداه لأكثر من 700 عام، حيث شيّدت مبانيها الحجرية الشاهقة باستعمال صخور البازلت مما منحها القوة والمتانة."  
                    "وينتشر حجر الكوارتز الأبيض على جدرانها الخارجية، بينما تزينها من الداخل رسوم فنية بديعة الصنع."
      ),
        "resources": [
        {"text": "Learn More About Rijal Heritage Village", "url": "https://www.visitsaudi.com/en/aseer/attractions/rijal-almaa-of-aseer"}
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
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
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


normalized_landmark_info = {key.lower(): value for key, value in landmark_info.items()}

# Camera Input
image_data = st.camera_input("Take a picture")
if image_data:
    # Convert image_data to OpenCV format
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the captured image
    st.image(img, caption="Captured Image", use_container_width=True)


    # Run detection on the captured image
    results = model.predict(source=img, save=False)


    # Extract detected classes
    detected_classes = results[0].boxes.cls
    class_names = results[0].names

    if len(detected_classes) > 0:
        detected_class = class_names[int(detected_classes[0])]

        # Normalize the detected class
        normalized_class = detected_class.strip().lower()

        # Check if the normalized class exists in the dictionary
        if normalized_class in normalized_landmark_info:
            info = normalized_landmark_info[normalized_class]
            st.subheader(f"📍 {info.get('English_title', 'Unknown Landmark')}")
            st.write(info.get('English_description', 'No description available.'))
            st.markdown(
                f"<h3 style='text-align: right;'>{info.get('Arabic_title', 'Unknown Landmark')} 📍</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align: right; direction: rtl;'>{info.get('Arabic_description', 'No description available.')}</p>",
                unsafe_allow_html=True
            )
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

        # Check if the normalized class exists in the dictionary
        if normalized_class in normalized_landmark_info:
            info = normalized_landmark_info[normalized_class]

            # Retrieve the title based on the detected class
            detected_title = info.get("English_title", "Unknown Landmark")

            # Display the title in the subheader
            st.subheader(f"📍 {detected_title}")

            # Display description and resources
            st.write(info.get("English_description", "No description available."))
            
                        # Retrieve the Arabic title and description
            Ar_detected_title = info.get("Arabic_title", "Unknown Landmark")
            Ar_detected_description = info.get("Arabic_description", "No description available.")
            
            
            # Display the Arabic title
            st.markdown(f"<h3 style='text-align: right;'>{Ar_detected_title} 📍</h3>", unsafe_allow_html=True)

            # Display the Arabic description
            st.markdown(
                f"<p style='text-align: right; direction: rtl;'>{Ar_detected_description}</p>",
                unsafe_allow_html=True
            )
            
            st.write("**Resources:**")
            for resource in info.get("resources", []):
                resource_text = resource.get("text", "No text")
                resource_url = resource.get("url", "#")
                st.markdown(f"- [{resource_text}]({resource_url})")
        else:
            st.write("No information available for this landmark.")
    else:
        st.write("No landmarks detected in the image.")
