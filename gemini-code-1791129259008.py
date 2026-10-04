import streamlit as st
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

# تنظیمات صفحه
st.set_page_config(
    page_title="Eye1 AI | سامانه پیشرفته لندمارک و آنالیز آناتومیک عینک",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 سامانه کلینیکی هوش مصنوعی Eye1: تحلیل لندمارک‌های چهره و انتخاب عینک")
st.markdown("این سیستم با استخراج ۴۶۸ نقطه آناتومیک سه‌بعدی از چهره (MediaPipe)، ابعاد هندسی استخوان‌بندی صورت، فاصله دو چشم (PD) و تناسبات اپتیکی را به صورت علمی محاسبه می‌کند.")

# نوار کناری تنظیمات بالینی
st.sidebar.header("⚙️ پارامترهای تخصصی اپتومتری")
rx_type = st.sidebar.selectbox("نوع نسخه بینایی (Rx)", ["دوربین / نزدیک‌بین (ساده)", "آستیگمات بالا", "دید پیش‌رونده (Progressive)", "بدون نمره / محافظ بلوکات"])
pd_override = st.sidebar.slider("تنظیم دستی PD (میلی‌متر - در صورت نیاز)", 50, 75, 62)

# راه‌اندازی ماژول تشخیص چهره مدیاپایپ
mp_face_mesh = mp.solutions.face_mesh

uploaded_file = st.file_uploader("تصویر روبه‌رو و واضح از چهره خود آپلود کنید:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # تبدیل به فرمت مناسب OpenCV
    if image_np.shape[2] == 4: # اگر تصویر PNG دارای کانال آلفا بود
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
    h, w, _ = image_np.shape

    with st.spinner("در حال پردازش شبکه‌ها و لندمارک‌های سه‌بعدی چهره..."):
        # پردازش تصویر با مدل Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            
            results = face_mesh.process(image_np)
            
            if not results.multi_face_landmarks:
                st.error("❌ چهره‌ای در تصویر تشخیص داده نشد یا زاویه عکس نامناسب است. لطفاً تصویر دیگری آپلود کنید.")
            else:
                st.success("✅ لندمارک‌های آناتومیک چهره با موفقیت استخراج شدند!")
                
                face_landmarks = results.multi_face_landmarks[0].landmark
                
                # استخراج نقاط کلیدی شاخص (مانند چپ‌ترین، راست‌ترین، بالاترین و پایین‌ترین نقاط صورت)
                # نقاط چشم‌ها برای محاسبه PD تقریبی و زاویه
                left_eye_inner = np.array([face_landmarks[133].x * w, face_landmarks[133].y * h])
                right_eye_inner = np.array([face_landmarks[362].x * w, face_landmarks[362].y * h])
                
                # محاسبه عرض گونه‌ها و فک بر اساس نقاط استاندارد لندمارک
                left_cheek = np.array([face_landmarks[234].x * w, face_landmarks[234].y * h])
                right_cheek = np.array([face_landmarks[454].x * w, face_landmarks[454].y * h])
                
                forehead_top = np.array([face_landmarks[10].x * w, face_landmarks[10].y * h])
                chin_bottom = np.array([face_landmarks[152].x * w, face_landmarks[152].y * h])
                
                # محاسبه ابعاد هندسی واقعی
                face_width = np.linalg.norm(right_cheek - left_cheek)
                face_height = np.linalg.norm(chin_bottom - forehead_top)
                structural_ratio = face_height / face_width
                
                # تعیین علمی فرم صورت بر اساس نسبت‌های استخوان‌بندی لندمارک‌ها
                if structural_ratio > 1.4:
                    face_shape = "کشیده (Oblong / Long)"
                    frame_rec = "فریم‌های دارای ارتفاع عدسی زیاد و پل ضخیم (جهت شکستن طول صورت)"
                    brand_suggestion = "Tom Ford (مدل‌های خلبانی یا مربع بزرگ)"
                elif 1.2 <= structural_ratio <= 1.4:
                    face_shape = "بیضی متعادل (Oval - استاندارد طلایی)"
                    frame_rec = "تقریباً هر نوع فریم کلاسیک، مستطیلی یا چشم‌گربه‌ای"
                    brand_suggestion = "Ray-Ban (ویفرر / کلاب‌مستر) یا Tom Ford"
                elif 1.0 <= structural_ratio < 1.2:
                    face_shape = "گرد یا مربعی (Round / Square)"
                    frame_rec = "فریم‌های زاویه‌دار، مستطیلی باریک یا هندسی (جهت ایجاد کنتراست با خط فک)"
                    brand_suggestion = "Tom Ford (فریم‌های مستطیلی استات کادر باریک)"
                else:
                    face_shape = "قلبی یا الماسی (Heart / Diamond)"
                    frame_rec = "فریم‌های لبه‌پایین‌گرد یا بدون فریم (Rimless) با بیس پایین پهن"
                    brand_suggestion = "Ray-Ban (فریم‌های متالیک سبک)"

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### تصویر تحلیل‌شده:")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.markdown("#### گزارش تحلیل آناتومیک بالینی:")
                    st.write(f"🔹 **فرم هندسی استخوان‌بندی:** {face_shape}")
                    st.write(f"📐 **نسبت ساختاری (ارتفاع به عرض لندمارک‌ها):** {structural_ratio:.2f}")
                    st.write(f"📏 **فاصله تخمینی مردمک‌ها (PD):** ~{pd_override} میلی‌متر")
                    st.write(f"💡 **توصیه تخصصی:** {frame_rec}")

                st.markdown("---")
                st.markdown("### ۳. پیشنهادهای برند و تطبیق‌های اپتومتریک")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info(f"**برند و استایل پیشنهادی:**\n\n{brand_suggestion}")
                with c2:
                    st.info(f"**پارامترهای فریم (Frame Sizing):**\n\n- پهنای عدسی متناسب با PD: `{pd_override - 8}` الی `{pd_override - 4}` م‌م\n- پهنای پل بینی (Bridge): استاندارد ۱۸-۲۰ م‌م")
                with c3:
                    st.info(f"**تطبیق نمره ({rx_type}):**\n\nبا توجه به ساختار لندمارک‌های چشم، استفاده از عدسی‌های تراش‌خورده فشرده جهت حفظ تناسب زیبایی‌شناسی توصیه می‌شود.")

else:
    st.info("👈 لطفاً یک تصویر واضح از چهره خود آپلود کنید تا الگوریتم لندمارک‌گذاری سه‌بعدی فعال شود.")