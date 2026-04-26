import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import io
import zipfile, tempfile, os

st.set_page_config(layout="wide")
st.markdown("""
<style>
/* HEADER BOX */
.header-box {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 25px;
    border-radius: 20px;
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 25px;
}

/* MAIN CARD */
.dashboard-card {
    background: #ffffff;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}

/* SECTION TITLE */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 10px;
}

/* TEXT */
.text {
    font-size: 15px;
    color: #334155;
    line-height: 1.6;
}

/* LINK BUTTON */
.link-btn a {
    display: inline-block;
    background-color: #2563eb;
    color: white !important;
    padding: 8px 14px;
    border-radius: 10px;
    text-decoration: none;
    margin-right: 10px;
}
.link-btn a:hover {
    background-color: #1d4ed8;
}

/* Result cards styling */
.card {
    background-color: white;
    border-radius: 18px;
    padding: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    transition: transform 0.2s;
}
.card:hover {
    transform: scale(1.02);
}

.card-title {
    color: #2563eb;
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
}

.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    font-size: 13px;
    gap: 4px;
}

.metric-box {
    background: #f1f5f9;
    padding: 6px;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box">Image Enhancement using Histogram Equalization and Bilateral Filtering</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">Project Details</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="text">
        <b>Course</b><br>
        IT416 Computer Vision Course Project
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="text">
        <br><b>Authors</b><br>
        1. Kalal Pavan Teja - 23I1T030<br>
        2. Medicharla Harsha Naga Durga Sathish - 23I1T038<br>
        3. Medikurthi Sai Govardhan Rao - 23I1T039
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><b>Resources</b>", unsafe_allow_html=True)
        if os.path.exists("paper.pdf"):
            st.markdown(f"""
            <div class="link-btn">
                <a href="https://colab.research.google.com/drive/1nTlh7rglwI4plA4wOejfknYRjCfSOmaw?usp=sharing" target="_blank">Open Colab</a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="link-btn">
                <a href="https://colab.research.google.com/drive/1nTlh7rglwI4plA4wOejfknYRjCfSOmaw?usp=sharing" target="_blank">Open Colab</a>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="text">
        This dashboard enhances low-light images using:
        <ul>
        <li>Histogram Equalization</li>
        <li>Bilateral Filtering</li>
        <li>Wavelet Fusion</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def compute_metrics(ref, img):
    ref = ref.astype(np.float32)
    img = img.astype(np.float32)

    mse = np.mean((ref - img) ** 2)
    p = 100 if mse == 0 else 10 * np.log10((255**2)/mse)

    s = ssim(ref, img, channel_axis=2, data_range=255)

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))

    contrast = np.std(img)

    return p, s, entropy, contrast

def apply_HE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def apply_CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(2.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

def apply_RD(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y,U,V = cv2.split(yuv)
    Y = Y.astype(np.float32)/255.0
    blur = cv2.GaussianBlur(Y,(15,15),0)
    rd = np.log(Y+1e-6) - np.log(blur+1e-6)
    rd = cv2.normalize(rd,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([rd,U,V]), cv2.COLOR_YUV2RGB)

def apply_ESIHE(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y = yuv[:,:,0]
    mean = int(np.mean(y))
    lower = y.copy()
    upper = y.copy()
    lower[y > mean] = 0
    upper[y <= mean] = 0
    result = np.zeros_like(y)
    result[y <= mean] = cv2.equalizeHist(lower)[y <= mean]
    result[y > mean] = cv2.equalizeHist(upper)[y > mean]
    yuv[:,:,0] = result
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def apply_RGHS(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L,A,B = cv2.split(lab)
    p_low, p_high = np.percentile(L,(1,99))
    if p_high - p_low < 1e-6:
        L_stretch = L
    else:
        L_stretch = np.clip((L - p_low)*255/(p_high - p_low),0,255)
    L_final = cv2.addWeighted(L,0.5,L_stretch.astype(np.uint8),0.5,0)
    return cv2.cvtColor(cv2.merge([L_final,A,B]), cv2.COLOR_LAB2RGB)

def our_method_with_stages(img):
    stages = {}
    stages["Input"] = img.copy()

    img_f = img.astype(np.float32)/255.0
    output = np.zeros_like(img_f)

    for c in range(3):
        channel = img_f[:,:,c]

        LL,(LH,HL,HH) = pywt.dwt2(channel,'haar')

        LL_vis = cv2.normalize(LL,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        LL_vis = cv2.resize(LL_vis,(img.shape[1],img.shape[0]))
        LL_vis = cv2.cvtColor(LL_vis, cv2.COLOR_GRAY2RGB)
        stages["Wavelet_LL"] = LL_vis

        LL = cv2.bilateralFilter(LL.astype(np.float32),5,50,50)
        blur = cv2.GaussianBlur(LL,(3,3),0.5)
        LL_retinex = np.log1p(LL) - 0.7*np.log1p(blur+1e-6)

        LL_norm = cv2.normalize(LL_retinex,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        LL_he = cv2.equalizeHist(LL_norm)
        LL_he = cv2.addWeighted(LL_norm,0.6,LL_he,0.4,0)

        LL_vis2 = cv2.resize(LL_he,(img.shape[1],img.shape[0]))
        LL_vis2 = cv2.cvtColor(LL_vis2, cv2.COLOR_GRAY2RGB)
        stages["LL_Enhanced"] = LL_vis2

        LL_he = LL_he.astype(np.float32)/255.0

        def threshold(x):
            T = 0.2*np.std(x)
            return np.where(np.abs(x)>T,x,0)

        LH,HL,HH = threshold(LH),threshold(HL),threshold(HH)

        rec = pywt.idwt2((LL_he,(LH,HL,HH)),'haar')
        rec = cv2.resize(rec,(img.shape[1],img.shape[0]))
        output[:,:,c] = rec

    output = np.clip(output,0,1)
    rec_img = (output*255).astype(np.uint8)
    stages["Reconstructed"] = rec_img

    final = (np.power(output,0.85)*255).astype(np.uint8)
    final = cv2.filter2D(final,-1,np.array([[0,-1,0],[-1,6,-1],[0,-1,0]]))
    stages["Final"] = final

    return final, stages

def load_images(uploaded):
    images=[]
    filename = uploaded.name.lower()
    if filename.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,uploaded.name)
            open(path,"wb").write(uploaded.read())
            try:
                zipfile.ZipFile(path).extractall(tmp)
            except zipfile.BadZipFile:
                st.error("Invalid ZIP file. Please upload a valid .zip archive.")
                return []
            for r,_,f in os.walk(tmp):
                for file in f:
                    if file.lower().endswith(("jpg","png","jpeg")):
                        img=cv2.imread(os.path.join(r,file))
                        if img is not None:
                            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            images.append((file,img))
    else:
        bytes=np.asarray(bytearray(uploaded.read()),dtype=np.uint8)
        img=cv2.imdecode(bytes,1)
        if img is None:
            st.error("Could not read the uploaded image. Please upload JPG/PNG/JPEG.")
            return []
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append((uploaded.name,img))
    return images

uploaded = st.file_uploader("Upload Image or ZIP", type=["jpg","png","jpeg","zip"])

if uploaded:

    imgs = load_images(uploaded)
    if not imgs:
        st.warning("No valid images found in the uploaded file.")
        st.stop()
    names = [n for n,_ in imgs]
    selected = st.selectbox("Select Image", names)
    img = dict(imgs)[selected]

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    methods = [
        ("Original", img),
        ("HE", apply_HE(img)),
        ("CLAHE", apply_CLAHE(img)),
        ("RD", apply_RD(img)),
        ("ESIHE", apply_ESIHE(img)),
        ("RGHS", apply_RGHS(img)),
        ("Fusion", cv2.bilateralFilter(apply_HE(img),5,75,75)),
        ("Our", our_method_with_stages(img)[0])
    ]

    metrics_rows = []

    cols = st.columns(4)

    for i,(name,im) in enumerate(methods):
        with cols[i%4]:
            p,s,e,c = compute_metrics(img,im)
            metrics_rows.append({
                "Method": name,
                "PSNR": round(float(p), 2),
                "SSIM": round(float(s), 2),
                "Entropy": round(float(e), 2),
                "Contrast": round(float(c), 2)
            })

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">{name}</div>', unsafe_allow_html=True)
            st.image(im, use_container_width=True)
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">PSNR<br>{p:.2f}</div>
                <div class="metric-box">SSIM<br>{s:.2f}</div>
                <div class="metric-box">Entropy<br>{e:.2f}</div>
                <div class="metric-box">Contrast<br>{c:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Detailed Analysis</div>', unsafe_allow_html=True)

    for name, im in methods:
        st.markdown(f"### {name}")

        c1,c2 = st.columns([1,1.3])
        with c1:
            st.image(im, use_container_width=True)
        with c2:
            gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(gray.ravel(), bins=256)
            ax.set_title(f"{name} Histogram")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown('<div class="section-title">Stage-wise Outputs</div>', unsafe_allow_html=True)

    _, stages = our_method_with_stages(img)

    stage_cols = st.columns(len(stages))

    for i,(k,v) in enumerate(stages.items()):
        with stage_cols[i]:
            v = cv2.resize(v,(img.shape[1],img.shape[0]))
            st.markdown(f"**{k}**")
            st.image(v, use_container_width=True)
            p,s,e,c = compute_metrics(img,v)
            st.caption(f"PSNR: {p:.2f} | SSIM: {s:.2f}")

    st.markdown('<div class="section-title">Metrics Summary</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown('<div class="section-title">Download Results</div>', unsafe_allow_html=True)

    if len(imgs) == 1:
        final_img = our_method_with_stages(img)[0]
        _, buffer = cv2.imencode(".png", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        st.download_button(
            label="Download Final Enhanced Image",
            data=buffer.tobytes(),
            file_name="final_output.png",
            mime="image/png"
        )
    else:
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, im in imgs:
                final_img = our_method_with_stages(im)[0]
                _, buffer = cv2.imencode(".png", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
                zf.writestr(f"{name}_final.png", buffer.tobytes())

        st.download_button(
            label="Download All Final Outputs (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="final_outputs.zip",
            mime="application/zip"
        )