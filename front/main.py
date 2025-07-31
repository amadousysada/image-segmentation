import os
import io
import requests
import streamlit as st
from PIL import Image
import numpy as np

# Config
STREAMLIT_API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip('/')

st.set_page_config(page_title="Segmentation Demo", layout="centered")
st.title("Segmentation sémantique d'image")

# 1. Upload d'image
uploaded_file = st.file_uploader("Choisissez une image PNG ou JPG", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Affiche l'image d'entrée
    input_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Image d'entrée")
    st.image(input_image, use_container_width=True)

    # 2. Bouton pour lancer la requête
    if st.button("Segmenter"):
        with st.spinner("Appel à l’API de segmentation…"):
            try:
                # On renvoie le fichier sous form-data (clé 'file' ou autre selon votre API)
                files = {"picture": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                resp = requests.post(f"{STREAMLIT_API_URL}/segment/", files=files, timeout=60)
                resp.raise_for_status()
                # 3. Lecture du mask PNG retourné
                mask_bytes = resp.content
                mask_image = Image.open(io.BytesIO(mask_bytes))
            except requests.exceptions.HTTPError as errh:
                st.error(f"Erreur HTTP : {errh}")
                st.stop()
            except requests.exceptions.RequestException as err:
                st.error(f"Erreur réseau : {err}")
                st.stop()
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
                st.stop()

        # 4. Affichage du mask en niveaux de gris
        st.subheader("Mask prédit (classes par pixel)")
        
        # Informations de débogage
        mask_array = np.array(mask_image)
        st.write(f"Dimensions du masque: {mask_array.shape}")
        st.write(f"Valeurs min/max du masque: {mask_array.min()}/{mask_array.max()}")
        st.write(f"Valeurs uniques: {np.unique(mask_array)}")
        
        st.image(mask_image, use_container_width=True, clamp=True, channels="L")

        # 5. (Optionnel) Superposition semi-transparente
        st.subheader("Superposition")
        overlay = Image.blend(input_image.resize(mask_image.size),
                              mask_image.convert("RGB").resize(input_image.size),
                              alpha=0.5)
        st.image(overlay, use_container_width=True)