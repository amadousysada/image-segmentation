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

# Option pour le mode couleur
color_mode = st.checkbox("Utiliser des couleurs pour le masque (plus facile à distinguer)", value=True)

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
                params = {"color_mode": color_mode}
                resp = requests.post(f"{STREAMLIT_API_URL}/segment/", files=files, params=params, timeout=60)
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
        if color_mode:
            st.subheader("Mask prédit (8 classes en couleurs)")
        else:
            st.subheader("Mask prédit (8 classes en niveaux de gris)")
        
        # Informations de débogage
        mask_array = np.array(mask_image)
        unique_values = np.unique(mask_array)
        
        # Créer deux colonnes pour l'affichage
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if color_mode:
                st.image(mask_image, use_container_width=True, clamp=True)
            else:
                st.image(mask_image, use_container_width=True, clamp=True, channels="L")
        
        with col2:
            st.write("**Informations du masque:**")
            st.write(f"📏 Dimensions: {mask_array.shape}")
            if color_mode:
                st.write(f"🎨 Mode: Couleurs RGB")
                st.write(f"🖼️ Classes détectées: {len(np.unique(mask_array.flatten())) if len(mask_array.shape) == 3 else len(unique_values)}")
                
                # Légende des couleurs pour le mode couleur (correspondant au GROUP_PALETTE du notebook)
                st.write("**Légende des couleurs:**")
                color_legend = [
                    "🟣 Classe 0: Flat (route, trottoir) - Violet-gris",
                    "🔴 Classe 1: Human (personne, cycliste) - Rouge-crimson", 
                    "🔵 Classe 2: Vehicle (voiture, camion) - Bleu foncé",
                    "⚫ Classe 3: Construction (bâtiment, mur) - Gris foncé",
                    "🟡 Classe 4: Object (poteau, panneau) - Jaune",
                    "🟢 Classe 5: Nature (végétation, terrain) - Vert olive",
                    "🩵 Classe 6: Sky (ciel) - Bleu ciel",
                    "🖤 Classe 7: Void (non labellisé, hors ROI) - Noir"
                ]
                for legend in color_legend:
                    st.write(f"• {legend}")
            else:
                st.write(f"📊 Valeurs min/max: {mask_array.min()}/{mask_array.max()}")
                st.write(f"🎨 Classes détectées: {len(unique_values)}")
                
                # Afficher la légende des couleurs pour niveaux de gris
                st.write("**Légende des niveaux de gris:**")
                color_legend = {
                    0: "Noir (classe 0)",
                    36: "Gris très foncé (classe 1)", 
                    73: "Gris foncé (classe 2)",
                    109: "Gris moyen-foncé (classe 3)",
                    146: "Gris moyen (classe 4)",
                    182: "Gris moyen-clair (classe 5)",
                    219: "Gris clair (classe 6)",
                    255: "Blanc (classe 7)"
                }
                
                for value in unique_values:
                    if value in color_legend:
                        st.write(f"• {color_legend[value]}")
                    else:
                        st.write(f"• Valeur {value} (classe inconnue)")

        # 5. (Optionnel) Superposition semi-transparente
        st.subheader("Superposition")
        overlay = Image.blend(input_image.resize(mask_image.size),
                              mask_image.convert("RGB").resize(input_image.size),
                              alpha=0.5)
        st.image(overlay, use_container_width=True)