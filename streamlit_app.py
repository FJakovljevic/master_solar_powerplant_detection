import folium
import streamlit as st
import tensorflow as tf
from streamlit_folium import st_folium

from utils import model_find_polygons

st.set_page_config(layout="wide")
st.markdown(
    """<style>
            [data-testid="block-container"] {padding: 25px 0px 0px 0px !important;}
            footer {visibility: hidden !important;}
        </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_checkpoints/model-89-0.7998unet_resnet34_bs16.hdf5", compile=False)


@st.cache_resource
def load_feature_group():
    return folium.FeatureGroup(name="Detected Polygons")


TF_MODEL = load_model()

tiles_provider = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
m = folium.Map(location=[41.85345216, 14.94921684], zoom_start=15, tiles=tiles_provider, attr="Google")
st_data = st_folium(m, feature_group_to_add=load_feature_group(), use_container_width=True, height=540)

if st.button("MARK SOLAR POWER PLANTS"):
    south_west = st_data["bounds"]["_southWest"]
    north_east = st_data["bounds"]["_northEast"]
    bbox = [south_west["lng"], south_west["lat"], north_east["lng"], north_east["lat"]]
    polygons = model_find_polygons(TF_MODEL, bbox)

    for poly_coords in polygons:
        p = folium.Polygon(locations=poly_coords, color="blue", fill=True, fill_color="blue", fill_opacity=0.4)
        load_feature_group().add_child(p)

    st.rerun()
