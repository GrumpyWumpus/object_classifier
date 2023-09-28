import streamlit as st
from PIL import Image
from my_photo import check_photo

LABELS = ["戦車", "軍艦", "軍用機"]

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像識別デモアプリ")
st.sidebar.write("入力画像を戦車、軍艦、軍用機のいずれかに識別分類するデモアプリです。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        idx, per = check_photo(img)
        # 結果の表示
        st.subheader("判定結果")
        #st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")
        st.write("この写真は、" + str(per) + "%の確率で" + LABELS[idx] + "です。")

st.sidebar.write("")
st.sidebar.write("")
