import os.path
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import cns
import ych
import codecs

sv_clrs = 0
st.markdown("<h1 style='text-align: center; color: black;'>Сервис для комбинированной стилизации изображений</h1>",
            unsafe_allow_html=True)
out_img = "output/ref1.png"
image_size = 320
epochs = 1
batch_size = 4
dataset = 'val2017'
save_model_dir = 'models'
content_weight = 1e5
style_weight = 1e10
lr = 1e-3
log_interval = 1
style_size = 256
flag = 0
c_image = None
s_images = None
choose = None


def st_share(share_html):
    share_file = codecs.open(share_html, 'r')
    page = share_file.read()
    components.html(page, width=1000, height=1000, scrolling=False)


with st.sidebar:
    st.title('Обучить модель')
    t_image = st.file_uploader("Загрузите стиль для обучения", type=["png", "jpg", "jpeg"])
    st.title('Исходное изображение')

    c_option = st.selectbox('Загрузите изображение или сделайте фото', ('Upload_image', 'Take_a_photo'))

    if c_option == 'Upload_image':
        upload_c_image = st.file_uploader("Загрузите исходное изображение", type=["png", "jpg", "jpeg"])
        if upload_c_image:
            c_image = upload_c_image
    elif c_option == 'Take_a_photo':
        take_a_photo = st.camera_input("Сделайте фото")
        if take_a_photo:
            c_image = take_a_photo

    if c_image is not None:
        with open(os.path.join('input', c_image.name), "wb") as f:
            f.write(c_image.getbuffer())
        # st.success("Файл сохранён")
        c_img = os.path.join("input", c_image.name)
        st.write("### Исходное изображение:")
        st.write("Название изображения: ", c_image.name)
        image = Image.open(c_img)
        st.image(image, width=300)

    st.title('Стиль')
    s_option = st.selectbox('Загрузите свои стили или выберите готовый', ('Upload_styles', 'Choose_model'))
    if s_option == 'Upload_styles':
        upload_s_image = st.file_uploader("Загрузите стилевые изображения", type=["png", "jpg", "jpeg"],
                                          accept_multiple_files=True)
        if upload_s_image:
            s_images = upload_s_image
    elif s_option == 'Choose_model':
        choose_model = st.selectbox("Выберите модель",
                                    ('BWScream',
                                     'Pixel_Waterfall', 'Shtrih', 'SWB'))
        if choose_model:
            choose = choose_model

    if s_images is not None:
        if len(s_images) != 0:
            st.write("### Изображения стилей:")
            c = 1
            flag = 0
            dirname = "style{}"
            while os.path.exists(dirname.format(c)):
                if len(os.listdir(dirname.format(c))) == 0:
                    flag = 1
                    break
                else:
                    c += 1
            if flag == 0:
                os.mkdir(dirname.format(c))
            s_img = dirname.format(c)

            for s_image in s_images:
                image = Image.open(s_image)
                st.write("Название стиля: ", s_image.name)
                st.image(image, width=300)

                with open(os.path.join(dirname.format(c), s_image.name), "wb") as f:
                    f.write(s_image.getbuffer())


if c_image and choose is not None:
    st.write('Нажмите, чтобы применить обученный стиль к исходному изображению')
    m_btn = st.button("Применить модель")
    model = os.path.join('models', choose + '.model')

    if m_btn:
        ych.stylize(c_img, model, out_img)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Исходное изображение:")
            image = Image.open(c_img)
            st.image(image, use_column_width=True)
        with col2:
            st.write("Стилизованное изображение:")
            image = Image.open(out_img)
            st.image(image, use_column_width=True)

if s_images is None:
    if c_image is None:
        if t_image is not None:
            st.write('Нажмите, чтобы обучить модель новому стилю')
            t_btn = st.button("Обучить модель")
            if t_btn:
                ych.train(image_size, epochs, batch_size, dataset, save_model_dir, content_weight, style_weight, lr,
                          log_interval, style_size, t_image)
                st.success('Модель успешно обучена')


if s_images and c_image is not None and t_image is None:
    iters = st.number_input('Количество итераций:', value=10)
    s_scl = st.number_input('Масштаб стиля', value=1.0)

    s_blend_w = []
    if len(s_images) > 1:
        for s_image in s_images:
            blend_slider = st.slider('Изменение влияния стиля: ' + str(s_image.name), 0, 100, 50, 5)
            s_blend_w.append(blend_slider)

    sw = st.number_input('Влияние стиля:', value=1000)
    clr_cb = st.checkbox("Сохранить цвета")
    s_btn = st.button("Стилизовать")

    if s_btn:
        if clr_cb:
            sv_clrs = 1

        cns.main(c_img, s_img, out_img, sv_clrs, s_scl, s_blend_w, iters, sw)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Исходное изображение:")
            image = Image.open(c_img)
            st.image(image, use_column_width=True)
        with col2:
            st.write("Стили:")
            for s_image in s_images:
                image = Image.open(s_image)
                st.image(image, use_column_width=True)
        with col3:
            st.write("Стилизованное изображение:")
            image = Image.open(out_img)
            st.image(image, use_column_width=True)
