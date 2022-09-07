# import io
import os.path

import streamlit as st
from PIL import Image
import cns

# import tkinter as tk
# from tkinter import filedialog

# c_image = st.sidebar.selectbox(
#    'Select content',
#    ('b.png', 'input.png')
# )
sv_clrs = 0
st.markdown("<h1 style='text-align: center; color: black;'>Сервис для комбинированной стилизации изображений</h1>",
            unsafe_allow_html=True)


# def load_image():
#     image = Image.open(c_img)
#     st.image(image, width=300)


with st.sidebar:
    c_image = None
    st.title('Исходное изображение')
    option = st.selectbox('Загрузите изображение или сделайте фото', ('Upload_image', 'Take_a_photo'))
    if option == 'Upload_image':
        upload_image = st.file_uploader("Загрузите исходное изображение", type=["png", "jpg", "jpeg"])
        if upload_image:
            c_image = upload_image
    elif option == 'Take_a_photo':
        take_a_photo = st.camera_input("Сделайте фото")
        if take_a_photo:
            c_image = take_a_photo

# accept_multiple_files=True)

    if c_image is not None:
        # print(type(c_image))
        # if len(c_image) != 0:
        # Посмотреть подробности
        # st.write(type(c_image))
        # file_details = {"Название файла": c_image.name,
        #                "Тип файла": c_image.type,
        #                "Размер файла": c_image.size}
        # st.write(file_details)
        # if len(c_image) != 0:
        with open(os.path.join('input', c_image.name), "wb") as f:
            f.write(c_image.getbuffer())
        # st.success("Файл сохранён")
        c_img = os.path.join("input", c_image.name)
        st.write("### Исходное изображение:")
        st.write("Название изображения: ", c_image.name)
        image = Image.open(c_img)
        st.image(image, width=300)


# save_content_path = 'C:/CNS/input/'
# completeName = os.path.join(save_content_path, c_image.name)
# img.save(completeName)


# with st.sidebar:
#     s_image = st.file_uploader("Upload style image", type=["png", "jpg", "jpeg"], accept_multiple_files=1)
#
# save_styles_path = 'C:/CNS/input/'
# completeName = os.path.join(save_styles_path, s_image.name)
# img = Image.open(s_image)
# img.save(completeName)

# s_image = st.sidebar.selectbox(
#    'Select style',
#    ('more.jpg', 'volna.jpg')
# )

# root = tk.Tk()
# root.withdraw()
# root.wm_attributes('-topmost', 1)
# with st.sidebar:
#     st.title('Стиль')
#     st.write('Пожалуйста выберите папку с вашими стилями:')
#     clicked = st.button('Выбрать папку')
#     if clicked:
#         dirname = st.text_input('Выберите папку:', filedialog.askdirectory(master=root))
#         if dirname is not None:
#             # Посмотреть подробности
#             st.write(type(dirname))

    st.title('Стиль')
    s_images = st.file_uploader("Загрузите стилевые изображения", type=["png", "jpg", "jpeg"],
                                accept_multiple_files=True)

    if s_images is not None:
        if len(s_images) != 0:
            st.write("### Изображения стилей:")
            # while os.path.exists("style" + str(i)) is True:
            c = 1
            flag = 0
            dirname = "style{}"
            while os.path.exists(dirname.format(c)):
                if len(os.listdir(dirname.format(c))) == 0:
                    # dirname = dirname.format(c)
                    flag = 1
                    break
                else:
                    # dirname = dirname.format(c)
                    c += 1
            # dirname = dirname.format(c)
            if flag == 0:
                os.mkdir(dirname.format(c))
            s_img = dirname.format(c)

            # st.write(len(s_image))
            for s_image in s_images:
                # bytes_data = s_image.read()
                image = Image.open(s_image)
                st.write("Название стиля: ", s_image.name)
                st.image(image, width=300)

                with open(os.path.join(dirname.format(c), s_image.name), "wb") as f:
                    f.write(s_image.getbuffer())
                # st.success("Файл сохранён")
            # st.write(type(i))
            # file_details = {"Название файла": i.name,
            #                 "Тип файла": i.type,
            #                 "Размер файла": i.size}
            # st.write(file_details)
            # with open(os.path.join('style', i.name), "wb") as f:
            #     f.write(i.getbuffer())
            # st.success("Файл сохранён")
            # i += 1
            # Посмотреть подробности

out_img = "output/ref1.png"

if s_images and c_image is not None:
    iters = st.number_input('Количество итераций:', value=100)
    s_scl = st.number_input('Масштаб стиля', value=1.0)
    # st.write(type(s_blend_w))
    # st.write(s_scl)
    s_blend_w = []
    if len(s_images) > 1:
        for s_image in s_images:
            blend_slider = st.slider('Изменение влияния стиля: ' + str(s_image.name), 0, 100, 50, 5)
            s_blend_w.append(blend_slider)

    sw = st.number_input('Влияние стиля:', value=1000)
    clr_cb = st.checkbox("Сохранить цвета")
    s_btn = st.button("Стилизовать")

# st.write("### Styles Images:")
# image = Image.open(s_img)
# st.image(image, width=400)

# st.write("### Styles Images:")
# image = Image.open(c_img)
# st.image(image, width=400
    if s_btn:
        if clr_cb:
            sv_clrs = 1
        # st.write(sv_clrs)
        # st.write("yes")
        # st.write(s_scl)
        cns.main(c_img, s_img, out_img, sv_clrs, s_scl, s_blend_w, iters, sw)
        # st.write("<h3 style='text-align: center; color: black;'>Стилизованное изображение:</h3>",
        #         unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Исходное изображение:")
            image = Image.open(c_img)
            st.image(image, use_column_width=True)
        with col2:
            st.write("Стили:")
            for s_image in s_images:
                # bytes_data = s_image.read()
                image = Image.open(s_image)
                st.image(image, use_column_width=True)
        with col3:
            st.write("Стилизованное изображение:")
            image = Image.open(out_img)
            st.image(image, use_column_width=True)
