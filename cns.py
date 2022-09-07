import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import streamlit as st

from PIL import Image

s_blend_w = []  # Смешивание весов стилей

Image.MAX_IMAGE_PIXELS = 1000000000  # Поддержка больших изображений


class fcn32s(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(fcn32s, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, (7, 7)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, (1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


def createSeq(source_ls, pool):
    layers = []
    in_channels = 3
    if pool == 'max':
        pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    elif pool == 'avg':
        pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else:
        raise ValueError("Неопознанный параметр пулинга")
    for c in source_ls:
        if c == 'Pool':
            layers += [pool2d]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


source_ls = {
    'VGG-16': [64, 64, 'Pool', 128, 128, 'Pool', 256, 256, 256, 'Pool', 512, 512, 512, 'Pool', 512, 512, 512, 'Pool']
}

dict_vgg16 = {
    'Conv': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
             'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
    'Relu': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2',
             'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3'],
    'Pool': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
}


def pickModel(f_mdl, pool):
    print("Обнаружена архитектура VGG-16")
    print("Используется модель fcn32s-heavy-pascal")
    cnn, lsLayers = fcn32s(createSeq(source_ls['VGG-16'], pool)), dict_vgg16
    return cnn, lsLayers


def loadmodels_print(cnn, lsLayers):
    c = 0
    for l in list(cnn):
        if "Conv2d" in str(l):
            in_c, out_c, ks = str(l.in_channels), str(l.out_channels), str(l.kernel_size)
            print(lsLayers['Conv'][c] + ": " +
                  (out_c + " " + in_c + " " + ks).replace(")", '').replace("(", '').replace(", ", ''))
            c += 1
        if c == len(lsLayers['Conv']):
            break


# Загрузка модели и конфигурация слоя пулинга
def loadModel(f_mdl, pool, use_gpu, dis_check):
    cnn, lsLayers = pickModel(str(f_mdl).lower(), pool)

    cnn.load_state_dict(torch.load(f_mdl), strict=(not dis_check))
    print("Успешная загрузка " + str(f_mdl))

    # Maybe convert the model to cuda now, to avoid later issues
    if "c" not in str(use_gpu).lower() or "c" not in str(use_gpu[0]).lower():
        cnn = cnn.cuda()
    cnn = cnn.features

    loadmodels_print(cnn, lsLayers)
    return cnn, lsLayers


def main(c_img, s_img, out_img, sv_clrs, s_scl, s_blend_w, iters, sw):
    cw = 5e0  # Веса контента
    dtype, backward_dev = gpuConf()
    pu = 'c'  # Процессор: Графический = 0, Центральный = c
    pool = 'max'  # Вид пулинга (средний или максимальный)
    f_mdl = 'models/nyud-fcn32s-color-heavy.pth'  # Предобученная архитектура модели
    dis_check = True  # Отключить проверку
    cnn, lsLayers = loadModel(f_mdl, pool, pu, dis_check)
    img_size = 512  # Максимальные параметры (высота/ширина) сгенерированного изображения

    c_img = preproc(c_img, img_size).type(dtype)
    s_img_input = s_img.split(',')
    s_img_ls, extension = [], [".jpg", ".jpeg", ".png"]
    for image in s_img_input:
        if os.path.isdir(image):
            images = (image + "/" + file for file in os.listdir(image) if
                      os.path.splitext(file)[1].lower() in extension)
            s_img_ls.extend(images)
        else:
            s_img_ls.append(image)
    s_imgs_caf = []
    for image in s_img_ls:
        s_size = int(img_size * s_scl)
        img_caf = preproc(image, s_size).type(dtype)
        s_imgs_caf.append(img_caf)
    init_img = None  # Изображение инициализации
    if init_img is not None:
        img_size = (c_img.size(2), c_img.size(3))
        init_img = preproc(init_img, img_size).type(dtype)

    # Обработка смешанных весов для нескольких стилей
    if s_blend_w == []:
        # Смешивание стилей не задано, поэтому используется равнозначные веса
        for i in s_img_ls:
            s_blend_w.append(1.0)
        for i, bws in enumerate(s_blend_w):
            s_blend_w[i] = int(s_blend_w[i])
    else:
        assert len(s_blend_w) == len(s_img_ls), \
            "-s_blend_w and -s_imgs должны иметь одинаковое количество элементов!"

    # Нормализация весов смешивания стилей по единице
    sum_s_blend = 0
    for i, bws in enumerate(s_blend_w):
        s_blend_w[i] = float(s_blend_w[i])
        sum_s_blend = float(sum_s_blend) + s_blend_w[i]
    for i, bws in enumerate(s_blend_w):
        s_blend_w[i] = float(s_blend_w[i]) / float(sum_s_blend)
    cl = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1'  # Слои исходного изображения
    cl = cl.split(',')
    sl = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1'  # Слои стиля
    sl = sl.split(',')

    # Настройка сети, подключение модулей потери стиля и содержимого
    cnn = copy.deepcopy(cnn)
    c_losses, s_losses, tv_losses = [], [], []
    next_c_idx, next_s_idx = 1, 1  # Индексы карт признаков контента и стиля
    net = nn.Sequential()
    c, r = 0, 0
    tv_w = 1e-5  # Вес сглаживания
    if tv_w > 0:
        tv_mod = LossTV(tv_w).type(dtype)
        net.add_module(str(len(net)), tv_mod)
        tv_losses.append(tv_mod)
    norm_g = True  # Нормализация градиентов
    for i, layer in enumerate(list(cnn), 1):
        if next_c_idx <= len(cl) or next_s_idx <= len(sl):
            if isinstance(layer, nn.Conv2d):
                net.add_module(str(len(net)), layer)

                if lsLayers['Conv'][c] in cl:
                    print("Настройка слоя содержимого " + str(i) + ": " + str(lsLayers['Conv'][c]))
                    loss_module = LossContent(cw, norm_g)
                    net.add_module(str(len(net)), loss_module)
                    c_losses.append(loss_module)

                if lsLayers['Conv'][c] in sl:
                    print("Настройка слоя стиля " + str(i) + ": " + str(lsLayers['Conv'][c]))
                    loss_module = LossStyle(sw, norm_g)
                    net.add_module(str(len(net)), loss_module)
                    s_losses.append(loss_module)
                c += 1

            if isinstance(layer, nn.ReLU):
                net.add_module(str(len(net)), layer)

                if lsLayers['Relu'][r] in cl:
                    print("Настройка слоя содержимого " + str(i) + ": " + str(lsLayers['Relu'][r]))
                    loss_module = LossContent(cw, norm_g)
                    net.add_module(str(len(net)), loss_module)
                    c_losses.append(loss_module)
                    next_c_idx += 1

                if lsLayers['Relu'][r] in sl:
                    print("Настройка слоя стиля " + str(i) + ": " + str(lsLayers['Relu'][r]))
                    loss_module = LossStyle(sw, norm_g)
                    net.add_module(str(len(net)), loss_module)
                    s_losses.append(loss_module)
                    next_s_idx += 1
                r += 1

            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                net.add_module(str(len(net)), layer)

    # Фиксация содержания
    for i in c_losses:
        i.mode = 'capture'
    print("Фиксация содержания")
    # print_torch(net, multidevice)
    net(c_img)

    # Фиксация стилей
    for i in c_losses:
        i.mode = 'None'

    for i, image in enumerate(s_imgs_caf):
        print("Фиксация стиля " + str(i+1))
        for j in s_losses:
            j.mode = 'capture'
            j.bw = s_blend_w[i]
        net(s_imgs_caf[i])

    # Установить все модули потерь к loss mode
    for i in c_losses:
        i.mode = 'loss'
    for i in s_losses:
        i.mode = 'loss'
    norm_w = True  # Нормализация весов
    # Нормализация весов содержания и стиля
    if norm_w is True:
        func_norm_w(c_losses, s_losses)

    # Заморозка сети, для предотвращения лишних вычислений градиента
    for param in net.parameters():
        param.requires_grad = False
    init = 'image'  # Инициализация (случайная или изображение)
    # Инициализация изображения
    if init == 'random':
        b, c, h, w = c_img.size()
        img = torch.randn(c, h, w).mul(0.001).unsqueeze(0).type(dtype)
    elif init == 'image':
        if init_img is not None:
            img = init_img.clone()
        else:
            img = c_img.clone()
    img = nn.Parameter(img)
    slot1 = st.empty()

    def ptProb(t, loss):
        c_loss = 0
        s_loss = 0
        pt_iter = 1  # Итерация на которой печатается прогресс
        if pt_iter > 0 and t % pt_iter == 0:
            print("Итерация " + str(t) + " / " + str(iters))
            slot1.write("Итерация " + str(t) + " / " + str(iters))
            for i, loss_module in enumerate(c_losses):
                c_loss += loss_module.loss.item()
            print("  Потеря контента: " + str(c_loss))
            for i, loss_module in enumerate(s_losses):
                s_loss += loss_module.loss.item()
            print("  Потеря стиля: " + str(s_loss))
            print("  Общая потеря: " + str(loss.item()))

    slot2 = st.empty()
    slot3 = st.empty()

    def svProb(t):
        sv_iter = 2  # Итерация на которой просиходит вывод промежуточного результата
        sv_must = sv_iter > 0 and t % sv_iter == 0
        sv_must = sv_must or t == iters
        if sv_must:
            name_out, ext_out = os.path.splitext(out_img)
            if t == iters:
                name = name_out + str(ext_out)
            else:
                name = str(name_out) + "_" + str(t) + str(ext_out)
            disp = unproc(img.clone())

            # Постобработка для стилизации с сохранением цвета
            if sv_clrs == 1:
                disp = sv_clr(unproc(c_img.clone()), disp)

            disp.save(str(name))
            if t != iters:
                prog1, prog2, prog3 = st.columns(3)
                with prog1:
                    st.write(' ')
                with prog2:
                    slot2.write('### Прогресс ' + str(t) + ' итераций')
                    image = Image.open(name)
                    slot3.image(image, use_column_width=True)
                with prog3:
                    st.write(' ')

    # Функция для вычисления потерь и градиента.
    # Прогонка сети вперед и назад, чтобы получить градиент и посчитать сумму потерь.
    calls = [0]

    def feval():
        calls[0] += 1
        opt.zero_grad()
        net(img)
        loss = 0
        tv_w = 1e-5  # Вес сглаживания

        for mod in c_losses:
            loss += mod.loss.to(backward_dev)
        for mod in s_losses:
            loss += mod.loss.to(backward_dev)
        if tv_w > 0:
            for mod in tv_losses:
                loss += mod.loss.to(backward_dev)

        loss.backward()

        svProb(calls[0])
        ptProb(calls[0], loss)

        return loss

    opt, loopVal = optimConf(img, iters)
    while calls[0] <= loopVal:
        opt.step(feval)
    slot2.empty()
    slot3.empty()


# Конфигурация оптимизатора
def optimConf(img, iters):
    opt = 'lbfgs'  # Оптимизатор
    lr = 1e0  # Шаг обучения
    lbfgs_num_of_corr = 100  # Количество корректировок оптимизатора L-BFGS
    if opt == 'lbfgs':
        print("Оптимизация запущена с оптимизатором L-BFGS")
        optim_state = {
            'max_iter': iters,
            'tolerance_change': -1,
            'tolerance_grad': -1,
        }
        if lbfgs_num_of_corr != 100:
            optim_state['history_size'] = lbfgs_num_of_corr
        opt = optim.LBFGS([img], **optim_state)
        loopVal = 1
    elif opt == 'adam':
        print("Оптимизация запущена с оптимизатором ADAM")
        opt = optim.Adam([img], lr=lr)
        loopVal = iters - 1
    return opt, loopVal


def gpuConf():
    def setup_cuda():
        engine = 'cudnn'  # Движок
        autoconf = True  # Автоматическая настройка
        if 'cudnn' in engine:
            torch.backends.cudnn.enabled = True
            if autoconf:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    pu = 'c'  # Процессор: Графический = 0, Центральный = c
    if "c" not in str(pu).lower():
        setup_cuda()
        dtype, backward_dev = torch.cuda.FloatTensor, "cuda:" + str(pu)
    else:
        # setup_cpu()
        dtype, backward_dev = torch.FloatTensor, "cpu"
    return dtype, backward_dev  # , multidevice


# Предобработка изображения перед передачей его в модель
# Изменение масштаба от [0, 1] к [0, 255], конвертация из RGB в BGR и вычитание среднего пикселя
def preproc(image_name, img_size):
    image = Image.open(image_name).convert('RGB')

    if type(img_size) is not tuple:
        img_size = tuple([int((float(img_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1])])
    tensor = Normalize(rgb2bgr(Loader(image) * 255)).unsqueeze(0)
    return tensor


#  Отмена предобработки
def unproc(output_tensor):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1, 1, 1])])
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 255
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image


# Объединение канала сгенерированного изображения и каналов UV/CbCr изображения содержимого,
# чтобы стилизовать изображение с сохранением цвета.
def sv_clr(content, generated):
    c_channels = list(content.convert('YCbCr').split())
    gen_channels = list(generated.convert('YCbCr').split())
    c_channels[0] = gen_channels[0]
    return Image.merge('YCbCr', c_channels).convert('RGB')


# Деление весов на размер канала
def func_norm_w(c_losses, s_losses):
    for n, i in enumerate(c_losses):
        i.strength = i.strength / max(i.target.size())
    for n, i in enumerate(s_losses):
        i.strength = i.strength / max(i.target.size())


# Масштабирование градиентов при обратном проходе
class GradsScale(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.strength * self.strength, None


# Вычисление потери содержимого
class LossContent(nn.Module):

    def __init__(self, strength, normalize):
        super(LossContent, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = 'None'
        self.normalize = normalize

    def forward(self, input):
        if self.mode == 'loss':
            loss = self.crit(input, self.target)
            if self.normalize:
                loss = GradsScale.apply(loss, self.strength)
            self.loss = loss * self.strength
        elif self.mode == 'capture':
            self.target = input.detach()
        return input


class GramMtx(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()  # B - кол-во изображений в пакете, C - кол-во цветовых каналов, H - высота W - ширина
        x_flat = input.view(c, h * w)
        return torch.mm(x_flat, x_flat.t())


# Вычисление потери стиля
class LossStyle(nn.Module):

    def __init__(self, strength, normalize):
        super(LossStyle, self).__init__()
        self.target = torch.Tensor()
        self.strength = strength
        self.gram = GramMtx()
        self.crit = nn.MSELoss()
        self.mode = 'None'
        self.bw = None
        self.normalize = normalize

    def forward(self, input):
        self.G = self.gram(input)
        self.G = self.G.div(input.nelement())
        if self.mode == 'capture':
            if self.bw is None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.bw)
            else:
                self.target = self.target.add(self.bw, self.G.detach())
        elif self.mode == 'loss':
            loss = self.crit(self.G, self.target)
            if self.normalize:
                loss = GradsScale.apply(loss, self.strength)
            self.loss = self.strength * loss
        return input


class LossTV(nn.Module):

    def __init__(self, strength):
        super(LossTV, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


if __name__ == "__main__":
    main()